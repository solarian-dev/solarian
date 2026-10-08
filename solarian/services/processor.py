"""ServiceProcessor — a calculated field whose value comes from a service.

The host (relay server, robot module or controller) calls a processor inline,
once per observation, on the path that forwards that observation. A model
behind HTTP takes tens of milliseconds per call, so the call cannot happen
there: a worker thread owns the client, takes the newest frame from a one-slot
buffer, and publishes its result. ``__call__`` only swaps the frame in and
returns the latest result, so the field lags the image by one round trip
instead of the observation stream slowing to the service's rate.

Subclasses implement :meth:`on_start`, :meth:`on_frame` and optionally
:meth:`on_stop`; all three run on the worker thread.
"""

import logging
import threading
import time
from typing import Any

from solarian.core.clock_sync import synced_time
from solarian.perception.processor import Processor
from solarian.utils import StatusBoard

logger = logging.getLogger(__name__)

_RETRY_BACKOFF_S = 0.5


class ServiceProcessor(Processor):
    def __init__(self, inputs, output, conf: str | None = None):
        super().__init__(inputs, output, conf)
        self.label = str(self.cfg.get("label", type(self).__name__)) if self.cfg else type(self).__name__
        self._cond = threading.Condition()
        #: Bumped by every start/stop so a result computed for an earlier
        #: request is discarded rather than published after it.
        self._generation = 0
        self._pending_start: dict | None = None
        self._pending_stop = False
        self._active = False
        self._frame: tuple[Any, float] | None = None
        #: ``(value, timestamp of the frame it was computed from)``.
        self._result: tuple[Any, float] | None = None
        self._worker: threading.Thread | None = None

    # -- hooks, run on the worker thread ------------------------------------

    def on_start(self, frame, params: dict):
        """Initialise on ``frame`` from ``params``; return the first result or None."""
        raise NotImplementedError

    def on_frame(self, frame):
        """Return the result for ``frame``, or None to keep the previous one."""
        raise NotImplementedError

    def on_stop(self) -> None:
        """Release whatever :meth:`on_start` set up on the server."""

    # -- control, from any thread -------------------------------------------

    @property
    def active(self) -> bool:
        return self._active or self._pending_start is not None

    def start(self, params: dict) -> None:
        with self._cond:
            self._generation += 1
            self._pending_start = dict(params)
            self._result = None
            self._ensure_worker()
            self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._generation += 1
            self._pending_start = None
            self._pending_stop = self._active
            self._active = False
            self._result = None
            self._frame = None
            self._cond.notify()

    # -- Processor interface --------------------------------------------------

    def __call__(self, obs):
        if self._input_field_names is None:
            self._resolve_input_fields(obs)
            assert self._input_field_names is not None
        # Absent when the robot's preview throttle dropped this message's image.
        source = obs.get(self._input_field_names[0])
        with self._cond:
            if self.active and source is not None:
                self._frame = (source, float(source.timestamp))
                self._cond.notify()
            result = self._result
        out = self.output.zeros()
        if result is None:
            out.timestamp = float(source.timestamp) if source is not None else synced_time()
            return out
        out.value, out.timestamp = result
        return out

    # -- worker ---------------------------------------------------------------

    def _ensure_worker(self) -> None:
        # Started on first use rather than in __init__: the robot builds its
        # processors before forking the module processes, and a thread does
        # not survive the fork.
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._run, name=f"{self.label}-worker", daemon=True
            )
            self._worker.start()

    def _next_job(self):
        with self._cond:
            while True:
                if self._pending_stop:
                    self._pending_stop = False
                    return "stop", None, None, self._generation
                if self._pending_start is not None and self._frame is not None:
                    params, self._pending_start = self._pending_start, None
                    frame, self._frame = self._frame, None
                    self._active = True
                    return "start", frame, params, self._generation
                if self._active and self._frame is not None:
                    frame, self._frame = self._frame, None
                    return "frame", frame, None, self._generation
                self._cond.wait()

    def _run(self) -> None:
        while True:
            kind, frame, params, generation = self._next_job()
            try:
                if kind == "stop":
                    self.on_stop()
                    continue
                source, frame_ts = frame
                t0 = time.perf_counter()
                value = (
                    self.on_start(source.value, params)
                    if kind == "start"
                    else self.on_frame(source.value)
                )
                call_ms = (time.perf_counter() - t0) * 1000.0
            except Exception:  # noqa: BLE001 - one bad frame must not end tracking
                logger.exception("%s: service call failed", self.label)
                if kind == "start":
                    with self._cond:
                        if generation == self._generation:
                            self._pending_start = params
                            self._active = False
                # A server that is down would otherwise be retried, and
                # logged, at camera rate.
                time.sleep(_RETRY_BACKOFF_S)
                continue
            if value is None:
                continue
            with self._cond:
                if generation != self._generation:
                    continue
                self._result = (value, frame_ts)
            StatusBoard.set(f"{self.label} call (ms)", round(call_ms, 1))
            StatusBoard.set(
                f"{self.label} age (ms)", round((synced_time() - frame_ts) * 1000.0, 1)
            )
