"""Runtime machinery that schedules a component according to its execution mode.

:class:`~solarian.core.module.Module` and
:class:`~solarian.core.skill.Skill` describe *what* to do; a
:class:`Runner` decides *when and where* it runs. Both expose the same worker
protocol, so both share these runners:

.. code-block:: text

    worker_setup(reconnect)  -> prepare per-execution-context state
    worker_tick() -> bool    -> do one unit of work; False means "finished"
    worker_teardown(exit)    -> release resources, report final status

The runner owns the loop, the frequency pacing, readiness signalling, crash
capture, and the event sink, so those behave identically whether the component
is a process, a thread, or ticked inline by its host.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import sys
import threading
import time
import traceback as _tb
from abc import ABC, abstractmethod
from collections import deque
from multiprocessing import Event as MpEvent, Process, Queue as MpQueue
from typing import Any, Protocol

from solarian.core.execution import ExecutionMode, WorkerExit
from solarian.utils.configs import ConfigError, config_error_in
from solarian.utils.general import name_process
from solarian.utils.precise_sleep import precise_wait
from solarian.utils.profiling import func_stats, get_stats_snapshot
from solarian.utils.status_board import StatusBoard

logger = logging.getLogger("runner")

_PARENT_POLL_S = 0.5

#: Seconds between the tick summaries a worker process posts to the status board.
_TICK_REPORT_S = 1.0

#: A tick that starts this late (ms) counts as late in the status board summary.
_LATE_MS = 1.0

#: Seconds of each wait spent spinning rather than asleep. Linux wakes a
#: sleeper within about 60 us of its deadline (Pi 5), a sliver of a tick,
#: while a millisecond spin is a tenth of a core per 90 Hz worker.
_SPIN_S = 0.0 if sys.platform == "linux" else 0.001

# Same bound the parent gives a terminated worker: a camera close gives up
# after 2s.
_ORPHAN_TEARDOWN_S = 3.0


def _exit_with_parent(parent_pid: int) -> None:
    """End this worker once the process that started it is gone.

    The stop event is the only shutdown a worker answers to, and it ignores
    Ctrl+C. A parent that dies before stopping it (a second Ctrl+C during
    shutdown, SIGKILL, a crash) would otherwise leave it running for good,
    holding the camera or serial port it opened.
    """
    while os.getppid() == parent_pid:
        time.sleep(_PARENT_POLL_S)
    logger.warning("Parent pid %s is gone; stopping this worker", parent_pid)
    os.kill(os.getpid(), signal.SIGTERM)
    # SIGTERM is ignored once teardown has begun, and a worker blocked in a
    # device call or flushing a queue nobody reads never finishes it.
    time.sleep(_ORPHAN_TEARDOWN_S)
    os.kill(os.getpid(), signal.SIGKILL)


class _TickStats:
    """How long each tick worked, and how late it started against its slot.

    A tick that overruns its period makes the next one late, so lateness is
    what a motor sees as jitter. Both go into the profiling summary; a worker
    process also posts the last second of them to the status board.
    """

    def __init__(self, name: str, report: bool):
        self._work_timings = func_stats[f"tick work {name}"]
        self._late_timings = func_stats[f"tick late {name}"]
        self._label = f"Tick {name} (ms)"
        self._report = report
        self._work: list[float] = []
        self._late_count = 0
        self._next_report = time.monotonic() + _TICK_REPORT_S

    def add(self, work_s: float, late_s: float, now: float) -> None:
        work_ms = work_s * 1e3
        late_ms = max(0.0, late_s * 1e3)
        self._work_timings.add(work_ms)
        self._late_timings.add(late_ms)
        if not self._report:
            return
        self._work.append(work_ms)
        if late_ms > _LATE_MS:
            self._late_count += 1
        if now < self._next_report:
            return
        self._next_report = now + _TICK_REPORT_S
        work = sorted(self._work)
        StatusBoard.set(
            self._label,
            f"work p50 {work[len(work) // 2]:.2f} p99 {work[int(len(work) * 0.99)]:.2f} "
            f"max {work[-1]:.2f}, late {self._late_count}/{len(work)}",
        )
        self._work.clear()
        self._late_count = 0


class Runnable(Protocol):
    """What a :class:`Runner` needs from the component it schedules."""

    name: str
    frequency: float

    def worker_setup(self, reconnect: bool) -> None: ...

    def worker_tick(self) -> bool: ...

    def worker_teardown(self, exit_reason: WorkerExit) -> None: ...


class Runner(ABC):
    """Base scheduler for a :class:`Runnable`.

    Subclasses supply the concurrency primitives (readiness flag, error and
    profiling channels, event sink) and the mechanism that gets
    :meth:`_run_worker` executing in the right context.
    """

    #: Whether the worker body executes in a different process than the caller.
    crosses_process_boundary: bool = False

    def __init__(self, target: Runnable):
        self.target = target
        self._exit_reason = WorkerExit.COMPLETED

    @property
    def name(self) -> str:
        return getattr(self.target, "name", type(self.target).__name__)

    # ========= lifecycle =============
    @abstractmethod
    def start(self, wait: bool = True) -> None:
        """Begin execution. When *wait*, block until the worker is ready."""

    @abstractmethod
    def stop(self, wait: bool = True) -> None:
        """Request a graceful stop, optionally waiting for the worker to exit."""

    @abstractmethod
    def is_alive(self) -> bool: ...

    def terminate(self) -> None:
        """Stop as abruptly as the execution mode allows."""
        self.stop(wait=False)

    def kill(self) -> None:
        """End the worker without waiting for it to unwind."""
        self.terminate()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker to finish."""

    @property
    def is_ready(self) -> bool:
        return self._ready_is_set()

    @property
    def pid(self) -> int | None:
        """OS pid of the worker, or ``None`` when it shares the host process."""
        return None

    @property
    def exitcode(self) -> int | None:
        """Worker exit code, or ``None`` when it shares the host process."""
        return None

    def start_wait(self) -> None:
        """Block until the worker completes its first tick, or raise."""
        self._ready_wait(getattr(self.target, "launch_timeout", 3.0))
        if not self._ready_is_set():
            raise RuntimeError(
                f"'{self.name}' did not become ready within "
                f"launch_timeout={getattr(self.target, 'launch_timeout', 3.0)}s."
            )

    # ========= parent-side channels =============
    def get_error(self) -> tuple[str, str] | None:
        """Return ``(name, traceback)`` if the worker crashed, else ``None``."""
        return None

    def get_profiling_stats(self) -> dict | None:
        """Return the worker's profiling snapshot if it has published one."""
        return None

    def drain_events(self) -> list:
        """Pop everything the component has emitted since the last drain."""
        return []

    @property
    def event_sink(self) -> Any:
        """Where the component's ``emit()`` writes. ``None`` disables events."""
        return None

    # ========= worker-side body =============
    def _run_worker(self) -> None:
        """Setup, loop at the target frequency, tear down. Runs in the worker."""
        target = self.target
        exit_reason = WorkerExit.COMPLETED
        # Set before teardown so a SIGTERM delivered mid-close does not
        # raise out of the close and leave the camera held.
        teardown_started = False

        def _on_sigterm(signum, _frame):
            if teardown_started:
                return
            raise SystemExit(128 + signum)

        if self.crosses_process_boundary:
            name_process(self.name)
            # Ctrl+C reaches every process in the terminal's foreground group,
            # so a worker that acts on it unwinds from inside a device read
            # while the parent is still shutting things down in order, and
            # prints a traceback for a stop that was asked for. The parent's
            # stop event is the only shutdown this worker answers to.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            # The default SIGTERM action skips this finally. The parent's
            # terminate() is a SIGTERM, and a CSI camera stays locked until
            # teardown closes it.
            signal.signal(signal.SIGTERM, _on_sigterm)
            threading.Thread(
                target=_exit_with_parent,
                args=(multiprocessing.parent_process().pid,),
                name="ParentWatchdog",
                daemon=True,
            ).start()
            # A forked worker inherits every timing its parent has taken. The
            # snapshot it queues on exit is read only once it has exited, and
            # a queue holds its process until the pipe has taken the lot, so
            # a snapshot bigger than the pipe keeps the worker alive until it
            # is killed.
            func_stats.clear()
        try:
            target.worker_setup(self.crosses_process_boundary)

            tick_stats = _TickStats(self.name, report=self.crosses_process_boundary)
            t_start = time.monotonic()
            t_slot = t_start
            idx = 0
            keep_running = True
            while keep_running:
                if self._stop_requested():
                    exit_reason = WorkerExit.STOPPED
                    break

                t_tick = time.monotonic()
                keep_running = target.worker_tick()
                t_done = time.monotonic()

                # Ready means "has completed a full cycle", so late failures
                # during the first tick surface as a crash rather than a hang.
                if idx == 0:
                    self._set_ready()
                idx += 1
                tick_stats.add(t_done - t_tick, t_tick - t_slot, t_done)

                period = 1.0 / max(float(getattr(target, "frequency", 0.0) or 0.0), 1e-9)
                t_slot = t_start + period * idx
                if t_done - t_slot > period:
                    # More than a period behind: drop the slots already gone.
                    # Running them back to back pins the process at full CPU
                    # and only adds lateness that the next overrun extends.
                    idx += int((t_done - t_slot) / period)
                    t_slot = t_start + period * idx
                precise_wait(t_end=t_slot, slack_time=_SPIN_S, time_func=time.monotonic)
        except SystemExit:
            exit_reason = WorkerExit.STOPPED
            raise
        except Exception as exc:  # noqa: BLE001
            exit_reason = WorkerExit.FAILED
            tb_str = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
            if getattr(exc, "deliberate", False):
                logger.warning("'%s' failed: %s", self.name, exc)
            else:
                logger.exception("'%s' crashed: %s", self.name, exc)
            self._put_error(tb_str)
        finally:
            teardown_started = True
            self._set_ready()
            self._put_profiling()
            try:
                target.worker_teardown(exit_reason)
            except Exception:  # noqa: BLE001
                logger.exception("'%s' teardown failed", self.name)

    # ========= hooks for subclasses =============
    def _stop_requested(self) -> bool:
        return False

    def _set_ready(self) -> None: ...

    def _ready_is_set(self) -> bool:
        return True

    def _ready_wait(self, timeout: float) -> None: ...

    def _put_error(self, tb_str: str) -> None: ...

    def _put_profiling(self) -> None: ...


class ProcessModuleRunner(Runner):
    """Runs the component in its own OS process.

    Isolates its loop timing from everything else, at the cost of a fork/spawn
    and shared-memory or queue plumbing for anything that has to cross back.
    """

    crosses_process_boundary = True

    def __init__(self, target: Runnable):
        super().__init__(target)
        self._process: Process | None = None
        self._ready = MpEvent()
        self._stop = MpEvent()
        self._error_queue: MpQueue = MpQueue()
        self._profiling_queue: MpQueue = MpQueue()
        self._event_queue: MpQueue = MpQueue()

    def start(self, wait: bool = True) -> None:
        if self._process is not None:
            return
        self._process = Process(target=self._run_worker, name=f"Worker-{self.name}")
        # Skills opt into daemon so they cannot outlive the runtime that
        # launched them; modules are torn down explicitly at shutdown.
        self._process.daemon = bool(getattr(self.target, "daemon", False))
        self._process.start()
        if wait:
            self.start_wait()

    def start_wait(self) -> None:
        timeout = float(getattr(self.target, "launch_timeout", 3.0))
        self._ready.wait(timeout)
        if not self.is_alive():
            error = self.get_error()
            if error:
                worker_name, tb_str = error
                config_error = config_error_in(tb_str)
                if config_error is not None:
                    raise ConfigError(message=f"'{worker_name}': {config_error}")
                raise RuntimeError(
                    f"'{worker_name}' process died during startup:\n{tb_str}"
                )
            raise RuntimeError(
                f"'{self.name}' process died during startup "
                f"(exit code {self.exitcode}). Check stderr for details."
            )
        if not self._ready.is_set():
            # Alive but never reached the first loop iteration — usually a
            # post-fork deadlock (ZMQ / StatusBoard threads in the parent).
            raise RuntimeError(
                f"'{self.name}' process (pid {self.pid}) did not become ready "
                f"within launch_timeout={timeout}s. The child is likely stuck "
                f"after fork; restart and check that the parent is not "
                f"multithreaded before start()."
            )

    def stop(self, wait: bool = True) -> None:
        self._stop.set()
        if wait:
            self.join()

    def terminate(self) -> None:
        self._stop.set()
        if self._process is not None and self._process.is_alive():
            self._process.terminate()

    def kill(self) -> None:
        # SIGKILL cannot run teardown. The caller already gave the process
        # a chance to close its camera; this is what frees the device when
        # that process is stuck.
        if self._process is not None and self._process.is_alive():
            self._process.kill()

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def join(self, timeout: float | None = None) -> None:
        if self._process is not None:
            self._process.join(timeout)

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def exitcode(self) -> int | None:
        return self._process.exitcode if self._process is not None else None

    def get_error(self) -> tuple[str, str] | None:
        try:
            return self._error_queue.get_nowait()
        except Exception:  # noqa: BLE001
            return None

    def get_profiling_stats(self) -> dict | None:
        try:
            return self._profiling_queue.get_nowait()
        except Exception:  # noqa: BLE001
            return None

    @property
    def event_sink(self) -> Any:
        return self._event_queue

    def drain_events(self) -> list:
        events = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except Exception:  # noqa: BLE001
                return events

    def _stop_requested(self) -> bool:
        return self._stop.is_set()

    def _set_ready(self) -> None:
        self._ready.set()

    def _ready_is_set(self) -> bool:
        return self._ready.is_set()

    def _ready_wait(self, timeout: float) -> None:
        self._ready.wait(timeout)

    def _put_error(self, tb_str: str) -> None:
        try:
            self._error_queue.put_nowait((self.name, tb_str))
        except Exception:  # noqa: BLE001
            pass

    def _put_profiling(self) -> None:
        try:
            self._profiling_queue.put_nowait(get_stats_snapshot())
        except Exception:  # noqa: BLE001
            pass


class ThreadModuleRunner(Runner):
    """Runs the component on a daemon thread in the host process.

    For blocking device APIs that must not stall the host loop but do not
    justify a process — no pickling, no shared memory, just the GIL.
    """

    def __init__(self, target: Runnable):
        super().__init__(target)
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error: tuple[str, str] | None = None
        self._profiling: dict | None = None
        self._events: deque = deque()

    def start(self, wait: bool = True) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_worker, name=f"Worker-{self.name}", daemon=True
        )
        self._thread.start()
        if wait:
            self.start_wait()

    def stop(self, wait: bool = True) -> None:
        self._stop.set()
        if wait:
            self.join()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def get_error(self) -> tuple[str, str] | None:
        error, self._error = self._error, None
        return error

    def get_profiling_stats(self) -> dict | None:
        stats, self._profiling = self._profiling, None
        return stats

    @property
    def event_sink(self) -> Any:
        return self._events

    def drain_events(self) -> list:
        events = []
        while self._events:
            try:
                events.append(self._events.popleft())
            except IndexError:
                break
        return events

    def _stop_requested(self) -> bool:
        return self._stop.is_set()

    def _set_ready(self) -> None:
        self._ready.set()

    def _ready_is_set(self) -> bool:
        return self._ready.is_set()

    def _ready_wait(self, timeout: float) -> None:
        self._ready.wait(timeout)

    def _put_error(self, tb_str: str) -> None:
        self._error = (self.name, tb_str)

    def _put_profiling(self) -> None:
        try:
            self._profiling = get_stats_snapshot()
        except Exception:  # noqa: BLE001
            pass


class InlineModuleRunner(Runner):
    """Runs the component on the host's own loop, one tick per host tick.

    There is no worker body and no pacing of its own: :meth:`start` prepares
    the component and the host calls :meth:`tick`. Cheapest possible option
    for event-driven components, and it keeps their state changes on the
    host's thread, so the host never reads half-updated state.
    """

    def __init__(self, target: Runnable):
        super().__init__(target)
        self._started = False
        self._finished = False
        self._error: tuple[str, str] | None = None
        self._events: deque = deque()

    def start(self, wait: bool = True) -> None:
        if self._started:
            return
        self.target.worker_setup(self.crosses_process_boundary)
        self._started = True

    def tick(self) -> None:
        """Advance the component one step. Called by the host's control loop."""
        if not self._started or self._finished:
            return
        try:
            if not self.target.worker_tick():
                self._finish(WorkerExit.COMPLETED)
        except Exception as exc:  # noqa: BLE001
            tb_str = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
            if getattr(exc, "deliberate", False):
                logger.warning("Inline '%s' failed: %s", self.name, exc)
            else:
                logger.exception("Inline '%s' crashed: %s", self.name, exc)
            self._put_error(tb_str)
            self._finish(WorkerExit.FAILED)

    def stop(self, wait: bool = True) -> None:
        self._finish(WorkerExit.STOPPED)

    def _finish(self, exit_reason: WorkerExit) -> None:
        if not self._started or self._finished:
            return
        self._finished = True
        try:
            self.target.worker_teardown(exit_reason)
        except Exception:  # noqa: BLE001
            logger.exception("Inline '%s' teardown failed", self.name)

    def is_alive(self) -> bool:
        return self._started and not self._finished

    @property
    def is_ready(self) -> bool:
        return self._started

    def start_wait(self) -> None:
        return None

    def get_error(self) -> tuple[str, str] | None:
        error, self._error = self._error, None
        return error

    @property
    def event_sink(self) -> Any:
        return self._events

    def drain_events(self) -> list:
        events = []
        while self._events:
            try:
                events.append(self._events.popleft())
            except IndexError:
                break
        return events

    def _put_error(self, tb_str: str) -> None:
        self._error = (self.name, tb_str)


_RUNNERS: dict[ExecutionMode, type[Runner]] = {
    ExecutionMode.PROCESS: ProcessModuleRunner,
    ExecutionMode.THREAD: ThreadModuleRunner,
    ExecutionMode.INLINE: InlineModuleRunner,
}


def make_runner(
    target: Runnable, execution_mode: ExecutionMode | str | None = None
) -> Runner:
    """Build the runner for *target*, defaulting to its own execution mode."""
    if execution_mode is None:
        execution_mode = getattr(
            target, "execution_mode", ExecutionMode.PROCESS
        )
    return _RUNNERS[ExecutionMode(execution_mode)](target)
