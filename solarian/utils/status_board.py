from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from solarian.core.action import Action, ModuleAction
    from solarian.core.observation import Observation, ModuleObs

from rich.live import Live
from rich.table import Table
from rich.console import Console, Group
from rich.pretty import Pretty
from rich.json import JSON
from rich import print as rprint
import multiprocessing
import sys
import threading
import traceback
import time
import os
import queue
import numpy as np
from rich.panel import Panel
from rich.console import RenderableType
from datetime import datetime
from solarian.utils.log_buffer import LogBuffer, get_log_queue, remove_fallback_handlers



def format_floats(obj, fmt="{:.3f}"):
    if isinstance(obj, float):
        return fmt.format(obj)        # keep as str to preserve formatting
    if isinstance(obj, dict):
        return {k: format_floats(v, fmt) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [format_floats(x, fmt) for x in obj]
    if isinstance(obj, (int, bool, str, type(None))):
        return obj
    return str(obj)


# def render_action(action: Action, *, max_rows: int = 6) -> RenderableType:
#     d = action.to_dict()
#     data = d.get("data", {})
#     action_type = d.get("action_type", "?")

#     # Pull a timestamp once (if consistent)
#     timestamp = None
#     if data:
#         first = next(iter(data.values()))
#         timestamp = first.get("timestamp")

#     inner = Table(box=None, show_header=False, pad_edge=False)
#     inner.add_column("joint", style="cyan", no_wrap=True)
#     inner.add_column("val", justify="right")

#     items = list(data.items())
#     for joint, joint_data in items[:max_rows]:
#         inner.add_row(str(joint), str(joint_data.get("value", "")))

#     if len(items) > max_rows:
#         inner.add_row("…", f"+{len(items) - max_rows} more")

#     title = f"Action type={action_type}"
#     subtitle = f"t={timestamp}" if timestamp is not None else None

#     return Panel(inner, title=title, subtitle=subtitle, border_style="blue")


from dataclasses import dataclass
from typing import Any, Mapping, Optional

from rich.console import RenderableType
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text


def _module_timestamp_label(data: Mapping[str, Any]) -> Optional[str]:
    """Return 't=<ts>' if all timestamps match, 't=mixed' if not, or None if missing."""
    ts = []
    for v in data.values():
        if isinstance(v, Mapping) and "timestamp" in v and v["timestamp"] is not None:
            ts.append(str(v["timestamp"]))
    if not ts:
        return None
    return f"t={ts[0]}" if len(set(ts)) == 1 else "t=mixed"


def _action_type_label(action_type: Any) -> str:
    """Format an action_type as 'Name=value' using the ActionType registry."""
    if action_type is None:
        return ""
    try:
        from solarian.core.action import ActionType
        value = int(action_type)
    except (TypeError, ValueError):
        return f"type={action_type}"
    name = ActionType._registry.get(value)
    if name:
        return f"{name}={value}"
    return f"type={value}"


def _build_fields_renderable(
    items: list,
    *,
    max_rows: int,
) -> tuple[RenderableType, Optional[Any]]:
    """Render a list of (field, payload) items as one or more side-by-side
    tables so long field lists overflow into additional columns instead of
    being truncated under '+N more'.

    Returns the renderable plus the most recent timestamp seen (or None).
    """
    last_timestamp = None
    total = len(items)
    if total == 0:
        inner = Table(box=None, show_header=False, pad_edge=False)
        inner.add_column("field", style="cyan", no_wrap=True)
        inner.add_column("value", justify="right")
        return inner, None

    num_cols = max(1, (total + max_rows - 1) // max_rows)
    rows_per_col = (total + num_cols - 1) // num_cols

    sub_tables: list[Table] = []
    for col_idx in range(num_cols):
        start = col_idx * rows_per_col
        end = min(start + rows_per_col, total)
        sub = Table(box=None, show_header=False, pad_edge=False)
        sub.add_column("field", style="cyan", no_wrap=True)
        sub.add_column("value", justify="right")
        for field, field_payload in items[start:end]:
            if isinstance(field_payload, Mapping):
                val = field_payload.get("value", "")
                if field_payload.get("timestamp") is not None:
                    last_timestamp = field_payload.get("timestamp")
            else:
                val = field_payload
            sub.add_row(str(field), value_str(val))
        sub_tables.append(sub)

    if len(sub_tables) == 1:
        return sub_tables[0], last_timestamp
    return Columns(sub_tables, expand=True, equal=False, padding=(0, 2)), last_timestamp


def _render_action_panel(
    module_name: str,
    module_payload: Mapping[str, Any],
    *,
    max_rows: int,
) -> RenderableType:
    data = module_payload.get("data") or {}
    action_type = module_payload.get("action_type", None)

    items = list(data.items()) if isinstance(data, Mapping) else []
    inner, last_timestamp = _build_fields_renderable(items, max_rows=max_rows)

    title_parts = [module_name]
    label = _action_type_label(action_type)
    if label:
        title_parts.append(label)
    title = "  ".join(title_parts)

    subtitle = None
    if last_timestamp is not None:
        dt = datetime.fromtimestamp(float(last_timestamp))
        subtitle = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    return Panel(inner, title=title, subtitle=subtitle, border_style="blue")


def _render_observation_panel(
    module_name: str,
    module_payload: Mapping[str, Any],
    *,
    max_rows: int,
) -> RenderableType:
    data = module_payload

    items = list(data.items()) if isinstance(data, Mapping) else []
    inner, last_timestamp = _build_fields_renderable(items, max_rows=max_rows)

    title = module_name

    subtitle = None
    if last_timestamp is not None:
        dt = datetime.fromtimestamp(float(last_timestamp))
        subtitle = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    return Panel(inner, title=title, subtitle=subtitle, border_style="blue")


def _render_raw_action_panel(
    action_type: Any,
    payload: Any,
    *,
    max_rows: int,
) -> RenderableType:
    """Render an Action whose payload is NOT MODULE_ACTION-shaped.

    Non-module actions (e.g. skill-launch actions) carry arbitrary JSON-y
    params in ``action.data`` instead of the nested
    ``{module: {"data": {...}, "action_type": ...}}`` structure used by
    MODULE_ACTION. Render them as a flat key/value table.
    """
    if isinstance(payload, Mapping):
        items = list(payload.items())
        inner, _ = _build_fields_renderable(items, max_rows=max_rows)
    else:
        inner = Table(box=None, show_header=False, pad_edge=False)
        inner.add_column("field", style="cyan", no_wrap=True)
        inner.add_column("value", justify="right")
        inner.add_row("value", value_str(payload))

    label = _action_type_label(action_type)
    title = f"action  {label}" if label else "action"
    return Panel(inner, title=title, border_style="magenta")


def render_action_or_observation(
    action: Action | Observation,
    *,
    max_rows: int = 6,
    module_order: Optional[list[str]] = None,
    max_modules: Optional[int] = None,
) -> RenderableType:
    """
    Render an Action with multiple modules (arm/base/...) into a compact Rich renderable
    suitable for a single Table cell.
    """
    from solarian.core.action import Action, ActionType
    from solarian.core.observation import Observation

    action.uncompress()
    if isinstance(action, Action):
        action_dict = action.to_dict()
        action_type = action_dict.get("action_type")
        payload = action_dict["data"]
        # Non-MODULE_ACTION payloads are free-form dicts (skill-launch
        # params etc.) — don't try to interpret them as per-module actions.
        if action_type != int(ActionType.MODULE_ACTION):
            return _render_raw_action_panel(action_type, payload, max_rows=max_rows)
    else:
        payload = action.to_dict()

    # Decide module iteration order
    module_names = list(payload.keys()) if isinstance(payload, Mapping) else []
    if module_order:
        order = [m for m in module_order if m in module_names]
        rest = [m for m in module_names if m not in order]
        module_names = order + rest
    else:
        module_names = sorted(module_names)

    if max_modules is not None:
        module_names = module_names[:max_modules]

    panels: list[RenderableType] = []
    for name in module_names:
        module_payload = payload[name]
        if isinstance(action, Action):
            if not isinstance(module_payload, Mapping):
                # Defensive fallback: malformed module payload.
                panels.append(_render_raw_action_panel(name, module_payload, max_rows=max_rows))
                continue
            panels.append(_render_action_panel(name, module_payload, max_rows=max_rows))
        elif isinstance(action, Observation):
            panels.append(_render_observation_panel(name, module_payload, max_rows=max_rows))
        else:
            raise ValueError(f"Unknown action type: {type(action)}")

    if not panels:
        return Text("<empty action>", style="dim")

    # In the status table cell, horizontal Columns can crop later panels.
    # Keep actions compact, but stack observations vertically so every
    # module (e.g. wrist_camera) remains visible.
    if isinstance(action, Observation):
        return Group(*panels)
    return Columns(panels, expand=True, equal=True)


def render_module_obs(
    module_obs: ModuleObs,
    *,
    max_rows: int = 6,
    name: Optional[str] = None,
) -> RenderableType:
    module_obs.uncompress()
    payload = module_obs.to_dict()
    title = name or type(module_obs).__name__
    return _render_observation_panel(title, payload, max_rows=max_rows)


def render_module_action(
    module_action: ModuleAction,
    *,
    max_rows: int = 6,
    name: Optional[str] = None,
) -> RenderableType:
    module_action.uncompress()
    payload = module_action.to_dict()
    title = name or type(module_action).__name__
    return _render_action_panel(title, payload, max_rows=max_rows)


def value_str(value):
    from solarian.core.action import Action, ModuleAction
    from solarian.core.observation import Observation, ModuleObs

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return f"{value.item():.3f}"
        if value.size <= 10:
            with np.printoptions(precision=3, suppress=True):
                return str(value)
        return f"<numpy.ndarray> shape={value.shape}"
    elif isinstance(value, bytes):
        return "<bytes>"
    elif isinstance(value, (float, np.floating)):
        return f"{value:.2f}"
    elif type(value) == list:
        return "[" + ", ".join([f"{x:.0f}" if isinstance(x, (float, np.floating)) else str(x) for x in value]) + "]"
    elif type(value) == dict:
        if value and all(
            isinstance(v, (int, float, np.integer, np.floating, type(None)))
            for v in value.values()
        ):
            inner = Table(box=None, show_header=False, pad_edge=False)
            inner.add_column("key", style="cyan", no_wrap=True)
            inner.add_column("val", justify="right")
            for k, v in value.items():
                if isinstance(v, (int, np.integer)):
                    inner.add_row(str(k), str(int(v)))
                elif isinstance(v, (float, np.floating)):
                    inner.add_row(str(k), f"{float(v):.2f}")
                else:
                    inner.add_row(str(k), str(v))
            return inner
        return JSON.from_data(format_floats(value, "{:.3f}"), indent=2)
    elif isinstance(value, Action) or isinstance(value, Observation):
        return render_action_or_observation(value)
    elif isinstance(value, ModuleObs):
        return render_module_obs(value)
    elif isinstance(value, ModuleAction):
        return render_module_action(value)
    else:
        return str(value)


class StatusBoard:
    _instance = None
    _lock = threading.Lock()
    _live: bool = True
    _child_q: "multiprocessing.Queue | None" = None

    @classmethod
    def configure(cls, *, live: bool = True):
        """Set display mode before the board is first instantiated.

        Args:
            live: If True (default), use Rich Live for in-place refresh.
                  If False, print status snapshots sequentially to stdout.
        """
        cls._live = live

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StatusBoard, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, *, start_ui: bool = True):
        if not hasattr(self, '_initialized') or not self._initialized:
            self.data = {}
            self._local_q = queue.Queue()
            self._mp_q = multiprocessing.Queue()
            self._log_q = get_log_queue()
            self._log_buf = LogBuffer(maxlen=1500)
            self._creator_pid = os.getpid()
            self._stop_event = threading.Event()
            self._stopped_event = threading.Event()
            self._console = Console()
            self._ui_started = False
            self._ui_thread: threading.Thread | None = None
            self.table_title = "Status"
            self._initialized = True

            remove_fallback_handlers()

            if start_ui:
                self._start_ui_thread()

    def _start_ui_thread(self) -> None:
        thread = getattr(self, "_ui_thread", None)
        if thread is not None and thread.is_alive():
            return
        # Allow a second start after stop() (e.g. solarian calibrate sweep,
        # then later run-robot). Reset the events the previous loop flipped.
        self._stop_event = threading.Event()
        self._stopped_event = threading.Event()
        self._ui_started = True
        target = self.ui_loop if self._live else self.ui_loop_plain
        self._ui_thread = threading.Thread(target=target, daemon=True)
        self._ui_thread.start()

    @classmethod
    def start_ui(cls) -> "StatusBoard":
        """Start the display thread (safe to call more than once).

        Robot forking must happen *before* this runs: the UI thread plus any
        inherited ZMQ I/O threads are a common cause of post-``fork`` deadlocks
        in module child processes.
        """
        # Do not hold `_lock` across StatusBoard() — `__new__` acquires the same
        # non-reentrant lock and would deadlock on first construction.
        with cls._lock:
            inst = cls._instance
            needs_create = inst is None or not getattr(inst, "_initialized", False)
        if needs_create:
            return StatusBoard(start_ui=True)
        with cls._lock:
            inst = cls._instance
            inst._start_ui_thread()
            return inst

    @classmethod
    def get_mp_queue(cls) -> "multiprocessing.Queue | None":
        """Return the multiprocessing queue for passing to child processes."""
        inst = cls._instance
        if inst is not None and hasattr(inst, '_mp_q'):
            return inst._mp_q
        return None

    @classmethod
    def init_child(cls, mp_q: "multiprocessing.Queue"):
        """Register the parent's mp queue so a child process can send updates."""
        cls._child_q = mp_q

    @classmethod
    def set(cls, key, value):
        inst = cls._instance
        if inst is not None and hasattr(inst, '_initialized') and inst._initialized:
            if os.getpid() == inst._creator_pid:
                inst._local_q.put((key, value))
            else:
                inst._mp_q.put((key, value))
        elif cls._child_q is not None:
            cls._child_q.put((key, value))
        elif multiprocessing.parent_process() is None:
            # Defer the UI thread so early set() calls (e.g. clock sync during
            # Robot.connect) do not make the parent multithreaded before module
            # processes are forked.
            inst = StatusBoard(start_ui=False)
            inst._local_q.put((key, value))

    @classmethod
    def get(cls):
        inst = cls._instance
        if inst is None:
            return None, None
        try:
            return inst._local_q.get_nowait()
        except Exception:
            return None, None

    @classmethod
    def stop(cls):
        """Stop the live display so exceptions/tracebacks render cleanly.

        Safe to call when the UI was never started (no-op, no timeout).
        After this returns the display thread is dead, so a later
        :meth:`start_ui` can spawn a new one.
        """
        inst = cls._instance
        if inst is None:
            return
        thread = getattr(inst, "_ui_thread", None)
        if thread is None or not thread.is_alive():
            inst._stopped_event.set()
            return
        inst._stop_event.set()
        inst._stopped_event.wait(timeout=5.0)
        thread.join(timeout=1.0)

    @classmethod
    def print_exception(cls, exc: BaseException | None = None):
        """Stop the live display and print an exception traceback cleanly.

        The traceback is captured eagerly (before stopping the display) so
        that ``sys.exc_info()`` is still available in the calling thread.
        Output goes to the *same* console / stream the StatusBoard uses so
        that Rich's ANSI cursor movements from the Live display cannot
        overwrite the error message.
        """
        from rich.traceback import Traceback
        from solarian.utils.profiling import suppress_atexit_summary

        suppress_atexit_summary()

        if exc is not None:
            tb = Traceback.from_exception(
                type(exc), exc, exc.__traceback__, show_locals=False,
            )
        elif sys.exc_info()[1] is not None:
            tb = Traceback(show_locals=False)
        else:
            tb = None

        cls.stop()

        inst = cls._instance
        if inst is not None and inst._stopped_event.is_set():
            console = inst._console
        else:
            console = Console()

        console.print()
        if tb is not None:
            console.print(tb)
        else:
            traceback.print_exc()

    def render(self):
        t = Table(title=getattr(self, "table_title", "Status"), expand=True, show_edge=False)
        t.add_column("State"); t.add_column("Value")
        t.add_row("Local time", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for k, v in self.data.items():
            t.add_row(k, value_str(v))
        return Group(t, self._log_buf.render())

    def _drain_queues(self):
        try:
            while True:
                k, v = self._local_q.get_nowait()
                self.data[k] = v
        except queue.Empty:
            pass
        try:
            while True:
                k, v = self._mp_q.get_nowait()
                self.data[k] = v
        except queue.Empty:
            pass
        return self._log_buf.drain(self._log_q)

    def ui_loop(self):
        try:
            with Live(self.render(), console=self._console, auto_refresh=False) as live:
                while not self._stop_event.is_set():
                    new_logs = self._drain_queues()
                    for rec in new_logs:
                        live.console.print(self._log_buf.format_record(rec))
                    live.update(self.render())
                    live.refresh()
                    time.sleep(0.05)
                # Final drain + render so the last snapshot is fully up-to-date
                # before Live.__exit__ prints it permanently.
                self._drain_queues()
                live.update(self.render())
                live.refresh()
        finally:
            self._stopped_event.set()

    def ui_loop_plain(self):
        try:
            while not self._stop_event.is_set():
                new_logs = self._drain_queues()
                for rec in new_logs:
                    self._console.print(self._log_buf.format_record(rec))
                self._console.print(self.render())
                self._console.print()
                time.sleep(1.0)
        finally:
            self._stopped_event.set()



# board = StatusBoard()

