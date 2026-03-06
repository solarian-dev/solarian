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
from solarian.core.action import Action, ModuleAction
from solarian.core.observation import Observation, ModuleObs
from datetime import datetime
from solarian.utils.log_buffer import LogBuffer, get_log_queue



def format_floats(obj, fmt="{:.3f}"):
    if isinstance(obj, float):
        return fmt.format(obj)        # keep as str to preserve formatting
    if isinstance(obj, dict):
        return {k: format_floats(v, fmt) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [format_floats(x, fmt) for x in obj]
    return obj


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


def _render_action_panel(
    module_name: str,
    module_payload: Mapping[str, Any],
    *,
    max_rows: int,
) -> RenderableType:
    data = module_payload.get("data") or {}
    action_type = module_payload.get("action_type", None)

    inner = Table(box=None, show_header=False, pad_edge=False)
    inner.add_column("field", style="cyan", no_wrap=True)
    inner.add_column("value", justify="right")

    # Only render field rows if it looks like field -> mapping with "value"
    items = list(data.items()) if isinstance(data, Mapping) else []
    shown = 0
    last_timestamp = None
    for field, field_payload in items[:max_rows]:
        if isinstance(field_payload, Mapping):
            val = field_payload.get("value", "")
            if field_payload.get("timestamp") is not None:
                last_timestamp = field_payload.get("timestamp")
        else:
            # fallback if structure isn't as expected
            val = field_payload
        inner.add_row(str(field), value_str(val))
        shown += 1

    if len(items) > max_rows:
        inner.add_row("…", f"+{len(items) - max_rows} more")

    # Title/subtitle
    title_parts = [module_name]
    if action_type is not None:
        title_parts.append(f"type={action_type}")
    title = "  ".join(title_parts)

    subtitle = None
    if last_timestamp is not None:
        dt = datetime.fromtimestamp(last_timestamp)
        subtitle = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    return Panel(inner, title=title, subtitle=subtitle, border_style="blue")


def _render_observation_panel(
    module_name: str,
    module_payload: Mapping[str, Any],
    *,
    max_rows: int,
) -> RenderableType:
    data = module_payload

    inner = Table(box=None, show_header=False, pad_edge=False)
    inner.add_column("field", style="cyan", no_wrap=True)
    inner.add_column("value", justify="right")

    # Only render field rows if it looks like field -> mapping with "value"
    items = list(data.items()) if isinstance(data, Mapping) else []
    shown = 0
    last_timestamp = None
    for field, field_payload in items[:max_rows]:
        if isinstance(field_payload, Mapping):
            val = field_payload.get("value", "")
            if field_payload.get("timestamp") is not None:
                last_timestamp = field_payload.get("timestamp")
        else:
            # fallback if structure isn't as expected
            val = field_payload
        inner.add_row(str(field), value_str(val))
        shown += 1

    if len(items) > max_rows:
        inner.add_row("…", f"+{len(items) - max_rows} more")

    # Title/subtitle
    title_parts = [module_name]
    title = "  ".join(title_parts)

    subtitle = None
    if last_timestamp is not None:
        dt = datetime.fromtimestamp(last_timestamp)
        subtitle = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    return Panel(inner, title=title, subtitle=subtitle, border_style="blue")


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
    action.uncompress()
    payload = action.to_dict()

    # Decide module iteration order
    module_names = list(payload.keys())
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
        if isinstance(action, Action):
            panels.append(_render_action_panel(name, payload[name], max_rows=max_rows))
        elif isinstance(action, Observation):
            panels.append(_render_observation_panel(name, payload[name], max_rows=max_rows))
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
    if isinstance(value, np.ndarray):
        return f"<numpy.ndarray> shape={value.shape}"
    elif isinstance(value, bytes):
        return "<bytes>"
    elif isinstance(value, (float, np.floating)):
        return f"{value:.2f}"
    elif type(value) == list:
        return "[" + ", ".join([f"{x:.0f}" if isinstance(x, (float, np.floating)) else str(x) for x in value]) + "]"
    elif type(value) == dict:
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

    @classmethod
    def configure(cls, *, live: bool = True):
        """Set display mode before the board is first instantiated.

        Args:
            live: If True (default), use Rich Live for in-place refresh.
                  If False, print status snapshots sequentially to stdout.
        """
        cls._live = live

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StatusBoard, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
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
            self._initialized = True

            target = self.ui_loop if self._live else self.ui_loop_plain
            threading.Thread(target=target, daemon=True).start()

    @classmethod
    def set(cls, key, value):
        inst = cls._instance
        if inst is None or not hasattr(inst, '_initialized') or not inst._initialized:
            inst = StatusBoard()
        if os.getpid() == inst._creator_pid:
            inst._local_q.put((key, value))
        else:
            inst._mp_q.put((key, value))

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
        """Stop the live display so exceptions/tracebacks render cleanly."""
        inst = cls._instance
        if inst is None:
            return
        inst._stop_event.set()
        inst._stopped_event.wait(timeout=2.0)

    @classmethod
    def print_exception(cls, exc: BaseException | None = None):
        """Stop the live display and print an exception traceback cleanly.
        
        Must be called from within an ``except`` block so that
        ``sys.exc_info()`` is available (or pass *exc* explicitly).
        """
        cls.stop()
        console = Console(stderr=True)
        console.print()
        if exc is not None:
            console.print_exception(show_locals=False)
        elif sys.exc_info()[1] is not None:
            console.print_exception(show_locals=False)
        else:
            traceback.print_exc()

    def render(self):
        t = Table(title="Status", expand=True, show_edge=False)
        t.add_column("State"); t.add_column("Value")
        t.add_row("Local time", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for k, v in self.data.items():
            t.add_row(k, value_str(v))
        return t

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
        with Live(self.render(), console=self._console, auto_refresh=False) as live:
            while not self._stop_event.is_set():
                new_logs = self._drain_queues()
                for rec in new_logs:
                    live.console.print(self._log_buf.format_record(rec))
                live.update(self.render())
                live.refresh()
                time.sleep(0.05)
        self._stopped_event.set()

    def ui_loop_plain(self):
        while not self._stop_event.is_set():
            new_logs = self._drain_queues()
            for rec in new_logs:
                self._console.print(self._log_buf.format_record(rec))
            self._console.print(self.render())
            self._console.print()
            time.sleep(1.0)
        self._stopped_event.set()



# board = StatusBoard()

