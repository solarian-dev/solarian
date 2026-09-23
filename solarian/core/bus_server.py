"""
Shared-bus server -- lets multiple module processes share a single hardware port.

When two or more modules reference the same port (e.g. an arm and a mobile base
on the same Feetech serial bus), a ``BusServer`` process owns the real
``ComponentBus`` connection while each module gets a ``BusClient`` proxy that
forwards every read/write over a ``multiprocessing.Pipe``.

The serial bus is inherently half-duplex, so requests from different modules are
served sequentially -- one module simply waits while the other's operation
completes.  The IPC overhead (~0.1 ms per round-trip) is negligible compared to
the serial I/O itself (~1-2 ms).
"""

from __future__ import annotations

import time
import logging
import traceback as _tb
from multiprocessing import Process, Event, Queue as MpQueue
from multiprocessing.connection import Connection, Pipe, wait as conn_wait
from typing import Any, Dict, cast, TYPE_CHECKING

from solarian.core.component_bus import ComponentBus
from solarian.core.data import Data

if TYPE_CHECKING:
    from solarian.core.component import Component
    from solarian.core.calibration import MotorCalibration

logger = logging.getLogger("bus_server")

_SHUTDOWN_SENTINEL = "__shutdown__"


# ======================================================================
# Server
# ======================================================================

class BusServer(Process):
    """Process that owns a ``ComponentBus`` and serves RPC requests from
    one or more ``BusClient`` instances via ``multiprocessing.Pipe``.
    """

    def __init__(
        self,
        port: str,
        bus_class: type,
        components: Dict[str, "Component"],
        calibration: Dict[str, "MotorCalibration"] | None,
        server_conns: Dict[str, Connection],
    ):
        super().__init__(daemon=True)
        self._port = port
        self._bus_class = bus_class
        self._components = components
        self._calibration = calibration
        # server-side pipe end per module: {module_name: Connection}
        self._server_conns = server_conns
        self.ready_event = Event()
        self._error_queue: MpQueue = MpQueue()

        from solarian.utils import get_log_queue
        self._log_queue = get_log_queue()

    def get_error(self) -> tuple[str, str] | None:
        try:
            return self._error_queue.get_nowait()
        except Exception:
            return None

    # ------------------------------------------------------------------

    def run(self):
        from solarian.utils import setup_process_logging

        setup_process_logging(logger_name=None, log_queue=self._log_queue, install_fallback=False)
        bus = None

        try:
            bus = self._bus_class(
                port=self._port,
                components=self._components,
                calibration=self._calibration,
            )
            bus.connect()
            logger.info(
                f"BusServer on {self._port}: {self._bus_class.__name__} "
                f"with {len(self._components)} components"
            )
            self.ready_event.set()

            conn_list = list(self._server_conns.values())

            while True:
                # Block until at least one client sends a request
                try:
                    ready = cast(list[Connection], conn_wait(conn_list, timeout=1.0))
                except OSError:
                    break

                for conn in ready:
                    try:
                        method, args, kwargs = conn.recv()
                    except (EOFError, OSError):
                        # Client disconnected
                        conn_list = [c for c in conn_list if c is not conn]
                        continue

                    if method == _SHUTDOWN_SENTINEL:
                        return

                    try:
                        result = getattr(bus, method)(*args, **kwargs)
                        conn.send((result, None))
                    except Exception as exc:
                        try:
                            conn.send((None, exc))
                        except Exception:
                            conn.send((None, RuntimeError(str(exc))))

                if not conn_list:
                    break

        except Exception as exc:
            tb_str = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
            logger.exception(
                "BusServer on %s crashed: %s", self._port, exc
            )
            self._error_queue.put_nowait(("bus_server", tb_str))
        finally:
            self.ready_event.set()
            if bus is not None:
                try:
                    if bus.is_connected:
                        bus.disconnect()
                except Exception:
                    pass
            # logger.info(f"BusServer on {self._port} terminated")


# ======================================================================
# Client (drop-in replacement for a ComponentBus)
# ======================================================================

class BusClient(ComponentBus):
    """``ComponentBus`` proxy that forwards every operation to a
    ``BusServer`` over a ``multiprocessing.Pipe``.

    From the module's perspective this behaves identically to a direct
    ``FeetechBus`` (or any other ``ComponentBus``).
    """

    def __init__(
        self,
        client_conn: Connection,
        port: str,
        components: Dict[str, "Component"],
    ):
        super().__init__(port=port, components=components)
        self._conn = client_conn
        self._connected = False

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        self._conn.send((method, args, kwargs))
        result, error = self._conn.recv()
        if error is not None:
            raise error
        return result

    # -- ComponentBus interface ----------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def read(self, data_type: type[Data], component_names: list[str], normalize: bool = True) -> Dict[str, Any]:
        values = self._call("read", data_type, component_names, normalize=normalize)
        # The bus doing the capture is in another process, so the archive is
        # tapped here, on the frame as it arrives. For a simulated camera this
        # is also the only frame there is: MuJoCo renders straight to the
        # observation size, so there is no higher-fidelity source to keep.
        for name in self.archive_sinks:
            frame = values.get(name)
            if frame is not None:
                self.archive_capture(name, frame, self.components[name].cfg)
        return values

    def write(self, values: Dict[str, Data], normalize: bool = True, snap: bool = False):
        # snap teleports MuJoCo qpos; hardware buses have no such kwarg.
        if snap:
            return self._call("write", values, normalize=normalize, snap=True)
        return self._call("write", values, normalize=normalize)

    def configure(self, params: dict[str, int], component_names: list[str]):
        return self._call("configure", params, component_names)

    def enable_torque(self, component_names: list[str]):
        return self._call("enable_torque", component_names)

    def disable_torque(self, component_names: list[str]):
        return self._call("disable_torque", component_names)

    def is_calibrated(self) -> bool:
        return self._call("is_calibrated")

    def read_calibration(self) -> dict:
        return self._call("read_calibration")

    def write_calibration(self, calibration_dict: dict) -> dict:
        return self._call("write_calibration", calibration_dict)

    def start_calibration_mode(self):
        return self._call("start_calibration_mode")

    def end_calibration_mode(self):
        return self._call("end_calibration_mode")

    def setup_component(self, component_name: str, configs: dict):
        return self._call("setup_component", component_name, configs)

    def set_half_turn_homings(self, component_names: list[str]) -> dict:
        return self._call("set_half_turn_homings", component_names)

    def drive_to_encoder_center(self, component_names: list[str]) -> dict:
        return self._call("drive_to_encoder_center", component_names)

    def record_ranges_of_motion(self, component_names: list[str]):
        return self._call("record_ranges_of_motion", component_names)

    def reset_calibration(self, component_names: list[str]):
        return self._call("reset_calibration", component_names)

    def counts_per_revolution(self, name: str):
        return self._call("counts_per_revolution", name)

    def center_offset_in_hardware(self, name: str):
        return self._call("center_offset_in_hardware", name)

    def max_center_offset_abs(self, name: str):
        return self._call("max_center_offset_abs", name)

    def max_homing_offset_abs(self, name: str):
        return self._call("max_homing_offset_abs", name)

    def highlight_joint(self, name: str | None):
        return self._call("highlight_joint", name)
