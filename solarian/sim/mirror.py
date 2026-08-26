"""Live sim view *alongside* real hardware, for IK sim<->real alignment.

Unlike :mod:`solarian.sim.runtime` (which replaces a module's hardware bus with a
MuJoCo proxy when ``is_sim`` is set), :class:`SimMirror` runs a MuJoCo world +
viewer next to a robot whose modules stay connected to real servos. It never
touches the module buses; instead it exposes :meth:`show` to push normalized
joint values into the sim so the on-screen arm mirrors a real pose, and
:meth:`mirror` to stream the real arm's live pose into the sim.

Because both real and sim exchange joints in the normalized ``[-100, 100]``
space, a correctly-calibrated real arm makes the sim track it 1:1 -- which is the
verification surface for the alignment wizard. It reuses the exact scene / world
/ viewer / proxy machinery from :mod:`solarian.sim.runtime`.
"""

from __future__ import annotations

import time
from multiprocessing.connection import Pipe
from typing import Callable, Dict, Iterable, Optional

from solarian.core.bus_server import BusClient
from solarian.sim.runtime import (
    SIM_LAUNCH_TIMEOUT,
    _conn_key,
    collect_component_specs,
    collect_ee_frames,
    view_enabled,
)

import logging

logger = logging.getLogger("sim_mirror")


class SimMirror:
    """A MuJoCo world + viewer that mirrors a real robot's joints.

    Best-effort: if MuJoCo/the display is unavailable, :meth:`start` logs a
    warning and leaves :attr:`available` ``False`` so the caller's guided flow
    keeps working with prompts only.
    """

    def __init__(self, robot):
        self.robot = robot
        self.available = False
        self._world = None
        self._viewer = None
        self._clients: Dict[tuple[str, str], BusClient] = {}

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> bool:
        try:
            self._start()
            self.available = True
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning("SimMirror unavailable (%s); continuing without live view.", exc)
            self.available = False
            self._teardown()
        return self.available

    def _start(self) -> None:
        from solarian.sim.mujoco_world import MujocoWorldServer

        specs = collect_component_specs(self.robot)

        server_conns: Dict[str, object] = {}
        client_conns: Dict[tuple[str, str], object] = {}
        conn_modules: Dict[str, str] = {}
        for name, module in self.robot.modules.items():
            for port in module.module_bus.components_by_port:
                key = _conn_key(name, port)
                server_conn, client_conn = Pipe()
                server_conns[key] = server_conn
                client_conns[(name, port)] = client_conn
                conn_modules[key] = name

        viewer_client_conn = None
        if view_enabled(self.robot):
            viewer_server_conn, viewer_client_conn = Pipe()
            server_conns["__viewer__"] = viewer_server_conn
            conn_modules["__viewer__"] = ""

        self._world = MujocoWorldServer(
            scene=self.robot.scene,
            component_specs=specs,
            server_conns=server_conns,
            conn_modules=conn_modules,
            ee_frames=collect_ee_frames(self.robot),
        )
        self._world.start()
        # Drop the parent's copies of the server ends (child owns them now).
        for server_conn in server_conns.values():
            server_conn.close()

        self._world.ready_event.wait(timeout=SIM_LAUNCH_TIMEOUT)
        err = self._world.get_error()
        if err:
            raise RuntimeError(f"MuJoCo world failed to start:\n{err[1]}")
        if not self._world.is_alive():
            raise RuntimeError("MuJoCo world died during startup")

        if viewer_client_conn is not None:
            from solarian.sim.viewer import SimViewer
            from solarian.utils import get_log_queue

            self._viewer = SimViewer(conn=viewer_client_conn, log_queue=get_log_queue())
            self._viewer.start()
            viewer_client_conn.close()

        for (name, port), client_conn in client_conns.items():
            components = self.robot.modules[name].module_bus.components_by_port[port]
            client = BusClient(client_conn=client_conn, port=port, components=components)
            client.connect()
            self._clients[(name, port)] = client

    def close(self) -> None:
        self._teardown()

    def _teardown(self) -> None:
        if self._viewer is not None:
            try:
                if self._viewer.is_alive():
                    self._viewer.terminate()
                    self._viewer.join(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            self._viewer = None
        if self._world is not None:
            try:
                if self._world.is_alive():
                    self._world.terminate()
                    self._world.join(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            self._world = None
        self._clients = {}

    # -- driving the sim ---------------------------------------------------
    def show(self, module_name: str, normalized_by_component: Dict[str, float]) -> None:
        """Push normalized joint targets into the sim for ``module_name``.

        No-op when the mirror is unavailable. Values are the same normalized
        ``[-100, 100]`` units read from / written to the real bus.
        """
        if not self.available:
            return
        from solarian.core.data import PosData

        module = self.robot.modules.get(module_name)
        if module is None:
            return
        for port, components in module.module_bus.components_by_port.items():
            client = self._clients.get((module_name, port))
            if client is None:
                continue
            values = {
                name: PosData(value=float(val))
                for name, val in normalized_by_component.items()
                if name in components
            }
            if values:
                try:
                    client.write(values, normalize=True)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SimMirror write failed for %s/%s: %s", module_name, port, exc)

    def mirror(
        self,
        module_name: str,
        read_normalized: Callable[[], Dict[str, float]],
        *,
        duration: Optional[float] = None,
        stop: Optional[Callable[[], bool]] = None,
        period: float = 1.0 / 30.0,
    ) -> None:
        """Continuously read the real arm and drive the sim to match.

        ``read_normalized`` returns ``{joint: normalized}`` for the real arm.
        Loops until ``duration`` elapses or ``stop()`` returns ``True``.
        """
        if not self.available:
            return
        deadline = None if duration is None else time.monotonic() + duration
        while True:
            if stop is not None and stop():
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            try:
                self.show(module_name, read_normalized())
            except Exception:  # noqa: BLE001
                pass
            time.sleep(period)


def joints_present(module, joints: Iterable[str]) -> Dict[str, float]:
    """Read the real module's present normalized joint positions.

    Joints without a calibration entry are skipped. Continuous joints (e.g.
    SO-106 ``wrist_roll``) are omitted by ``calibrate-module``, so including
    them here would KeyError inside the servo ``_normalize`` and freeze the
    whole live mirror.
    """
    from solarian.core.data import PosData

    names = list(joints)
    calib = getattr(module, "calibration", None) or {}
    readable = [n for n in names if n in calib]
    if not readable:
        return {}
    present = module.read_sensors(PosData, readable)
    return {
        name: float(present[name]["value"])
        for name in readable
        if name in present and present[name] is not None
    }
