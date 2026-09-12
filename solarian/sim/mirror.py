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

import threading
import time
from multiprocessing.connection import Pipe
from typing import Callable, Dict, Iterable, Optional

from solarian.core.bus_server import BusClient
from solarian.sim.runtime import (
    SIM_LAUNCH_TIMEOUT,
    _conn_key,
    collect_component_specs,
    collect_tcp_frames,
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
        self._prompt_parent = None
        self._prompt_lock = threading.Lock()
        self._client_lock = threading.Lock()
        self._active_joint: str | None = None
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
            tcp_frames=collect_tcp_frames(self.robot),
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

            prompt_parent, prompt_child = Pipe()
            self._prompt_parent = prompt_parent
            self._viewer = SimViewer(
                conn=viewer_client_conn,
                log_queue=get_log_queue(),
                prompt_conn=prompt_child,
            )
            self._viewer.start()
            viewer_client_conn.close()
            prompt_child.close()

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
        if self._prompt_parent is not None:
            try:
                self._prompt_parent.close()
            except Exception:  # noqa: BLE001
                pass
            self._prompt_parent = None
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
    def show(
        self,
        module_name: str,
        values_by_component: Dict[str, float],
        *,
        normalize: bool = True,
    ) -> None:
        """Push joint targets into the sim for ``module_name``.

        No-op when the mirror is unavailable. ``normalize=True`` (default) uses
        the same ``[-100, 100]`` units as the real bus. ``normalize=False``
        writes radians (model ``q``).
        """
        if not self.available:
            return
        from solarian.core.data import PosData

        module = self.robot.modules.get(module_name)
        if module is None:
            return
        with self._client_lock:
            for port, components in module.module_bus.components_by_port.items():
                client = self._clients.get((module_name, port))
                if client is None:
                    continue
                values = {
                    name: PosData(value=float(val))
                    for name, val in values_by_component.items()
                    if name in components
                }
                if values:
                    try:
                        # Snap qpos so the overview shows the pose this frame (PD
                        # tracking can leave the mesh at the previous q on first look).
                        client.write(values, normalize=normalize, snap=True)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("SimMirror write failed for %s/%s: %s", module_name, port, exc)

    def show_model_zero(self, module_name: str, joints: Iterable[str]) -> None:
        """Park every listed joint at model ``q=0`` (radians, not normalized mid)."""
        self.show(module_name, {name: 0.0 for name in joints}, normalize=False)

    def highlight(self, module_name: str, joint: str | None) -> None:
        """Mark ``joint`` with a red hinge arrow, or clear when ``None``."""
        self._active_joint = joint
        if not self.available:
            return
        with self._client_lock:
            for (name, _port), client in self._clients.items():
                if name != module_name:
                    continue
                try:
                    client.highlight_joint(joint)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SimMirror highlight failed for %s/%s: %s", module_name, joint, exc)
                return

    def update_hud(
        self,
        module_name: str,
        joints: Iterable[str],
        real: Dict[str, float],
    ) -> None:
        """Push live real-normalized and sim-radian values onto the viewer."""
        if not self.available or self._prompt_parent is None:
            return
        names = list(joints)
        payload = {
            "joints": names,
            "real": {n: real[n] for n in names if n in real},
            "sim": self.sim_joint_radians(module_name, names),
            "active": self._active_joint,
        }
        with self._prompt_lock:
            try:
                self._prompt_parent.send(("hud", payload))
            except (EOFError, OSError):
                return

    def sim_joint_radians(self, module_name: str, joints: Iterable[str]) -> Dict[str, float]:
        """Read the sim world's joint angles in radians."""
        from solarian.core.data import PosData

        out: Dict[str, float] = {}
        module = self.robot.modules.get(module_name) if self.robot is not None else None
        if module is None:
            return out
        with self._client_lock:
            for port, components in module.module_bus.components_by_port.items():
                client = self._clients.get((module_name, port))
                names = [n for n in joints if n in components]
                if client is None or not names:
                    continue
                try:
                    raw = client.read(PosData, names, normalize=False)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SimMirror sim-angle read failed for %s/%s: %s", module_name, port, exc)
                    continue
                for name in names:
                    val = raw.get(name)
                    if val is None:
                        continue
                    out[name] = float(val["value"] if isinstance(val, dict) else val)
        return out

    def ask(self, text: str, *, yes_no: bool = False) -> str:
        """Show ``text`` in the sim window and wait for Enter or Y/N.

        Falls back to the console when the viewer is unavailable. Always
        prints the same prompt on stdout so the session log has a copy.
        """
        print(f"\n{text}")
        if yes_no:
            print("  Answer in the sim window: Y / N  (console also works).")
        else:
            print("  Press ENTER in the sim window to continue (console also works).")
        if not self.available or self._prompt_parent is None:
            return self._ask_console(yes_no)
        kind = "yesno" if yes_no else "enter"
        with self._prompt_lock:
            try:
                self._prompt_parent.send(("ask", {"text": text, "kind": kind}))
            except (EOFError, OSError):
                return self._ask_console(yes_no)
        while True:
            if self._viewer is not None and not self._viewer.is_alive():
                print("  (sim window closed; answer here)")
                return self._ask_console(yes_no)
            typed = self._poll_console(yes_no)
            if typed is not None:
                return typed
            try:
                with self._prompt_lock:
                    ready = self._prompt_parent.poll(0)
                    reply = self._prompt_parent.recv() if ready else None
                if reply is not None:
                    return str(reply).strip().lower()
            except (EOFError, OSError):
                return self._ask_console(yes_no)
            time.sleep(0.15)

    @staticmethod
    def _ask_console(yes_no: bool) -> str:
        if yes_no:
            return input("[y/n]: ").strip().lower()
        input("Press ENTER...")
        return ""

    @staticmethod
    def _poll_console(yes_no: bool) -> str | None:
        import select
        import sys

        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        line = sys.stdin.readline()
        if yes_no:
            return line.strip().lower()
        return ""

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
    SO-106 ``wrist_roll``) are omitted by the hard-stop sweep, so including
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
