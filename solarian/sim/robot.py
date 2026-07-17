"""Simulated robot base class.

A :class:`SimRobot` runs exactly like a real :class:`~solarian.core.robot.Robot`
-- same ZMQ transport, same observation/action schemas, same control loop -- but
instead of opening hardware buses it composes a MuJoCo scene from its modules'
MJCF models and launches a single :class:`~solarian.sim.mujoco_world.MujocoWorldServer`
that owns the physics. Each module talks to that world through a
:class:`~solarian.core.bus_server.BusClient` proxy, reusing the exact same
shared-port machinery hardware modules use when several of them share one
physical serial port.

Because the module subprocesses, observation sender and control loop are all
inherited unchanged from :class:`~solarian.core.robot.Robot`, driving a
``SimRobot`` with a real leader arm (via the unchanged controller) requires zero
schema or transport changes.
"""

from __future__ import annotations

import os
import threading
import time
from multiprocessing.connection import Pipe
from typing import Any, Dict

from solarian.core.robot import Robot
from solarian.core.bus_server import BusClient
from solarian.core.ik.actuator import CartesianIKActuator
from solarian.utils import setup_process_logging, StatusBoard

from solarian.sim.configs import SceneConfig
from solarian.sim.module import SimModule
from solarian.sim.servo import SimServo
from solarian.sim.camera import SimCamera
from solarian.sim.mujoco_world import MujocoWorldServer

import logging

logger = logging.getLogger("sim_robot")

_SIM_LAUNCH_TIMEOUT = 60.0


class SimRobot(Robot):
    """Robot whose modules are simulated in a shared MuJoCo world.

    Subclasses declare simulated modules (``SimModule`` instances) as class
    attributes -- exactly like a real robot -- and a :class:`SceneConfig`
    describing the environment and how each module's MJCF attaches::

        class So106SimRobot(SimRobot):
            arm = SimArmModule()
            gripper = SimGripperModule()
            wrist_camera = SimCameraModule()

            scene = SceneConfig(
                world_mjcf="assets/sim_scene.xml",
                attachments=[ModuleAttachment(module="arm", mjcf="so_arm101_mj_description")],
                cameras=[CameraMount(name="wrist_cam", parent_body="gripper", ...)],
            )
    """

    #: Declared by subclasses -- describes the environment + module attachments.
    scene: SceneConfig = None

    #: Show a live overview window of the whole scene while running. Overridable
    #: at runtime with the ``SOLARIAN_SIM_VIEW`` env var (``1``/``0``). The window
    #: is best-effort: with no display available the sim just runs headless.
    view: bool = False

    def _view_enabled(self) -> bool:
        env = os.environ.get("SOLARIAN_SIM_VIEW")
        if env is not None:
            return env.strip().lower() not in ("", "0", "false", "no")
        return bool(self.view)

    def __init__(self, cfg):
        super().__init__(cfg)

        for name, module in self.modules.items():
            if not isinstance(module, SimModule):
                raise TypeError(
                    f"SimRobot '{type(self).__name__}' may only declare SimModule "
                    f"instances, but module '{name}' is a {type(module).__name__}. "
                    f"Wrap it as a SimModule (e.g. subclass the real module and "
                    f"SimModule, re-declaring components as SimServo/SimCamera)."
                )

        if self.scene is None:
            raise ValueError(
                f"SimRobot '{type(self).__name__}' must declare a `scene` "
                f"(SceneConfig) describing its environment and module attachments."
            )

        self._world_server: MujocoWorldServer | None = None
        self._viewer_proc = None

    # ------------------------------------------------------------------
    def _collect_component_specs(self) -> Dict[str, Dict[str, Any]]:
        """Build the per-component MuJoCo mapping keyed by component name.

        All simulated components -- regardless of the (real) ``port`` declared
        in their yaml -- are served by the single MuJoCo world. Specs are keyed
        by ``"{module}/{component}"`` so identically named components on
        different modules (e.g. both arms' ``shoulder_pan``) stay distinct; the
        world server namespaces each request by its calling module.
        """
        specs: Dict[str, Dict[str, Any]] = {}

        for module_name, module in self.modules.items():
            for components in module.module_bus.components_by_port.values():
                for name, component in components.items():
                    key = f"{module_name}/{name}"
                    if isinstance(component, CartesianIKActuator):
                        # Cartesian IK actuators are not MuJoCo joints; they are
                        # driven by the module's IK wrapper over the joint bus.
                        continue
                    if isinstance(component, SimCamera):
                        specs[key] = self._camera_spec(component)
                    elif isinstance(component, SimServo):
                        specs[key] = self._joint_spec(component)
                    else:
                        raise TypeError(
                            f"Component '{name}' ({type(component).__name__}) is not a "
                            f"simulated component (SimServo/SimCamera)."
                        )

        return specs

    @staticmethod
    def _joint_spec(component: SimServo) -> Dict[str, Any]:
        cfg = component.cfg
        sim = cfg.sim
        if sim is None:
            raise ValueError(
                f"Simulated servo '{cfg.__class__.__name__}' is missing a `sim:` "
                f"block in its module yaml (joint/actuator + radian range)."
            )
        return {
            "kind": "joint",
            "joint": sim.joint,
            "actuator": sim.actuator,
            "mode": cfg.mode,
            "range_min": float(sim.range_min),
            "range_max": float(sim.range_max),
            "drive_mode": int(sim.drive_mode),
        }

    @staticmethod
    def _camera_spec(component: SimCamera) -> Dict[str, Any]:
        cfg = component.cfg
        sim = cfg.sim
        if sim is None:
            raise ValueError(
                "Simulated camera is missing a `sim:` block in its module yaml "
                "(the MuJoCo `camera` name to render)."
            )
        if cfg.resize_width is not None and cfg.resize_height is not None:
            h, w = cfg.resize_height, cfg.resize_width
        else:
            h, w = cfg.height, cfg.width
        return {
            "kind": "camera",
            "camera": sim.camera,
            "height": int(h),
            "width": int(w),
            # Optional calibrated fisheye distortion; applied by the world
            # server when rendering this camera. None for a plain pinhole.
            "distortion": sim.distortion,
        }

    def _collect_ee_frames(self) -> list[tuple[str, str]]:
        """Derive end-effector frames to visualize from each attached module's
        IK backend spec.

        For every scene attachment whose module declares an ``ik_backend`` with
        a ``frame`` (e.g. ``MinkIK(frame="gripper")``), the composed frame name
        is ``prefix + frame`` (the attach prefix namespaces the sub-model). The
        world server draws an axis triad at each. Empty if frames are disabled.
        """
        if not getattr(self.scene, "show_frames", True):
            return []
        ee_frames: list[tuple[str, str]] = []
        for att in self.scene.attachments:
            module = self.modules.get(att.module)
            if module is None:
                continue
            spec = getattr(type(module), "ik_backend", None)
            frame = getattr(spec, "frame", None)
            if not frame:
                continue
            frame_type = getattr(spec, "frame_type", "site")
            ee_frames.append((f"{att.prefix}{frame}", frame_type))
        return ee_frames

    # ------------------------------------------------------------------
    def connect(self):
        """Compose the scene, launch the world server, and proxy module I/O.

        One ``BusClient`` proxy is injected per (module, port): every module's
        components -- whatever their declared port -- are routed to the single
        MuJoCo world, which owns the physics and resolves requests by component
        name.
        """
        specs = self._collect_component_specs()

        # One pipe per (module, port) so each module process talks to the world
        # over its own connection. ``conn_modules`` tells the world which module
        # each connection belongs to so it can namespace component names.
        server_conns: Dict[str, Any] = {}
        client_conns: Dict[str, Any] = {}
        conn_modules: Dict[str, str] = {}
        for name, module in self.modules.items():
            for port in module.module_bus.components_by_port:
                conn_key = self._conn_key(name, port)
                server_conn, client_conn = Pipe()
                server_conns[conn_key] = server_conn
                client_conns[(name, port)] = client_conn
                conn_modules[conn_key] = name

        # An extra connection for the live viewer process, if enabled. It talks
        # to the world server over the same RPC protocol (render_overview).
        viewer_client_conn = None
        if self._view_enabled():
            viewer_server_conn, viewer_client_conn = Pipe()
            server_conns["__viewer__"] = viewer_server_conn
            conn_modules["__viewer__"] = ""

        self._world_server = MujocoWorldServer(
            scene=self.scene,
            component_specs=specs,
            server_conns=server_conns,
            conn_modules=conn_modules,
            ee_frames=self._collect_ee_frames(),
        )
        self._world_server.start()
        # The child now holds its own duplicated ends of every pipe. Drop the
        # parent's copies of the *server* ends so that if the world server dies,
        # the modules' BusClient reads raise EOFError instead of blocking forever
        # (the parent must not keep the write end of a client's pipe open).
        for server_conn in server_conns.values():
            server_conn.close()

        self._world_server.ready_event.wait(timeout=_SIM_LAUNCH_TIMEOUT)

        # ``ready_event`` is set from the world server's ``finally`` even on a
        # build failure, and the crash traceback is queued *before* that, so an
        # error here is authoritative. Give the queue's feeder thread a brief
        # moment to flush the traceback if the process has already exited.
        err = self._world_server.get_error()
        if err is None and not self._world_server.is_alive():
            deadline = time.monotonic() + 2.0
            while err is None and time.monotonic() < deadline:
                time.sleep(0.05)
                err = self._world_server.get_error()
        if err:
            _name, tb_str = err
            raise RuntimeError(f"MujocoWorldServer failed to start:\n{tb_str}")
        if not self._world_server.is_alive():
            raise RuntimeError("MujocoWorldServer died during startup (check stderr)")

        # Launch the live viewer in its own process (its main thread hosts the
        # OpenCV window). It renders via the world server, so it stays in sync
        # with the physics; closing it does not stop the sim.
        if viewer_client_conn is not None:
            from solarian.sim.viewer import SimViewer
            from solarian.utils import get_log_queue

            self._viewer_proc = SimViewer(conn=viewer_client_conn, log_queue=get_log_queue())
            self._viewer_proc.start()
            viewer_client_conn.close()  # child owns it now

        for name, module in self.modules.items():
            for port, components in module.module_bus.components_by_port.items():
                client = BusClient(
                    client_conn=client_conns[(name, port)],
                    port=port,
                    components=components,
                )
                module.module_bus.override_port_bus(port, client)
            # No hardware calibration in sim; establishes module.calibration = {}.
            module.load_calibration()
            module.connect()

        self.connected = True
        logger.info("SimRobot connected to MuJoCo world (%d components)", len(specs))

    @staticmethod
    def _conn_key(module_name: str, port: str) -> str:
        return f"{module_name}::{port}"

    def initialize(self):
        """Run each module's initialize against the live sim world."""
        logger.info("Initializing simulated modules")
        for module in self.modules.values():
            module.initialize()
        logger.info("SimRobot initialized")

    def start(self):
        """Fork module subprocesses and run the control loop.

        Mirrors :meth:`Robot.start` but keeps the already-running world server
        as the sole "bus server" and does not open any hardware.
        """
        self.running = True
        self._bus_servers = [self._world_server] if self._world_server is not None else []
        setup_process_logging(logger_name=None)

        StatusBoard()
        status_mp_q = StatusBoard.get_mp_queue()

        try:
            for module in self.modules.values():
                module._status_mp_q = status_mp_q
                # Clears parent-side component buses (keeps the BusClient
                # override) so forked children re-attach to the world server.
                module.disconnect()

            for module in self.modules.values():
                module.start()

            self.on_start()

            threading.Thread(target=self._observation_sender, daemon=True).start()
            self.control_loop()
        finally:
            self.running = False
            self.disconnect()

    def disconnect(self):
        """Tear down the viewer process, then the base robot / world server."""
        proc = self._viewer_proc
        if proc is not None:
            self._viewer_proc = None
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        super().disconnect()
