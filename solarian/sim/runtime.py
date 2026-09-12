"""Simulation runtime for the ``is_sim`` switch on :class:`~solarian.core.robot.Robot`.

A robot runs in MuJoCo by declaring ``is_sim = True`` and a ``scene``
(:class:`~solarian.sim.configs.SceneConfig`) -- no separate ``SimRobot``,
``SimModule`` or ``SimServo`` classes are needed. The ordinary robot keeps its
real modules and servos; when ``is_sim`` is set, :class:`Robot` delegates its
``connect`` / ``start`` / ``disconnect`` hardware paths to the helpers here.

Instead of opening hardware buses, the robot composes a MuJoCo scene from its
modules' MJCF models and launches a single
:class:`~solarian.sim.mujoco_world.MujocoWorldServer` that owns the physics.
Each module talks to that world through a
:class:`~solarian.core.bus_server.BusClient` proxy -- reusing the exact same
shared-port machinery hardware modules use when several of them share one
physical serial port. Because the module subprocesses, observation sender and
control loop are all inherited unchanged, driving a simulated robot with a real
leader arm requires zero schema or transport changes.

MuJoCo and the other ``[sim]`` extras are imported lazily inside these helpers
so that ``import solarian`` never requires them; only running a robot with
``is_sim = True`` does.
"""

from __future__ import annotations

import os
import time
from multiprocessing.connection import Pipe
from typing import Any, Dict

from solarian.core.actuator import Actuator
from solarian.core.bus_server import BusClient
from solarian.core.camera import Camera
from solarian.core.ik.actuator import CartesianIKActuator

import logging

logger = logging.getLogger("sim_runtime")

#: How long to wait for the MuJoCo world server to finish building the scene.
SIM_LAUNCH_TIMEOUT = 60.0


def view_enabled(robot) -> bool:
    """Whether to show the live overview window.

    Defaults to the robot's ``view`` class attribute, overridable at runtime
    with the ``SOLARIAN_SIM_VIEW`` env var (``1``/``0``). The window is
    best-effort: with no display available the sim just runs headless.
    """
    env = os.environ.get("SOLARIAN_SIM_VIEW")
    if env is not None:
        return env.strip().lower() not in ("", "0", "false", "no")
    return bool(getattr(robot, "view", False))


def _conn_key(module_name: str, port: str) -> str:
    return f"{module_name}::{port}"


def _joint_spec(component) -> Dict[str, Any]:
    cfg = component.cfg
    joint = getattr(cfg, "sim_joint", None)
    return {
        "kind": "joint",
        "joint": joint,
        "actuator": joint,
        "mode": cfg.mode,
        "range_min": float(cfg.sim_range_min),
        "range_max": float(cfg.sim_range_max),
        "drive_mode": int(getattr(cfg, "sim_drive_mode", 0) or 0),
    }


def _camera_spec(component) -> Dict[str, Any]:
    cfg = component.cfg
    sim = getattr(cfg, "sim", None)
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
        # Optional calibrated fisheye distortion; applied by the world server
        # when rendering this camera. None for a plain pinhole.
        "distortion": sim.distortion,
    }


def collect_component_specs(robot) -> Dict[str, Dict[str, Any]]:
    """Build the per-component MuJoCo mapping keyed by ``"{module}/{component}"``.

    All components -- regardless of the (real) ``port`` declared in their yaml
    -- are served by the single MuJoCo world. Keys are namespaced by module so
    identically named components on different modules (e.g. both arms'
    ``shoulder_pan``) stay distinct; the world server namespaces each request by
    its calling module.

    Component kind is inferred from its (real) type: a :class:`Camera` renders a
    MuJoCo camera; any other :class:`Actuator` (servo/motor) drives a MuJoCo
    joint. :class:`CartesianIKActuator` are skipped -- they are not MuJoCo
    joints but are driven by the module's IK wrapper over the joint bus.
    """
    specs: Dict[str, Dict[str, Any]] = {}

    for module_name, module in robot.modules.items():
        for components in module.module_bus.components_by_port.values():
            for name, component in components.items():
                key = f"{module_name}/{name}"
                if isinstance(component, CartesianIKActuator):
                    continue
                if isinstance(component, Camera):
                    specs[key] = _camera_spec(component)
                elif isinstance(component, Actuator):
                    specs[key] = _joint_spec(component)
                else:
                    raise TypeError(
                        f"Component '{name}' ({type(component).__name__}) cannot be "
                        f"simulated: only actuators (MuJoCo joints) and cameras "
                        f"(MuJoCo cameras) are supported under `is_sim`."
                    )

    return specs


def collect_tcp_frames(robot) -> list[tuple[str, str]]:
    """Derive TCP frames to visualize from each attached module's IK spec.

    For every scene attachment whose module declares an ``ik_backend`` with a
    ``frame`` (e.g. ``MinkIK(frame="tcp")``), the composed site name is
    ``prefix + frame``. If ``frame`` already contains ``/`` it is treated as
    a fully composed name (a tool TCP living on another attachment).
    Empty if frames are disabled.
    """
    if not getattr(robot.scene, "show_frames", True):
        return []
    tcp_frames: list[tuple[str, str]] = []
    for att in robot.scene.attachments:
        module = robot.modules.get(att.module)
        if module is None:
            continue
        spec = getattr(type(module), "ik_backend", None)
        frame = getattr(spec, "frame", None)
        if not frame:
            continue
        frame_type = getattr(spec, "frame_type", "site")
        composed = frame if "/" in frame else f"{att.prefix}{frame}"
        tcp_frames.append((composed, frame_type))
    return tcp_frames


def sim_connect(robot) -> None:
    """Compose the scene, launch the world server, and proxy module I/O.

    One ``BusClient`` proxy is injected per (module, port): every module's
    components -- whatever their declared port -- are routed to the single
    MuJoCo world, which owns the physics and resolves requests by component
    name. This replaces :meth:`Robot.connect`'s hardware path when ``is_sim``.
    """
    from solarian.sim.mujoco_world import MujocoWorldServer

    specs = collect_component_specs(robot)

    # One pipe per (module, port) so each module process talks to the world over
    # its own connection. ``conn_modules`` tells the world which module each
    # connection belongs to so it can namespace component names.
    server_conns: Dict[str, Any] = {}
    client_conns: Dict[Any, Any] = {}
    conn_modules: Dict[str, str] = {}
    for name, module in robot.modules.items():
        for port in module.module_bus.components_by_port:
            conn_key = _conn_key(name, port)
            server_conn, client_conn = Pipe()
            server_conns[conn_key] = server_conn
            client_conns[(name, port)] = client_conn
            conn_modules[conn_key] = name

    # An extra connection for the live viewer process, if enabled. It talks to
    # the world server over the same RPC protocol (render_overview).
    viewer_client_conn = None
    if view_enabled(robot):
        viewer_server_conn, viewer_client_conn = Pipe()
        server_conns["__viewer__"] = viewer_server_conn
        conn_modules["__viewer__"] = ""

    robot._world_server = MujocoWorldServer(
        scene=robot.scene,
        component_specs=specs,
        server_conns=server_conns,
        conn_modules=conn_modules,
        tcp_frames=collect_tcp_frames(robot),
    )
    robot._world_server.start()
    # The child now holds its own duplicated ends of every pipe. Drop the
    # parent's copies of the *server* ends so that if the world server dies, the
    # modules' BusClient reads raise EOFError instead of blocking forever (the
    # parent must not keep the write end of a client's pipe open).
    for server_conn in server_conns.values():
        server_conn.close()

    robot._world_server.ready_event.wait(timeout=SIM_LAUNCH_TIMEOUT)

    # ``ready_event`` is set from the world server's ``finally`` even on a build
    # failure, and the crash traceback is queued *before* that, so an error here
    # is authoritative. Give the queue's feeder thread a brief moment to flush
    # the traceback if the process has already exited.
    err = robot._world_server.get_error()
    if err is None and not robot._world_server.is_alive():
        deadline = time.monotonic() + 2.0
        while err is None and time.monotonic() < deadline:
            time.sleep(0.05)
            err = robot._world_server.get_error()
    if err:
        _name, tb_str = err
        raise RuntimeError(f"MujocoWorldServer failed to start:\n{tb_str}")
    if not robot._world_server.is_alive():
        raise RuntimeError("MujocoWorldServer died during startup (check stderr)")

    # Launch the live viewer in its own process (its main thread hosts the
    # OpenCV window). It renders via the world server, so it stays in sync with
    # the physics; closing it does not stop the sim.
    if viewer_client_conn is not None:
        from solarian.sim.viewer import SimViewer
        from solarian.utils import get_log_queue

        robot._viewer_proc = SimViewer(conn=viewer_client_conn, log_queue=get_log_queue())
        robot._viewer_proc.start()
        viewer_client_conn.close()  # child owns it now

    for name, module in robot.modules.items():
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

    robot.connected = True
    logger.info("Robot connected to MuJoCo world (%d components)", len(specs))


def sim_teardown(robot) -> None:
    """Terminate the live viewer process, if any.

    Called at the top of :meth:`Robot.disconnect` when ``is_sim``; the world
    server itself is torn down by the base ``disconnect`` because it is listed
    in ``robot._bus_servers``.
    """
    proc = getattr(robot, "_viewer_proc", None)
    if proc is not None:
        robot._viewer_proc = None
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
