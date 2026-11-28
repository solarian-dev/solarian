"""Shared MuJoCo world server.

Owns a single composed ``MjModel``/``MjData`` for a simulated robot, steps the
physics in real time in a background thread, and answers read / write / render
requests from every simulated module. It mirrors the Pipe-RPC protocol of
:class:`~solarian.core.bus_server.BusServer` so each module talks to it through
an ordinary :class:`~solarian.core.bus_server.BusClient` proxy -- exactly like
several modules sharing one physical serial port.

Simulated servos are position-controlled: a module ``write`` sets the MuJoCo
actuator ``ctrl`` target and a ``read`` returns the present joint position. The
physics loop advances autonomously, just like real servos running their own
control loop, which is what keeps sim and real transparent.
"""

from __future__ import annotations

import math
import threading
import time
import traceback as _tb
from multiprocessing import Process, Event, Queue as MpQueue
from multiprocessing.connection import Connection, wait as conn_wait
from typing import Any, Dict

import numpy as np

from solarian.sim.configs import SceneConfig

import logging

logger = logging.getLogger("mujoco_world")

_SHUTDOWN_SENTINEL = "__shutdown__"


def _unnormalize(value: float, mode: str, range_min: float, range_max: float, drive_mode: int) -> float:
    """Map a normalized command to a MuJoCo control target (radians)."""
    if mode == "range_m100_100":
        v = -value if drive_mode else value
        v = min(100.0, max(-100.0, v))
        return ((v + 100.0) / 200.0) * (range_max - range_min) + range_min
    if mode == "range_0_100":
        v = 100.0 - value if drive_mode else value
        v = min(100.0, max(0.0, v))
        return (v / 100.0) * (range_max - range_min) + range_min
    # raw / unknown: pass through
    return value


def _normalize(value: float, mode: str, range_min: float, range_max: float, drive_mode: int) -> float:
    """Map a MuJoCo joint position (radians) to normalized units."""
    if range_max == range_min:
        return 0.0
    if mode == "range_m100_100":
        bounded = min(range_max, max(range_min, value))
        norm = (((bounded - range_min) / (range_max - range_min)) * 200.0) - 100.0
        return -norm if drive_mode else norm
    if mode == "range_0_100":
        bounded = min(range_max, max(range_min, value))
        norm = ((bounded - range_min) / (range_max - range_min)) * 100.0
        return 100.0 - norm if drive_mode else norm
    return value


def hinge_world_frame(xanchor, xaxis, joint_id: int, dofadr: int, njnt: int | None = None):
    """World hinge origin/axis for one joint.

    MuJoCo 3 stores ``xanchor`` / ``xaxis`` per joint (``njnt``). Older builds
    used one row per dof (``nv``). ``dofadr`` is only valid for the latter —
    using it on an ``njnt`` array points at the wrong mate and goes out of
    range once the scene has extra dofs (free cube, gripper slides).
    """
    n = int(np.asarray(xanchor).shape[0])
    jid, dof = int(joint_id), int(dofadr)
    if njnt is not None and n == int(njnt) and 0 <= jid < n:
        idx = jid
    elif 0 <= dof < n:
        idx = dof
    elif 0 <= jid < n:
        idx = jid
    else:
        return None
    return np.asarray(xanchor[idx], dtype=float), np.asarray(xaxis[idx], dtype=float)


def hinge_arrow_ends(origin, axis, length: float = 0.10):
    """World-frame segment along the joint axis, centered on the hinge."""
    origin = np.asarray(origin, dtype=float)
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return origin.copy(), origin.copy()
    half = axis / norm * (length * 0.5)
    return origin - half, origin + half


class MujocoWorldServer(Process):
    """Process that owns the composed MuJoCo scene and serves module I/O."""

    def __init__(
        self,
        scene: SceneConfig,
        component_specs: Dict[str, Dict[str, Any]],
        server_conns: Dict[str, Connection],
        conn_modules: Dict[str, str],
        realtime: bool = True,
        tcp_frames: list[tuple[str, str]] | None = None,
    ):
        super().__init__(daemon=True)
        self._scene = scene
        self._component_specs = component_specs
        self._server_conns = server_conns
        # TCP frames to visualize in the overview viewer, as
        # ``(composed_frame_name, frame_type)`` where frame_type is
        # "site"/"body"/"geom". Resolved to ids in ``_build``.
        self._tcp_frames = list(tcp_frames or [])
        self._tcp_frame_ids: list[tuple[str, int]] = []
        # Every other site (CAD frames such as ``flange``), drawn smaller.
        self._site_frame_ids: list[int] = []
        # {conn_key: module_name} -- lets the server namespace bare component
        # names per connection so identically named components on different
        # modules (e.g. both arms' ``shoulder_pan``) don't collide.
        self._conn_modules = conn_modules
        self._realtime = realtime

        self.ready_event = Event()
        self._error_queue: MpQueue = MpQueue()

        from solarian.utils import get_log_queue

        self._log_queue = get_log_queue()

        # Everything below is created inside the child process (``run``): the
        # server may be launched with the 'spawn' start method, which pickles
        # this object, and threading primitives / MjModel are not picklable.
        self.model = None
        self.data = None
        self._lock = None
        self._stop = None
        self._renderers: dict = {}
        # Cached fisheye remaps keyed like the renderers: {(camera, h, w): (map1, map2)}.
        self._distort_maps: dict = {}
        self._distort_warned: set = set()
        self._virtual: dict[str, float] = {}
        self._render_failed = False
        self._overview_failed = False
        self._highlight_key: str | None = None

    # ------------------------------------------------------------------
    def get_error(self) -> tuple[str, str] | None:
        try:
            return self._error_queue.get_nowait()
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _build(self):
        import mujoco

        from solarian.sim.scene import compose_scene

        self.model = compose_scene(self._scene)
        self.data = mujoco.MjData(self.model)

        # Resolve MuJoCo indices for every declared component.
        for name, spec in self._component_specs.items():
            if spec["kind"] == "joint":
                joint = spec.get("joint")
                actuator = spec.get("actuator")
                spec["joint_id"] = self._name2id(mujoco.mjtObj.mjOBJ_JOINT, joint)
                spec["actuator_id"] = self._name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
                if spec["joint_id"] is not None:
                    spec["qposadr"] = int(self.model.jnt_qposadr[spec["joint_id"]])
                    spec["dofadr"] = int(self.model.jnt_dofadr[spec["joint_id"]])
                    spec["range_min"], spec["range_max"] = self._servo_range(name, spec)
                else:
                    self._virtual.setdefault(name, 0.0)
                bound = []
                for follower in spec.get("followers") or []:
                    follower["joint_id"] = self._name2id(mujoco.mjtObj.mjOBJ_JOINT, follower["joint"])
                    follower["actuator_id"] = self._name2id(
                        mujoco.mjtObj.mjOBJ_ACTUATOR, follower["actuator"]
                    )
                    if follower["joint_id"] is None or follower["actuator_id"] is None:
                        logger.warning(
                            "Sim follower '%s' for %s was not found in the scene.",
                            follower["joint"],
                            name,
                        )
                        continue
                    follower["qposadr"] = int(self.model.jnt_qposadr[follower["joint_id"]])
                    follower["range_min"], follower["range_max"] = self._joint_range(follower["joint_id"])
                    bound.append(follower)
                spec["followers"] = bound
            elif spec["kind"] == "camera":
                spec["camera_id"] = self._name2id(mujoco.mjtObj.mjOBJ_CAMERA, spec.get("camera"))

        # Resolve TCP frames used for the overview axis triads.
        objtype_by_frame = {
            "site": mujoco.mjtObj.mjOBJ_SITE,
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
        }
        for frame_name, frame_type in self._tcp_frames:
            objtype = objtype_by_frame.get(frame_type, mujoco.mjtObj.mjOBJ_SITE)
            fid = self._name2id(objtype, frame_name)
            if fid is None:
                logger.warning(
                    "TCP frame '%s' (%s) not found in scene; not drawing its axes.",
                    frame_name, frame_type,
                )
                continue
            self._tcp_frame_ids.append((frame_type, fid))
        tcp_sites = {fid for frame_type, fid in self._tcp_frame_ids if frame_type == "site"}
        self._site_frame_ids = [i for i in range(self.model.nsite) if i not in tcp_sites]

        mujoco.mj_forward(self.model, self.data)

    def _joint_range(self, joint_id: int) -> tuple[float, float]:
        """The joint's limits in the model, which the normalized range spans.

        A joint without limits (a wheel, a free roll) gets one turn, -pi..pi.
        """
        if not self.model.jnt_limited[joint_id]:
            return -math.pi, math.pi
        lo, hi = self.model.jnt_range[joint_id]
        return float(lo), float(hi)

    def _servo_range(self, name: str, spec: Dict[str, Any]) -> tuple[float, float]:
        """The model's limits, with either end replaced by the yaml's override."""
        lo, hi = self._joint_range(spec["joint_id"])
        override_lo, override_hi = spec["range_override"]
        if override_lo is None and override_hi is None:
            return lo, hi
        range_lo = lo if override_lo is None else float(override_lo)
        range_hi = hi if override_hi is None else float(override_hi)
        # Logged because an override outlives a re-import of the model it was
        # fitted to; the model's limits beside it show when it has gone stale.
        logger.info(
            "%s: sim_range %.5g..%.5g overrides joint '%s' limits %.5g..%.5g in the model.",
            name, range_lo, range_hi, spec["joint"], lo, hi,
        )
        return range_lo, range_hi

    def _name2id(self, objtype, name):
        import mujoco

        if name is None:
            return None
        idx = mujoco.mj_name2id(self.model, objtype, name)
        return None if idx < 0 else int(idx)

    # ------------------------------------------------------------------
    def _stepping_loop(self):
        import mujoco

        dt = float(self.model.opt.timestep)
        next_t = time.perf_counter()
        while not self._stop.is_set():
            with self._lock:
                mujoco.mj_step(self.model, self.data)
            if self._realtime:
                next_t += dt
                sleep = next_t - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    # Fell behind; resync to avoid spiraling.
                    next_t = time.perf_counter()

    # ------------------------------------------------------------------
    # RPC handlers (names match BusClient method calls)
    # ------------------------------------------------------------------
    def read(self, module: str, data_type, component_names: list[str], normalize: bool = True) -> Dict[str, Any]:
        from solarian.core.data import ImageData, SpeedData, TemperatureData

        out: Dict[str, Any] = {}
        if issubclass(data_type, ImageData):
            for name in component_names:
                spec = self._component_specs.get(f"{module}/{name}")
                if spec is None or spec["kind"] != "camera":
                    continue
                out[name] = self._render(spec)
            return out

        if issubclass(data_type, TemperatureData):
            for name in component_names:
                out[name] = 25.0
            return out

        want_speed = issubclass(data_type, SpeedData)
        with self._lock:
            for name in component_names:
                full = f"{module}/{name}"
                spec = self._component_specs.get(full)
                if spec is None or spec["kind"] != "joint":
                    out[name] = 0.0
                    continue
                if spec.get("joint_id") is None:
                    out[name] = self._virtual.get(full, 0.0)
                    continue
                if want_speed:
                    out[name] = float(self.data.qvel[spec["dofadr"]])
                else:
                    raw = float(self.data.qpos[spec["qposadr"]])
                    out[name] = (
                        _normalize(raw, spec["mode"], spec["range_min"], spec["range_max"], spec["drive_mode"])
                        if normalize
                        else raw
                    )
        return out

    def write(self, module: str, values: Dict[str, Any], normalize: bool = True, snap: bool = False):
        from solarian.core.data import PosData
        import mujoco

        snapped = False
        with self._lock:
            for name, data in values.items():
                full = f"{module}/{name}"
                spec = self._component_specs.get(full)
                if spec is None or spec["kind"] != "joint":
                    continue
                if not isinstance(data, PosData):
                    # Only position control is supported for simulated servos.
                    continue
                value = data.value
                if spec.get("actuator_id") is None:
                    # Virtual joint: remember the (normalized) command so reads echo it.
                    self._virtual[full] = value
                    continue
                target = (
                    _unnormalize(value, spec["mode"], spec["range_min"], spec["range_max"], spec["drive_mode"])
                    if normalize
                    else value
                )
                self.data.ctrl[spec["actuator_id"]] = target
                if snap and spec.get("qposadr") is not None:
                    self.data.qpos[spec["qposadr"]] = target
                    snapped = True
                for follower in spec.get("followers") or []:
                    # Same normalized command, this joint's own exported limits.
                    # An unnormalized write is already in the primary joint's units.
                    if not normalize:
                        continue
                    followed = _unnormalize(
                        value,
                        spec["mode"],
                        follower["range_min"],
                        follower["range_max"],
                        follower["drive_mode"],
                    )
                    self.data.ctrl[follower["actuator_id"]] = followed
                    if snap:
                        self.data.qpos[follower["qposadr"]] = followed
                        snapped = True
            if snapped:
                mujoco.mj_forward(self.model, self.data)
        return None

    def _render(self, spec) -> np.ndarray:
        import mujoco

        h, w = int(spec["height"]), int(spec["width"])
        if self._render_failed or spec.get("camera_id") is None:
            return np.zeros((h, w, 3), dtype=np.uint8)

        dist = spec.get("distortion")
        # Render at (possibly supersampled) source size; distortion remaps it
        # down to the (h, w) output afterwards.
        if dist is not None:
            scale = max(1.0, float(getattr(dist, "render_scale", 1.0)))
            src_h, src_w = max(1, round(h * scale)), max(1, round(w * scale))
        else:
            src_h, src_w = h, w

        key = (spec["camera"], h, w)
        renderer = self._renderers.get(key)
        try:
            if renderer is None:
                renderer = self._new_renderer(src_h, src_w)
                self._renderers[key] = renderer
            with self._lock:
                renderer.update_scene(self.data, camera=spec["camera"])
            frame = renderer.render()
            if dist is not None:
                frame = self._distort(spec, dist, frame, h, w, src_h, src_w)
            return frame
        except Exception as exc:  # noqa: BLE001 - degrade gracefully when headless GL is unavailable
            if not self._render_failed:
                logger.warning("MuJoCo rendering unavailable (%s); returning blank frames.", exc)
                self._render_failed = True
            return np.zeros((h, w, 3), dtype=np.uint8)

    def _new_renderer(self, height: int, width: int):
        import mujoco

        # The GL offscreen framebuffer is sized from the model when the renderer
        # is created, and mujoco.Renderer rejects anything larger than it.
        vis = self.model.vis.global_
        vis.offwidth = max(int(vis.offwidth), width)
        vis.offheight = max(int(vis.offheight), height)
        # A scene may ask for a shadow map larger than this GPU's texture limit
        # (8192 is fine on a Mac, a Pi 5 stops at 4096), so it shrinks until
        # the context builds.
        quality = self.model.vis.quality
        while True:
            try:
                return mujoco.Renderer(self.model, height=height, width=width)
            except mujoco.FatalError as exc:
                if "Shadow framebuffer" not in str(exc) or quality.shadowsize <= 1024:
                    raise
                quality.shadowsize //= 2
                logger.warning("Shadow map too large for this GPU; retrying at %d.", quality.shadowsize)

    def _distort(self, spec, dist, frame, h, w, src_h, src_w) -> np.ndarray:
        """Warp a pinhole render into a calibrated fisheye-distorted image."""
        from solarian.sim import distortion

        key = (spec["camera"], h, w)
        maps = self._distort_maps.get(key)
        if maps is None:
            fovy = float(self.model.cam_fovy[spec["camera_id"]])
            map1, map2, outside_frac = distortion.build_fisheye_maps(
                dist, fovy, out_w=w, out_h=h, src_w=src_w, src_h=src_h
            )
            self._distort_maps[key] = (map1, map2)
            maps = (map1, map2)
            # A large black border means the source FOV is too narrow to feed
            # the fisheye; the user likely needs a wider CameraMount.fovy.
            if outside_frac > 0.02 and spec["camera"] not in self._distort_warned:
                self._distort_warned.add(spec["camera"])
                logger.warning(
                    "Camera '%s': %.0f%% of the fisheye output maps outside the "
                    "pinhole render (fovy=%.1f). Increase CameraMount.fovy to fill "
                    "the frame.",
                    spec["camera"], outside_frac * 100.0, fovy,
                )
        return distortion.apply(frame, maps[0], maps[1])

    # -- live overview viewer support ----------------------------------
    def overview_defaults(self, module: str) -> Dict[str, Any]:
        """Return a sensible free-camera framing for the whole scene.

        Called once by the :class:`~solarian.sim.viewer.SimViewer` process so
        its orbit camera starts framed on the model.
        """
        import mujoco

        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, cam)
        return {
            "azimuth": float(cam.azimuth),
            "elevation": float(cam.elevation),
            "distance": float(cam.distance),
            "lookat": [float(x) for x in cam.lookat],
            "show_frames": bool(self._scene.show_frames),
        }

    def render_overview(self, module: str, cam: Dict[str, Any]) -> np.ndarray:
        """Render the whole scene from a free (orbit) camera.

        Runs on the world server's main (RPC) thread, which owns the GL context
        -- rendering off the main thread is unsafe on macOS. The
        :class:`SimViewer` process drives ``cam`` (azimuth/elevation/distance/
        lookat) and displays the returned RGB frame.
        """
        import mujoco

        h = int(cam.get("height", 720))
        w = int(cam.get("width", 960))
        if self._overview_failed:
            return np.zeros((h, w, 3), dtype=np.uint8)

        key = ("__overview__", h, w)
        renderer = self._renderers.get(key)
        try:
            if renderer is None:
                renderer = self._new_renderer(h, w)
                self._renderers[key] = renderer
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth = float(cam.get("azimuth", 90.0))
            camera.elevation = float(cam.get("elevation", -20.0))
            camera.distance = float(cam.get("distance", 1.5))
            lookat = cam.get("lookat")
            if lookat is not None:
                camera.lookat = np.asarray(lookat, dtype=float)
            with self._lock:
                renderer.update_scene(self.data, camera=camera)
                try:
                    if cam.get("show_frames", self._scene.show_frames):
                        self._draw_frames(renderer.scene)
                    if self._scene.show_cameras:
                        self._draw_camera_frustums(renderer.scene)
                    self._draw_highlight_markers(renderer.scene)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Overview overlay failed (%s); drawing the mesh only.", exc)
                frame = renderer.render()
            return frame
        except Exception as exc:  # noqa: BLE001 - degrade gracefully when GL is unavailable
            if not self._overview_failed:
                logger.warning("Overview rendering unavailable (%s); viewer will blank.", exc)
                self._overview_failed = True
            return np.zeros((h, w, 3), dtype=np.uint8)

    # -- coordinate-frame overlays -------------------------------------
    #: Axis length (m) for the world / end-effector triads and arrow width (m).
    _WORLD_AXIS_LEN = 0.15
    _EE_AXIS_LEN = 0.08
    _SITE_AXIS_LEN = 0.04
    _AXIS_WIDTH = 0.004
    _SITE_AXIS_WIDTH = 0.002

    #: Frustum depth (m) and line/arrow widths (m) for the camera overlays.
    _CAM_FRUSTUM_DEPTH = 0.15
    _CAM_LINE_WIDTH = 0.002
    _CAM_ARROW_WIDTH = 0.004
    #: Fallback aspect (width / height) when a camera's render size is unknown.
    _CAM_DEFAULT_ASPECT = 4.0 / 3.0
    #: Cyan wireframe / yellow look-arrow so frustums stand out from the triads.
    _CAM_FRUSTUM_RGBA = (0.1, 0.9, 0.9, 1.0)
    _CAM_ARROW_RGBA = (1.0, 0.9, 0.1, 1.0)

    def highlight_joint(self, module: str, name: str | None) -> None:
        """Select a hinge to mark with a red axis arrow, or clear when ``name`` is empty."""
        if not name:
            self._highlight_key = None
            return
        self._highlight_key = f"{module}/{name}"

    def _highlight_spec(self):
        if not self._highlight_key:
            return None
        return self._component_specs.get(self._highlight_key)

    _HIGHLIGHT_ARROW_LEN = 0.20
    _HIGHLIGHT_ARROW_WIDTH = 0.010
    _HIGHLIGHT_SPHERE_R = 0.014
    _HIGHLIGHT_RGBA = (0.95, 0.12, 0.08, 1.0)

    def _draw_highlight_markers(self, scene) -> None:
        import mujoco

        spec = self._highlight_spec()
        if spec is None or spec.get("joint_id") is None:
            return
        frame = hinge_world_frame(
            self.data.xanchor,
            self.data.xaxis,
            int(spec["joint_id"]),
            int(spec["dofadr"]) if spec.get("dofadr") is not None else -1,
            njnt=int(self.model.njnt),
        )
        if frame is None:
            return
        origin, axis = frame
        start, tip = hinge_arrow_ends(origin, axis, self._HIGHLIGHT_ARROW_LEN)
        rgba = np.asarray(self._HIGHLIGHT_RGBA, dtype=np.float32)
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([self._HIGHLIGHT_SPHERE_R, 0.0, 0.0], dtype=float),
            origin,
            np.eye(3).ravel(),
            rgba,
        )
        scene.ngeom += 1
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            rgba,
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_ARROW, self._HIGHLIGHT_ARROW_WIDTH, start, tip
        )
        scene.ngeom += 1

    def _draw_frames(self, scene) -> None:
        """Append world, TCP and labelled site axis triads to the render scene."""
        self._add_triad(scene, np.zeros(3), np.eye(3).ravel(), self._WORLD_AXIS_LEN)
        for sid in self._site_frame_ids:
            self._add_triad(
                scene,
                self.data.site_xpos[sid],
                self.data.site_xmat[sid],
                self._SITE_AXIS_LEN,
                width=self._SITE_AXIS_WIDTH,
                label=self.model.site(sid).name,
            )
        for frame_type, fid in self._tcp_frame_ids:
            if frame_type == "site":
                pos, mat = self.data.site_xpos[fid], self.data.site_xmat[fid]
            elif frame_type == "geom":
                pos, mat = self.data.geom_xpos[fid], self.data.geom_xmat[fid]
            else:  # body
                pos, mat = self.data.xpos[fid], self.data.xmat[fid]
            self._add_triad(scene, pos, mat, self._EE_AXIS_LEN)

    def _add_triad(
        self, scene, pos, mat, length: float, *, width: float | None = None, label: str = ""
    ) -> None:
        """Draw an RGB (x/y/z) arrow triad at ``pos`` oriented by ``mat`` (3x3)."""
        import mujoco

        R = np.asarray(mat, dtype=float).reshape(3, 3)
        origin = np.asarray(pos, dtype=float)
        axes = (
            (R[:, 0], (1.0, 0.2, 0.2, 1.0)),  # x -> red
            (R[:, 1], (0.2, 1.0, 0.2, 1.0)),  # y -> green
            (R[:, 2], (0.3, 0.4, 1.0, 1.0)),  # z -> blue
        )
        for i, (direction, rgba) in enumerate(axes):
            if scene.ngeom >= scene.maxgeom:
                return
            tip = origin + direction * length
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3),
                np.zeros(3),
                np.zeros(9),
                np.asarray(rgba, dtype=np.float32),
            )
            mujoco.mjv_connector(
                geom, mujoco.mjtGeom.mjGEOM_ARROW, width or self._AXIS_WIDTH, origin, tip
            )
            if label and i == 0:
                geom.label = label
            scene.ngeom += 1

    def _camera_aspects(self) -> dict[str, float]:
        """Map composed camera name -> render aspect (width / height)."""
        aspects: dict[str, float] = {}
        for spec in self._component_specs.values():
            if spec.get("kind") != "camera":
                continue
            name = spec.get("camera")
            h, w = spec.get("height"), spec.get("width")
            if name and h:
                aspects[name] = float(w) / float(h)
        return aspects

    def _draw_camera_frustums(self, scene) -> None:
        """Append a wireframe frustum + look-arrow for every mounted camera.

        MuJoCo cameras look down their local -Z axis (local +X right, +Y up), so
        the frustum is built in world frame from ``cam_xpos``/``cam_xmat`` and
        the vertical FOV in ``cam_fovy``.
        """
        import mujoco

        aspects = self._camera_aspects()
        depth = self._CAM_FRUSTUM_DEPTH
        for cam in self._scene.cameras:
            cam_id = self._name2id(mujoco.mjtObj.mjOBJ_CAMERA, cam.name)
            if cam_id is None:
                continue
            origin = np.asarray(self.data.cam_xpos[cam_id], dtype=float)
            R = np.asarray(self.data.cam_xmat[cam_id], dtype=float).reshape(3, 3)
            right, up = R[:, 0], R[:, 1]
            forward = -R[:, 2]  # cameras look down local -Z

            fovy = float(self.model.cam_fovy[cam_id])
            aspect = aspects.get(cam.name, self._CAM_DEFAULT_ASPECT)
            half_h = depth * np.tan(np.deg2rad(fovy) / 2.0)
            half_w = half_h * aspect

            center = origin + forward * depth
            corners = [
                center + up * sy * half_h + right * sx * half_w
                for sy, sx in ((1, -1), (1, 1), (-1, 1), (-1, -1))
            ]

            # Four edges apex -> far corners, then the far-plane rectangle.
            edges = [(origin, c) for c in corners]
            edges += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
            for start, end in edges:
                if not self._add_line(
                    scene, start, end, self._CAM_LINE_WIDTH, self._CAM_FRUSTUM_RGBA
                ):
                    return

            # A short arrow along -Z makes the look direction unambiguous.
            self._add_arrow(
                scene, origin, origin + forward * depth,
                self._CAM_ARROW_WIDTH, self._CAM_ARROW_RGBA,
            )

    def _add_line(self, scene, start, end, width: float, rgba) -> bool:
        """Draw a single line segment; return ``False`` when the geom cap is hit."""
        import mujoco

        if scene.ngeom >= scene.maxgeom:
            return False
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_LINE, width,
            np.asarray(start, dtype=float), np.asarray(end, dtype=float),
        )
        scene.ngeom += 1
        return True

    def _add_arrow(self, scene, start, end, width: float, rgba) -> bool:
        """Draw a single arrow; return ``False`` when the geom cap is hit."""
        import mujoco

        if scene.ngeom >= scene.maxgeom:
            return False
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_ARROW, width,
            np.asarray(start, dtype=float), np.asarray(end, dtype=float),
        )
        scene.ngeom += 1
        return True

    # -- no-op hardware operations (all take the calling module first) --
    def configure(self, module: str, params: dict, component_names: list[str]):
        return None

    def enable_torque(self, module: str, component_names: list[str]):
        return None

    def disable_torque(self, module: str, component_names: list[str]):
        return None

    def is_calibrated(self, module: str) -> bool:
        return True

    def read_calibration(self, module: str) -> dict:
        return {}

    def write_calibration(self, module: str, calibration_dict: dict) -> dict:
        return calibration_dict

    def start_calibration_mode(self, module: str):
        return None

    def end_calibration_mode(self, module: str):
        return None

    def reset_calibration(self, module: str, component_names: list[str]):
        return None

    def setup_component(self, module: str, component_name: str, configs: dict):
        return None

    def set_half_turn_homings(self, module: str, component_names: list[str]) -> dict:
        return {}

    def drive_to_encoder_center(self, module: str, component_names: list[str]) -> dict:
        return {}

    def record_ranges_of_motion(self, module: str, component_names: list[str]):
        return {}, {}

    # ------------------------------------------------------------------
    def run(self):
        from solarian.utils import setup_process_logging

        setup_process_logging(logger_name=None, log_queue=self._log_queue, install_fallback=False)

        self._lock = threading.Lock()
        self._stop = threading.Event()

        try:
            self._build()
            logger.info(
                "MujocoWorldServer ready: %d qpos, %d actuators, %d components",
                self.model.nq,
                self.model.nu,
                len(self._component_specs),
            )
            self.ready_event.set()

            stepper = threading.Thread(target=self._stepping_loop, daemon=True)
            stepper.start()

            handlers = {
                "read": self.read,
                "write": self.write,
                "configure": self.configure,
                "enable_torque": self.enable_torque,
                "disable_torque": self.disable_torque,
                "is_calibrated": self.is_calibrated,
                "read_calibration": self.read_calibration,
                "write_calibration": self.write_calibration,
                "start_calibration_mode": self.start_calibration_mode,
                "end_calibration_mode": self.end_calibration_mode,
                "reset_calibration": self.reset_calibration,
                "setup_component": self.setup_component,
                "set_half_turn_homings": self.set_half_turn_homings,
                "drive_to_encoder_center": self.drive_to_encoder_center,
                "record_ranges_of_motion": self.record_ranges_of_motion,
                "overview_defaults": self.overview_defaults,
                "render_overview": self.render_overview,
                "highlight_joint": self.highlight_joint,
            }

            # Serve module RPC on the main thread. The caller module for each
            # connection is looked up so bare component names can be namespaced
            # per module (avoids collisions across modules, e.g. both arms'
            # ``shoulder_pan``).
            conn_to_module = {
                conn: self._conn_modules.get(key, "")
                for key, conn in self._server_conns.items()
            }
            conn_list = list(self._server_conns.values())
            while True:
                try:
                    ready = conn_wait(conn_list, timeout=1.0)
                except OSError:
                    break

                for conn in ready:
                    try:
                        method, args, kwargs = conn.recv()
                    except (EOFError, OSError):
                        conn_list = [c for c in conn_list if c is not conn]
                        continue

                    if method == _SHUTDOWN_SENTINEL:
                        return

                    module = conn_to_module.get(conn, "")
                    handler = handlers.get(method)
                    try:
                        if handler is None:
                            raise AttributeError(f"MujocoWorldServer has no operation '{method}'")
                        result = handler(module, *args, **kwargs)
                        conn.send((result, None))
                    except Exception as exc:  # noqa: BLE001 - forward to caller
                        try:
                            conn.send((None, exc))
                        except Exception:
                            conn.send((None, RuntimeError(str(exc))))

                if not conn_list:
                    break

        except Exception as exc:  # noqa: BLE001
            tb_str = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
            logger.exception("MujocoWorldServer crashed: %s", exc)
            self._error_queue.put_nowait(("mujoco_world", tb_str))
        finally:
            self.ready_event.set()
            self._stop.set()
