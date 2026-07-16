"""MuJoCo scene composition from module MJCF models.

Builds a single ``MjModel`` from a :class:`~solarian.sim.configs.SceneConfig`:
starts from the environment world, attaches each module's MJCF sub-model at its
declared pose (namespacing joints/actuators/cameras via a prefix), and mounts
extra cameras. ``mujoco`` is imported lazily so importing ``solarian`` never
requires the optional ``[sim]`` dependencies.
"""

from __future__ import annotations

import os

from solarian.sim.configs import SceneConfig


# Minimal environment used when a scene declares no ``world_mjcf``.
_DEFAULT_WORLD_MJCF = """
<mujoco model="sim_world">
  <compiler angle="radian"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="2 2 0.05" material="grid"/>
  </worldbody>
</mujoco>
"""


def resolve_mjcf(mjcf: str) -> str:
    """Resolve an MJCF reference to an absolute file path.

    Accepts an absolute path, a path relative to ``SOLARIAN_PROJECT_PATH``, or
    the name of a ``robot_descriptions`` module (e.g.
    ``"so_arm101_mj_description"``) whose ``MJCF_PATH`` is used.
    """
    if os.path.isabs(mjcf) and os.path.exists(mjcf):
        return mjcf

    project_path = os.environ.get("SOLARIAN_PROJECT_PATH", os.getcwd())
    candidate = os.path.join(project_path, mjcf)
    if os.path.exists(candidate):
        return candidate

    if os.path.exists(mjcf):
        return os.path.abspath(mjcf)

    if mjcf.endswith(".xml"):
        raise FileNotFoundError(f"MJCF file not found: {mjcf}")

    # Treat as a robot_descriptions module name.
    import importlib

    try:
        module = importlib.import_module(f"robot_descriptions.{mjcf}")
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not resolve MJCF '{mjcf}': not a file and not a "
            f"robot_descriptions module. Install robot_descriptions or point "
            f"'mjcf' at an .xml file."
        ) from exc
    return module.MJCF_PATH


def _load_spec(mjcf: str | None):
    import mujoco

    if mjcf is None:
        return mujoco.MjSpec.from_string(_DEFAULT_WORLD_MJCF)
    return mujoco.MjSpec.from_file(resolve_mjcf(mjcf))


def _find_body(spec, name: str):
    if name in ("world", "worldbody", ""):
        return spec.worldbody
    body = spec.body(name)
    if body is None:
        raise ValueError(f"Body '{name}' not found in composed scene")
    return body


def compose_scene(scene: SceneConfig):
    """Compose and compile a scene into a MuJoCo ``MjModel``.

    Returns:
        ``mujoco.MjModel`` for the fully composed scene.
    """
    import mujoco

    world = _load_spec(scene.world_mjcf)

    for att in scene.attachments:
        child = mujoco.MjSpec.from_file(resolve_mjcf(att.mjcf))
        parent = _find_body(world, att.parent_body)
        frame = parent.add_frame(pos=att.pos, quat=att.quat)
        world.attach(child, frame=frame, prefix=att.prefix)

    for cam in scene.cameras:
        body = _find_body(world, cam.parent_body)
        body.add_camera(name=cam.name, pos=cam.pos, quat=cam.quat, fovy=cam.fovy)

    if scene.gravity is not None:
        world.option.gravity = scene.gravity

    # Enlarge the offscreen framebuffer so camera and overview renders aren't
    # capped at MuJoCo's 640x480 default (which otherwise fails larger renders).
    try:
        world.visual.global_.offwidth = max(int(world.visual.global_.offwidth), 1920)
        world.visual.global_.offheight = max(int(world.visual.global_.offheight), 1080)
    except Exception:  # noqa: BLE001 - older MjSpec layouts; fall back to XML default
        pass

    model = world.compile()

    if scene.timestep is not None:
        model.opt.timestep = scene.timestep

    return model
