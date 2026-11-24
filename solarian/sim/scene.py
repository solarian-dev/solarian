"""MuJoCo scene composition from module MJCF models.

Builds a single ``MjModel`` from a :class:`~solarian.sim.configs.SceneConfig`:
starts from the environment world, attaches each module's MJCF sub-model either
at a body pose or by mating two sites (namespacing joints/actuators/cameras via
a prefix), and mounts extra cameras. ``mujoco`` is imported lazily so importing
``solarian`` never requires the optional ``[sim]`` dependencies.
"""

from __future__ import annotations

import os

from solarian.sim.configs import ModuleAttachment, SceneConfig


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

# Used when a scene sets ``backdrop=False``. ``compose_scene`` adds its floor
# after framing the scene on the modules.
_EMPTY_WORLD_MJCF = """
<mujoco model="sim_world">
  <compiler angle="radian"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
  </worldbody>
</mujoco>
"""


def isolate_module(scene: SceneConfig, module: str) -> SceneConfig:
    """``scene`` cut down to ``module``'s own model, at the origin of a bare world.

    Whatever the module is mounted on or carries in the full scene is left out,
    so a site-mated module (a gripper on a flange) stands alone too.
    """
    attachment = next((a for a in scene.attachments if a.module == module), None)
    if attachment is None:
        raise ValueError(f"Module '{module}' has no attachment in the scene")
    return SceneConfig(
        backdrop=False,
        attachments=[
            ModuleAttachment(module=module, mjcf=attachment.mjcf, prefix=attachment.prefix)
        ],
        show_frames=scene.show_frames,
        show_cameras=False,
    )


def scene_without(scene: SceneConfig, modules: set[str]) -> SceneConfig:
    """``scene`` with ``modules``' attachments removed, and whatever is mounted on them.

    An attachment mated to a site under a removed prefix goes too (a gripper
    on a removed arm's flange), and so does a camera mounted on one. The
    module of an attachment removed that way is not in ``modules``; the
    caller finds it by comparing attachments.
    """
    dropped = {a.prefix for a in scene.attachments if a.module in modules and a.prefix}
    kept = [a for a in scene.attachments if a.module not in modules]

    def under_dropped(name: str | None) -> bool:
        return bool(name) and any(name.startswith(prefix) for prefix in dropped)

    while True:
        orphans = [a for a in kept if under_dropped(a.parent_site) or under_dropped(a.parent_body)]
        if not orphans:
            break
        dropped |= {a.prefix for a in orphans if a.prefix}
        kept = [a for a in kept if a not in orphans]
    cameras = [
        c for c in scene.cameras if not (under_dropped(c.site) or under_dropped(c.parent_body))
    ]
    return scene.model_copy(update={"attachments": kept, "cameras": cameras})


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


def _compose_poses(pos_a, quat_a, pos_b, quat_b):
    """Compose ``T_a * T_b`` → ``(pos, quat)`` as numpy arrays."""
    import mujoco
    import numpy as np

    pos_a = np.asarray(pos_a, dtype=float)
    quat_a = np.asarray(quat_a, dtype=float)
    pos_b = np.asarray(pos_b, dtype=float)
    quat_b = np.asarray(quat_b, dtype=float)

    rotated = np.zeros(3)
    mujoco.mju_rotVecQuat(rotated, pos_b, quat_a)
    pos = pos_a + rotated

    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, quat_a, quat_b)
    return pos, quat


def _invert_pose(pos, quat):
    """Invert a pose ``(pos, quat)`` → ``(pos, quat)`` as numpy arrays."""
    import mujoco
    import numpy as np

    pos = np.asarray(pos, dtype=float)
    quat = np.asarray(quat, dtype=float)
    qinv = np.zeros(4)
    mujoco.mju_negQuat(qinv, quat)
    pinv = np.zeros(3)
    mujoco.mju_rotVecQuat(pinv, -pos, qinv)
    return pinv, qinv


def _compose_site_pose(site, off_pos, off_quat):
    """Camera pose in the site's body frame: the site frame composed with an
    optional local offset (``off_pos``/``off_quat``, relative to the site).

    Returns ``(pos, quat)`` as plain lists suitable for ``add_camera``.
    """
    pos, quat = _compose_poses(site.pos, site.quat, off_pos, off_quat)
    return pos.tolist(), quat.tolist()


def _site_pose_in_root(spec, site):
    """Pose of ``site`` relative to ``spec``'s worldbody (default body poses).

    Walks the body chain from the site's parent up to the worldbody, composing
    each body's ``pos``/``quat``. Joint DOFs are ignored (default configuration),
    which is correct for fixed mating frames exported from CAD.
    """
    import numpy as np

    pos = np.asarray(site.pos, dtype=float)
    quat = np.asarray(site.quat, dtype=float)
    body = site.parent
    while body is not None and body is not spec.worldbody:
        pos, quat = _compose_poses(body.pos, body.quat, pos, quat)
        parent = body.parent
        # worldbody.parent is itself; stop rather than loop forever.
        if parent is body:
            break
        body = parent
    return pos, quat


def _attachment_frame(world, child, att):
    """Resolve the parent body + local frame pose for one ``ModuleAttachment``.

    Body mode: frame on ``parent_body`` at ``pos``/``quat``.
    Site-mate mode: frame on the parent site's body at
    ``T_parent_site * T_offset * inv(T_child_site)``, so the two sites coincide
    (with ``pos``/``quat`` as an optional offset in the parent-site frame).
    """
    if att.parent_site is None:
        parent = _find_body(world, att.parent_body)
        return parent, list(att.pos), list(att.quat)

    parent_site = world.site(att.parent_site)
    if parent_site is None:
        raise ValueError(
            f"Site '{att.parent_site}' not found in composed scene (needed to "
            f"attach module '{att.module}'). Sites exported from a module "
            f"model are namespaced by its attach prefix."
        )

    child_site = child.site(att.child_site)
    if child_site is None:
        raise ValueError(
            f"Site '{att.child_site}' not found in child MJCF for module "
            f"'{att.module}' (needed to mate with parent_site "
            f"'{att.parent_site}')."
        )

    # T_attach = T_parent_site * T_offset * inv(T_child_site_in_root)
    child_in_root = _site_pose_in_root(child, child_site)
    mid = _compose_poses(parent_site.pos, parent_site.quat, att.pos, att.quat)
    attach_pos, attach_quat = _compose_poses(*mid, *_invert_pose(*child_in_root))
    return parent_site.parent, attach_pos.tolist(), attach_quat.tolist()


def compose_scene(scene: SceneConfig):
    """Compose and compile a scene into a MuJoCo ``MjModel``.

    Returns:
        ``mujoco.MjModel`` for the fully composed scene.
    """
    import mujoco

    if scene.backdrop:
        world = _load_spec(scene.world_mjcf)
    else:
        world = mujoco.MjSpec.from_string(_EMPTY_WORLD_MJCF)

    for att in scene.attachments:
        child = mujoco.MjSpec.from_file(resolve_mjcf(att.mjcf))
        parent, pos, quat = _attachment_frame(world, child, att)
        frame = parent.add_frame(pos=pos, quat=quat)
        world.attach(child, frame=frame, prefix=att.prefix)

    for cam in scene.cameras:
        if cam.site:
            site = world.site(cam.site)
            if site is None:
                raise ValueError(
                    f"Site '{cam.site}' not found in composed scene (needed to "
                    f"mount camera '{cam.name}'). Sites exported from a module "
                    f"model are namespaced by its attach prefix."
                )
            body = site.parent
            pos, quat = _compose_site_pose(site, cam.pos, cam.quat)
            body.add_camera(name=cam.name, pos=pos, quat=quat, fovy=cam.fovy)
        else:
            body = _find_body(world, cam.parent_body)
            body.add_camera(name=cam.name, pos=cam.pos, quat=cam.quat, fovy=cam.fovy)

    if scene.gravity is not None:
        world.option.gravity = scene.gravity

    if not scene.backdrop:
        # The infinite floor would count as a 2 m extent, and the viewer frames
        # its camera on the extent, so the scene keeps the modules' own.
        framed = world.compile()
        world.stat.center = framed.stat.center
        world.stat.extent = framed.stat.extent
        world.worldbody.add_geom(
            name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid"
        )

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
