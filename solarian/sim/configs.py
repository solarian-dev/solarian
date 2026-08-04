"""Configuration models for the MuJoCo-backed simulation framework.

These are plain pydantic models (picklable) so they can be passed to the
``MujocoWorldServer`` process. They describe how a robot's simulated modules
are composed into a single MuJoCo scene and how each declared component maps to
a MuJoCo joint / actuator / camera.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ModuleAttachment(BaseModel):
    """Declares an MJCF model contributed by a module and how it attaches.

    A module composes its own MJCF sub-model into the shared scene.
    ``prefix`` namespaces every joint / actuator / camera in the child model
    so that models with colliding names can coexist.

    Two attachment modes (pick one):

    1. **Body + pose** — place the child's worldbody origin on ``parent_body``
       at ``pos``/``quat`` (the original API).
    2. **Site mate** — coincide ``parent_site`` (already in the composed scene,
       namespaced by a prior attach prefix) with ``child_site`` (unprefixed
       name in the child MJCF). Typical for mating matching Onshape frames
       (e.g. both named ``side_bracket``). ``pos``/``quat`` become an optional
       offset applied in the parent-site frame after the mate.
    """

    module: str
    #: Path (absolute, or relative to ``SOLARIAN_PROJECT_PATH``) to an MJCF
    #: ``.xml`` file, or the name of a ``robot_descriptions`` module
    #: (e.g. ``"so_arm101_mj_description"``) to resolve at runtime.
    mjcf: str
    #: Body in the scene to attach to. ``"world"`` attaches to the worldbody.
    #: Ignored when ``parent_site`` is set.
    parent_body: str = "world"
    #: Site/frame already in the composed scene to mate against (e.g.
    #: ``"right/side_bracket"``). Requires ``child_site``.
    parent_site: str | None = None
    #: Site/frame in the child MJCF (unprefixed) that should coincide with
    #: ``parent_site`` (e.g. ``"side_bracket"``). Requires ``parent_site``.
    child_site: str | None = None
    #: Attachment position, or offset in the parent-site frame when mating.
    pos: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    #: Attachment orientation ``[w, x, y, z]``, or offset when mating.
    quat: list[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    #: Name prefix applied to the child model's joints/actuators/cameras.
    prefix: str = ""

    @model_validator(mode="after")
    def _sites_come_as_a_pair(self):
        if (self.parent_site is None) ^ (self.child_site is None):
            raise ValueError(
                "parent_site and child_site must be set together (site-mate "
                "mode), or both left unset (parent_body + pos/quat mode)"
            )
        return self


class CameraMount(BaseModel):
    """A MuJoCo camera added to the composed scene.

    Mount the camera either directly on a body (``parent_body``) or at a named
    ``site`` (frame) already present in the composed scene -- e.g. a frame
    exported from Onshape by ``onshape-to-robot`` (which becomes a ``<site>``),
    namespaced by the owning module's attach ``prefix``. When ``site`` is set it
    takes precedence over ``parent_body``, the camera is placed on the site's
    own body, and ``pos``/``quat`` are interpreted as an offset relative to the
    site frame (defaults leave the camera exactly on the frame). This keeps the
    mount location as a single source of truth in the CAD/MJCF model.

    Note: a MuJoCo camera looks down its local -Z axis (with +X right, +Y up).
    If the exported frame's axes don't already match that convention, use
    ``quat`` to rotate the camera onto the frame.
    """

    name: str
    #: Body to attach to (e.g. ``"arm/gripper"``). Ignored when ``site`` is set.
    parent_body: str = ""
    #: Name of a site/frame in the composed scene to mount at (prefixed by the
    #: owning module). When set, the camera is placed on the site's body at the
    #: site's pose, composed with ``pos``/``quat`` as an offset.
    site: str | None = None
    pos: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    quat: list[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    fovy: float = 58.0


class SceneConfig(BaseModel):
    """Declares the simulated environment an ``is_sim`` robot lives in.

    The scene starts from ``world_mjcf`` (the environment: ground, lights,
    objects the robot interacts with). Each module's MJCF is then attached to
    it, and extra cameras (e.g. a wrist camera) are mounted onto composed
    bodies.
    """

    #: MJCF for the environment (floor, lights, manipulable objects). If
    #: ``None`` a minimal ground+light world is generated.
    world_mjcf: str | None = None
    attachments: list[ModuleAttachment] = Field(default_factory=list)
    cameras: list[CameraMount] = Field(default_factory=list)
    #: Optional override of the physics timestep (seconds).
    timestep: float | None = None
    #: Optional override of gravity.
    gravity: list[float] | None = None
    #: Draw coordinate-frame triads in the overview viewer: the world frame at
    #: the origin and the end-effector frame of every arm that declares an IK
    #: backend. Set ``False`` to hide them.
    show_frames: bool = True
    #: Draw a wireframe frustum for every mounted camera in the overview viewer
    #: so its position and viewing direction are visible. Set ``False`` to hide.
    show_cameras: bool = True
