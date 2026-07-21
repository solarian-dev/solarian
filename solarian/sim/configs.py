"""Configuration models for the MuJoCo-backed simulation framework.

These are plain pydantic models (picklable) so they can be passed to the
``MujocoWorldServer`` process. They describe how a robot's simulated modules
are composed into a single MuJoCo scene and how each declared component maps to
a MuJoCo joint / actuator / camera.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModuleAttachment(BaseModel):
    """Declares an MJCF model contributed by a module and how it attaches.

    A module composes its own MJCF sub-model into the shared scene. The
    sub-model is attached to a body of the (already-composed) scene at the
    given pose. ``prefix`` namespaces every joint / actuator / camera in the
    child model so that models with colliding names can coexist.
    """

    module: str
    #: Path (absolute, or relative to ``SOLARIAN_PROJECT_PATH``) to an MJCF
    #: ``.xml`` file, or the name of a ``robot_descriptions`` module
    #: (e.g. ``"so_arm101_mj_description"``) to resolve at runtime.
    mjcf: str
    #: Body in the scene to attach to. ``"world"`` attaches to the worldbody.
    parent_body: str = "world"
    pos: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    quat: list[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    #: Name prefix applied to the child model's joints/actuators/cameras.
    prefix: str = ""


class CameraMount(BaseModel):
    """A MuJoCo camera added to a body of the composed scene."""

    name: str
    parent_body: str
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
