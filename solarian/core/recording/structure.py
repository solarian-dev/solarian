"""The robot's expanded structure, recorded beside the episodes it produced.

``session.json`` already says which columns exist and what units they carry.
What it cannot say from the schema alone is what produced them: that
``obs.right_arm.x`` is one axis of an end-effector pose solved by IK over six
named MuJoCo joints, that ``obs.right_gripper.aperture_mm`` is the opening of a
tool mated to that arm's flange, or that the pose is expressed in the arm's own
base frame rather than a lab world frame. Without that, a consumer is left
guessing from column names, which is how a viewer ends up hard-coding one
robot's vocabulary.

So the block below is built by introspecting the live robot at session start:
modules with their kind, the components behind each column, the IK joint
mapping, and the attachment graph. It is what lets a reader draw a gripper
trajectory, and later an arm, without being told which robot it is looking at.

MJCF and STL assets are *named*, not embedded — the Solarian Arm 1 mesh tree alone is
about 9.7 MB per session, and a reader that wants meshes can resolve the paths
itself. The joint mapping and the attachment graph are enough to animate a
model it already has.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from solarian.core.calibration import MotorNormMode
from solarian.core.configs import ActuatorConfig
from solarian.core.gripper_module import APERTURE_UNIT
from solarian.core.ik.actuator import CARTESIAN_AXES
from solarian.core.ik.spec import MinkIK

#: Ends of the normalized span each mode maps a servo's travel onto. ``raw``
#: (the Cartesian axes) and any unimplemented mode are absent.
_NORMALIZED_RANGE = {
    MotorNormMode.RANGE_M100_100.value: (-100.0, 100.0),
    MotorNormMode.RANGE_0_100.value: (0.0, 100.0),
}

#: Where a module's Cartesian pose columns are measured. ``MinkBackend.get_pose``
#: resolves the TCP frame against the *uncomposed* arm model, so the origin is
#: the arm's own MJCF base -- not the scene world, and not the attachment pose
#: the scene graph gives that base.
_POSE_FRAME = "arm_mjcf_base"


def build_structure(robot) -> dict:
    """Describe *robot* well enough to interpret an episode recorded from it."""
    structure = {
        "class": type(robot).__name__,
        "is_sim": bool(robot.is_sim),
        "modules": {
            name: _module(module) for name, module in robot.modules.items()
        },
    }
    if robot.scene is not None:
        # show_frames / show_cameras are viewer preferences, not structure.
        scene = robot.scene.model_dump(exclude={"show_frames", "show_cameras"})
        for attachment in scene["attachments"]:
            attachment["mjcf"] = _asset_ref(attachment["mjcf"])
        structure["scene"] = scene
    return structure


def _asset_ref(mjcf: str) -> str:
    """A portable reference to an MJCF file.

    A plugin resolves its assets to an absolute path, which names a directory on
    the recording machine and nothing on any other. The path relative to the
    containing package's parent -- ``solarian_arm1/assets/arm1.xml`` -- names
    the same file wherever that plugin is installed. Anything already relative
    (a project path, or a ``robot_descriptions`` module name) passes through.
    """
    path = Path(mjcf)
    if not path.is_absolute():
        return mjcf
    # Outermost ancestor that is a package, so an intermediate `assets/`
    # directory does not cut the search short.
    package = next(
        (p for p in reversed(path.parents) if (p / "__init__.py").exists()), None
    )
    if package is None:
        return mjcf
    return str(path.relative_to(package.parent))


def _module(module) -> dict:
    entry = {
        "class": type(module).__name__,
        "kind": module.kind,
        "components": _components(module),
    }
    kinematics = _kinematics(module)
    if kinematics is not None:
        entry["kinematics"] = kinematics
    # Only a gripper has one, and it is what maps a recorded aperture onto a
    # mesh: the ends of travel the aperture table or endpoints resolved to. The
    # unit comes along because it is how a consumer finds the aperture *column* --
    # the gripper's other column is the servo's normalized position.
    if module.kind == "gripper":
        low, high = module.scalar_limits()
        entry["aperture"] = {
            "range": [float(low), float(high)],
            "unit": APERTURE_UNIT,
        }
    return entry


def _components(module) -> dict:
    """Actuators in declaration order, then sensors that are not actuators.

    Declaration order is the order the schema walked, so this lines up with the
    columns. A servo appears once: it is the same object in both dicts.
    """
    components = dict(module.actuators)
    components.update(
        {name: s for name, s in module.sensors.items() if name not in components}
    )
    return {name: _component(c) for name, c in components.items()}


def _component(component) -> dict:
    entry = {"class": type(component).__name__, "kind": component.kind}
    cfg = component.cfg

    if component.kind == "cartesian_axis":
        entry["axis"] = CARTESIAN_AXES[int(cfg.id)]
        return entry

    # A sensor's cfg is a SensorConfig, with neither a model nor a
    # normalization; a camera's geometry is in the session's `cameras` block.
    if not isinstance(cfg, ActuatorConfig):
        return entry

    if cfg.model is not None:
        entry["model"] = cfg.model
    span = _NORMALIZED_RANGE.get(cfg.mode)
    if span is not None:
        entry["normalization"] = {
            "mode": cfg.mode,
            "range": list(span),
            # The span covers this MJCF joint's limits in the scene's model,
            # which is how a normalized reading becomes an angle.
            "joint": cfg.sim_joint,
        }
        if cfg.sim_range_min is not None or cfg.sim_range_max is not None:
            # Either end may be None: that end keeps the model's limit.
            entry["normalization"]["joint_range_override"] = [
                cfg.sim_range_min,
                cfg.sim_range_max,
            ]
    return entry


def _kinematics(module) -> dict | None:
    """The module's IK spec, or ``None`` when it solves no Cartesian pose."""
    # Only an ArmModule declares the attribute at all.
    spec = getattr(module, "ik_backend", None)
    if spec is None:
        return None
    if not isinstance(spec, MinkIK):
        # ArmModule also accepts a ready-built backend, whose parameters are not
        # declarative and cannot be read back. Name it rather than claim none.
        return {"backend": type(spec).__name__, "pose_frame": _POSE_FRAME}
    kinematics = {
        "mjcf": _asset_ref(spec.mjcf),
        # Ordered actuator name -> MJCF joint name. The order is the solver's
        # joint vector, which is why it is recorded as a mapping and not a set.
        "joints": dict(spec.joints),
        "frame": spec.frame,
        "frame_type": spec.frame_type,
        "pose_frame": _POSE_FRAME,
    }
    if spec.tcp_offset is not None:
        kinematics["tcp_offset"] = asdict(spec.tcp_offset)
    return kinematics
