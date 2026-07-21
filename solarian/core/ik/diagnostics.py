"""Backend-agnostic IK / FK diagnostics.

Tools to validate that a :class:`~solarian.core.ik.base.DifferentialIKBackend`
-- and the MuJoCo/MJCF model behind it -- behaves correctly: forward
kinematics, per-axis Cartesian *translation* and *rotation* response, and
convergence to a target pose.

Everything here operates purely through the ``DifferentialIKBackend`` interface
(``set_q`` / ``get_pose`` / ``solve`` / ``integrate``), so it can validate any
module's IK backend without knowing anything about the specific robot. That
makes it the go-to sanity check when bringing up a new module: build the
module's backend and run :func:`diagnose_backend` to see, axis by axis, whether
commanding +x actually moves the end-effector along +x while holding
orientation, etc. If those responses are wrong, the model wiring (joint axes,
frame, limits) is the first suspect -- not the teleop/controller code.

Poses use the framework convention: a 6-vector ``(x, y, z, rx, ry, rz)`` where
``x, y, z`` is position in metres and ``(rx, ry, rz)`` is an axis-angle
(rotation-vector) orientation in radians.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from solarian.core.ik.base import DifferentialIKBackend

POSE_AXES = ("x", "y", "z", "rx", "ry", "rz")
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


# ---------------------------------------------------------------------------
# Low-level FK helpers
# ---------------------------------------------------------------------------
def home_q(backend: DifferentialIKBackend) -> np.ndarray:
    """Zero joint configuration sized for ``backend`` (radians)."""
    return np.zeros(len(backend.joint_actuator_names))


def fk(backend: DifferentialIKBackend, q: Sequence[float]) -> np.ndarray:
    """Forward kinematics: end-effector pose at joint configuration ``q``."""
    backend.set_q(np.asarray(q, dtype=float))
    return np.asarray(backend.get_pose(), dtype=float).copy()


def _rotation(pose: np.ndarray) -> Rotation:
    return Rotation.from_rotvec(np.asarray(pose, dtype=float)[3:6])


def orientation_error_rad(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    """Geodesic angle (radians) between the orientations of two poses."""
    rel = _rotation(pose_a) * _rotation(pose_b).inv()
    return float(np.linalg.norm(rel.as_rotvec()))


def position_error(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    """Euclidean distance (metres) between the positions of two poses."""
    return float(np.linalg.norm(np.asarray(pose_a)[:3] - np.asarray(pose_b)[:3]))


def ee_axes_world(pose: np.ndarray) -> np.ndarray:
    """Columns are the EE frame's x/y/z unit axes expressed in world coords."""
    return _rotation(pose).as_matrix()


# ---------------------------------------------------------------------------
# Closed-loop servoing (mirrors what IKActuatorBus does each control tick)
# ---------------------------------------------------------------------------
def servo_to_pose(
    backend: DifferentialIKBackend,
    target_pose: Sequence[float],
    q0: Sequence[float] | None = None,
    steps: int = 400,
    dt: float = 1.0 / 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run differential IK toward ``target_pose`` for ``steps`` iterations.

    Each iteration reproduces one control tick of
    :class:`~solarian.core.ik.bus.IKActuatorBus`: set the current config, solve
    for a velocity toward the target, integrate. Returns ``(final_pose,
    final_q)``.
    """
    q = home_q(backend) if q0 is None else np.asarray(q0, dtype=float).copy()
    target = np.asarray(target_pose, dtype=float)
    for _ in range(steps):
        backend.set_q(q)
        v = backend.solve(target, dt)
        q = np.asarray(backend.integrate(v, dt), dtype=float)
    return fk(backend, q), q


# ---------------------------------------------------------------------------
# Per-axis Cartesian response reports
# ---------------------------------------------------------------------------
@dataclass
class TranslationResponse:
    axis: str
    frame: str
    commanded: np.ndarray       # commanded displacement in world coords (m)
    achieved: np.ndarray        # achieved EE displacement in world coords (m)
    orientation_drift_rad: float
    start_pose: np.ndarray
    final_pose: np.ndarray

    @property
    def direction_cos(self) -> float:
        """Cosine between achieved and commanded displacement (1.0 == perfect)."""
        a, c = self.achieved, self.commanded
        na, nc = np.linalg.norm(a), np.linalg.norm(c)
        if na < 1e-12 or nc < 1e-12:
            return 0.0
        return float(np.dot(a, c) / (na * nc))

    @property
    def magnitude_ratio(self) -> float:
        """Achieved distance / commanded distance."""
        nc = np.linalg.norm(self.commanded)
        return float(np.linalg.norm(self.achieved) / nc) if nc > 1e-12 else 0.0


@dataclass
class RotationResponse:
    axis: str
    frame: str
    commanded_axis: np.ndarray  # world-frame unit rotation axis commanded
    achieved_rotvec: np.ndarray  # world-frame achieved rotation (axis-angle)
    commanded_angle: float
    position_drift: float
    start_pose: np.ndarray
    final_pose: np.ndarray

    @property
    def achieved_angle(self) -> float:
        return float(np.linalg.norm(self.achieved_rotvec))

    @property
    def direction_cos(self) -> float:
        rv = self.achieved_rotvec
        n = np.linalg.norm(rv)
        if n < 1e-12:
            return 0.0
        return float(np.dot(rv / n, self.commanded_axis))


def translation_response(
    backend: DifferentialIKBackend,
    axis: str,
    dist: float = 0.03,
    frame: str = "world",
    q0: Sequence[float] | None = None,
    steps: int = 400,
    dt: float = 1.0 / 120.0,
) -> TranslationResponse:
    """Command a pure translation along ``axis`` and measure the EE response.

    ``frame`` is ``"world"`` (move along a fixed world axis) or ``"ee"`` (move
    along the end-effector's own axis). Orientation is commanded to stay fixed.
    """
    if axis not in _AXIS_INDEX:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    p0 = fk(backend, home_q(backend) if q0 is None else q0)
    local = np.zeros(3)
    local[_AXIS_INDEX[axis]] = dist
    delta_world = local if frame == "world" else ee_axes_world(p0) @ local

    target = p0.copy()
    target[:3] = p0[:3] + delta_world
    pf, _ = servo_to_pose(backend, target, q0=q0, steps=steps, dt=dt)
    return TranslationResponse(
        axis=axis,
        frame=frame,
        commanded=delta_world,
        achieved=pf[:3] - p0[:3],
        orientation_drift_rad=orientation_error_rad(pf, p0),
        start_pose=p0,
        final_pose=pf,
    )


def rotation_response(
    backend: DifferentialIKBackend,
    axis: str,
    angle: float = 0.2,
    frame: str = "world",
    q0: Sequence[float] | None = None,
    steps: int = 400,
    dt: float = 1.0 / 120.0,
) -> RotationResponse:
    """Command a pure rotation about ``axis`` and measure the EE response.

    ``frame`` is ``"world"`` (rotate about a fixed world axis) or ``"ee"``
    (rotate about the end-effector's own axis). Position is commanded to stay
    fixed.
    """
    if axis not in _AXIS_INDEX:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    p0 = fk(backend, home_q(backend) if q0 is None else q0)
    r0 = _rotation(p0)
    unit = np.zeros(3)
    unit[_AXIS_INDEX[axis]] = 1.0
    axis_world = unit if frame == "world" else r0.apply(unit)

    r_delta = Rotation.from_rotvec(axis_world * angle)
    target = p0.copy()
    target[3:6] = (r_delta * r0).as_rotvec()
    pf, _ = servo_to_pose(backend, target, q0=q0, steps=steps, dt=dt)

    achieved = (_rotation(pf) * r0.inv()).as_rotvec()
    return RotationResponse(
        axis=axis,
        frame=frame,
        commanded_axis=axis_world,
        achieved_rotvec=achieved,
        commanded_angle=angle,
        position_drift=position_error(pf, p0),
        start_pose=p0,
        final_pose=pf,
    )


# ---------------------------------------------------------------------------
# Joint sensitivity (are all joints "alive" and distinct?)
# ---------------------------------------------------------------------------
def joint_sensitivity(
    backend: DifferentialIKBackend,
    q0: Sequence[float] | None = None,
    delta: float = 0.05,
) -> dict[str, float]:
    """EE displacement magnitude (m) from perturbing each joint by ``delta``.

    A near-zero entry means that joint does not move the end-effector at the
    tested configuration -- usually a sign of a mis-mapped joint name or a
    broken model.
    """
    q_base = home_q(backend) if q0 is None else np.asarray(q0, dtype=float)
    p_base = fk(backend, q_base)
    out: dict[str, float] = {}
    for i, name in enumerate(backend.joint_actuator_names):
        q = q_base.copy()
        q[i] += delta
        out[name] = position_error(fk(backend, q), p_base) + orientation_error_rad(
            fk(backend, q), p_base
        )
    return out


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------
def convergence(
    backend: DifferentialIKBackend,
    target_pose: Sequence[float],
    q0: Sequence[float] | None = None,
    steps: int = 800,
    dt: float = 1.0 / 120.0,
) -> tuple[float, float, np.ndarray]:
    """Servo to ``target_pose``; return ``(pos_error_m, ori_error_rad, final_pose)``."""
    pf, _ = servo_to_pose(backend, target_pose, q0=q0, steps=steps, dt=dt)
    target = np.asarray(target_pose, dtype=float)
    return position_error(pf, target), orientation_error_rad(pf, target), pf


# ---------------------------------------------------------------------------
# One-shot report
# ---------------------------------------------------------------------------
@dataclass
class BackendReport:
    joint_names: list[str]
    joint_limits: dict[str, tuple[float, float]]
    home_pose: np.ndarray
    translations: dict[str, TranslationResponse] = field(default_factory=dict)
    rotations: dict[str, RotationResponse] = field(default_factory=dict)
    joint_sensitivity: dict[str, float] = field(default_factory=dict)

    def format(self) -> str:
        np.set_printoptions(precision=4, suppress=True)
        lines = ["IK backend diagnostics", "=" * 60]
        lines.append(f"joints: {self.joint_names}")
        lines.append("limits (rad):")
        for n, (lo, hi) in self.joint_limits.items():
            lines.append(f"  {n:<14} [{lo:+.3f}, {hi:+.3f}]")
        lines.append(f"home pose (x,y,z,rx,ry,rz): {self.home_pose}")
        lines.append("\ntranslation response (cos should be ~1, drift ~0):")
        for r in self.translations.values():
            lines.append(
                f"  {r.frame:>5} +{r.axis}: cos={r.direction_cos:+.3f} "
                f"mag_ratio={r.magnitude_ratio:.3f} "
                f"ori_drift={r.orientation_drift_rad:.4f} rad"
            )
        lines.append("\nrotation response (cos should be ~1, drift ~0):")
        for r in self.rotations.values():
            lines.append(
                f"  {r.frame:>5} +{r.axis}: cos={r.direction_cos:+.3f} "
                f"angle={r.achieved_angle:.3f}/{r.commanded_angle:.3f} "
                f"pos_drift={r.position_drift:.4f} m"
            )
        lines.append("\njoint sensitivity (EE change per +0.05 perturbation):")
        for n, v in self.joint_sensitivity.items():
            lines.append(f"  {n:<14} {v:.4f}")
        return "\n".join(lines)


def diagnose_backend(
    backend: DifferentialIKBackend,
    frame: str = "world",
    trans_dist: float = 0.03,
    rot_angle: float = 0.2,
) -> BackendReport:
    """Run the full battery and return a printable :class:`BackendReport`.

    Handy in a REPL or a test bring-up::

        from solarian.core.ik.diagnostics import diagnose_backend
        print(diagnose_backend(MyArm.ik_backend.build()).format())
    """
    report = BackendReport(
        joint_names=list(backend.joint_actuator_names),
        joint_limits=dict(backend.joint_limits),
        home_pose=fk(backend, home_q(backend)),
    )
    for axis in ("x", "y", "z"):
        report.translations[axis] = translation_response(
            backend, axis, dist=trans_dist, frame=frame
        )
        report.rotations[axis] = rotation_response(
            backend, axis, angle=rot_angle, frame=frame
        )
    report.joint_sensitivity = joint_sensitivity(backend)
    return report
