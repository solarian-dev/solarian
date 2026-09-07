"""End-effector pose helpers (axis-angle + quaternions).

Cartesian commands stored on the wire are world-frame ``(x, y, z, rx, ry, rz)``
(metres + rotvec). Teleop jogs and ``MOVE`` with ``frame="ee"`` apply deltas
in the tool frame, then convert back to that world 6-vector.
"""

from __future__ import annotations

import numpy as np


def rotvec_to_quat(rotvec) -> np.ndarray:
    """Axis-angle (rotation vector) -> unit quaternion ``[w, x, y, z]``."""
    rotvec = np.asarray(rotvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotvec / theta
    half = theta / 2.0
    return np.concatenate(([np.cos(half)], axis * np.sin(half)))


def quat_to_rotvec(q) -> np.ndarray:
    """Unit quaternion ``[w, x, y, z]`` -> axis-angle (rotation vector)."""
    q = np.asarray(q, dtype=float).reshape(4)
    if q[0] < 0.0:
        q = -q
    w = float(np.clip(q[0], -1.0, 1.0))
    sin_half = float(np.linalg.norm(q[1:]))
    if sin_half < 1e-9:
        return np.zeros(3)
    theta = 2.0 * np.arctan2(sin_half, w)
    return (q[1:] / sin_half) * theta


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product of two ``[w, x, y, z]`` quaternions."""
    w1, x1, y1, z1 = np.asarray(a, dtype=float).reshape(4)
    w2, x2, y2, z2 = np.asarray(b, dtype=float).reshape(4)
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_rotate(q, v) -> np.ndarray:
    """Rotate 3-vector ``v`` by unit quaternion ``q`` (``q v q*``)."""
    qv = np.concatenate(([0.0], np.asarray(v, dtype=float).reshape(3)))
    q_conj = np.asarray(q, dtype=float).reshape(4) * np.array([1.0, -1.0, -1.0, -1.0])
    return quat_mul(quat_mul(q, qv), q_conj)[1:]


def apply_ee_translation(pose: np.ndarray, axis: int, dist: float) -> np.ndarray:
    """Translate ``pose`` by ``dist`` metres along the EE's own ``axis`` (0/1/2)."""
    pose = np.asarray(pose, dtype=float).copy().reshape(6)
    delta_local = np.zeros(3)
    delta_local[axis] = dist
    pose[:3] += quat_rotate(rotvec_to_quat(pose[3:6]), delta_local)
    return pose


def apply_ee_rotation(pose: np.ndarray, axis: int, angle: float) -> np.ndarray:
    """Rotate ``pose`` by ``angle`` radians about the EE's own ``axis`` (intrinsic)."""
    pose = np.asarray(pose, dtype=float).copy().reshape(6)
    delta = np.zeros(3)
    delta[axis] = angle
    q_new = quat_mul(rotvec_to_quat(pose[3:6]), rotvec_to_quat(delta))
    pose[3:6] = quat_to_rotvec(q_new)
    return pose


def compose_ee_delta(current: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Interpret ``delta`` as an EE-frame increment on ``current`` world pose.

    Position ``delta[:3]`` is a translation in the current tool frame.
    Rotation ``delta[3:6]`` is an intrinsic (right-compose) rotvec about EE axes.
    """
    current = np.asarray(current, dtype=float).reshape(6)
    delta = np.asarray(delta, dtype=float).reshape(6)
    pose = current.copy()
    pose[:3] = current[:3] + quat_rotate(rotvec_to_quat(current[3:6]), delta[:3])
    q_new = quat_mul(rotvec_to_quat(current[3:6]), rotvec_to_quat(delta[3:6]))
    pose[3:6] = quat_to_rotvec(q_new)
    return pose
