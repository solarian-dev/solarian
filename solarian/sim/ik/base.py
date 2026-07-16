"""Differential inverse-kinematics backend interface.

Kinematics are pure functions of a model that is shared by simulation and real
hardware, so a single IK backend can drive both. This module declares only the
interface; concrete backends (e.g. a mink/MuJoCo ``MinkBackend``, a
``PinocchioPinkBackend``, or an ``AnalyticBackend`` for simple arms) implement
it and are wired in at the module level later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DifferentialIKBackend(ABC):
    """Interface for velocity-based (differential) inverse kinematics.

    Typical usage each control tick::

        backend.set_q(current_joint_positions)
        v = backend.solve(target_pose, dt)
        q_next = backend.integrate(v, dt)
    """

    @abstractmethod
    def set_q(self, q: Any) -> None:
        """Set the current joint configuration the solver reasons about."""
        raise NotImplementedError

    @abstractmethod
    def solve(self, target_pose: Any, dt: float) -> Any:
        """Return joint velocities driving the end-effector toward ``target_pose``."""
        raise NotImplementedError

    @abstractmethod
    def integrate(self, v: Any, dt: float) -> Any:
        """Integrate joint velocities ``v`` over ``dt`` into a new configuration."""
        raise NotImplementedError
