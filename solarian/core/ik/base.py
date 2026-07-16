"""Differential inverse-kinematics backend interface.

Kinematics are pure functions of a model that is shared by simulation and real
hardware, so a single IK backend can drive both. This module declares only the
interface; concrete backends (e.g. a mink/MuJoCo :class:`MinkBackend`, a
``PinocchioPinkBackend``, or an ``AnalyticBackend`` for simple arms) implement
it and are wired in at the module level via an
:class:`~solarian.core.ik.bus.IKActuatorBus`.

Poses are exchanged as a 6-vector ``(x, y, z, rx, ry, rz)`` where ``x, y, z`` is
the end-effector position in metres and ``(rx, ry, rz)`` is an axis-angle
(rotation-vector) orientation in radians -- the same convention the UR5 bus uses
for its Cartesian actuators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class DifferentialIKBackend(ABC):
    """Interface for velocity-based (differential) inverse kinematics.

    Typical usage each control tick::

        backend.set_q(current_joint_positions)     # radians
        target = backend.get_pose()                # FK of current config
        target[:3] = commanded_xyz                 # overlay the command
        v = backend.solve(target, dt)
        q_next = backend.integrate(v, dt)          # radians
    """

    @abstractmethod
    def set_q(self, q: Any) -> None:
        """Set the current joint configuration (radians) the solver reasons about.

        ``q`` is ordered to match :attr:`joint_actuator_names`.
        """
        raise NotImplementedError

    @abstractmethod
    def solve(self, target_pose: Any, dt: float) -> Any:
        """Return joint velocities driving the end-effector toward ``target_pose``.

        ``target_pose`` is a 6-vector ``(x, y, z, rx, ry, rz)``.
        """
        raise NotImplementedError

    @abstractmethod
    def integrate(self, v: Any, dt: float) -> Any:
        """Integrate joint velocities ``v`` over ``dt`` into a new configuration.

        Returns the next joint positions (radians), ordered to match
        :attr:`joint_actuator_names`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_pose(self) -> np.ndarray:
        """Return the current end-effector pose ``(x, y, z, rx, ry, rz)``.

        Forward kinematics of the controlled frame at the configuration last set
        with :meth:`set_q`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_actuator_names(self) -> list[str]:
        """Ordered names of the joint actuators the backend controls.

        Defines the ordering of the vectors passed to :meth:`set_q` and returned
        by :meth:`integrate`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """Per-joint ``(min, max)`` position limits in radians.

        Keyed by joint actuator name. Used by the
        :class:`~solarian.core.ik.bus.IKActuatorBus` to convert between the
        normalized ``[-100, 100]`` joint space and radians.
        """
        raise NotImplementedError
