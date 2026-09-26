"""Cartesian-space IK actuator.

A :class:`CartesianIKActuator` represents one component of an end-effector pose
(x, y, z, rx, ry, rz). An arm that implements IK declares six of these (like the
UR5 does) alongside its joint-space actuators; the pose index each one drives is
taken from its ``cfg.id`` (0..5 == x, y, z, rx, ry, rz).

Cartesian IK actuators never talk to hardware directly. They are handled by the
:class:`~solarian.core.ik.bus.IKActuatorBus` that a module installs to wrap its
joint bus, which converts Cartesian reads/writes to FK/IK on the joint bus. The
module bus recognizes them *by class* (``isinstance(..., CartesianIKActuator)``)
and routes them through IK instead of building a hardware bus for them. The
``bus_class`` below is only a marker to satisfy :class:`Component`; it is never
instantiated for a Cartesian IK actuator.
"""

from __future__ import annotations

from solarian.core.servo import Servo
from solarian.core.data import PosData
from solarian.core.ik.bus import IKActuatorBus

#: Pose-axis index from the class attribute name when yaml omits ``id``.
CARTESIAN_AXIS_IDS = {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5}


class CartesianIKActuator(Servo):
    """End-effector pose component (metres for x/y/z, radians for rx/ry/rz).

    Values are raw metric units; they are not normalized like joint servos.
    Declare on the module with no yaml entry — ``id`` comes from the
    attribute name (x..rz), ``mode`` is ``raw``, and they share the hardware
    servos' port.
    """

    #: Placeholder only -- the real bus is the module's IKActuatorBus.
    bus_class = IKActuatorBus
    default_mode = "raw"
    uses_motor_calibration = False

    pos = PosData()

    @classmethod
    def unit_for(cls, component_name: str, data_name: str) -> str | None:
        if data_name != "pos":
            return super().unit_for(component_name, data_name)
        axis = CARTESIAN_AXIS_IDS.get(component_name)
        if axis is None:
            return None
        return "m" if axis < 3 else "rad"
