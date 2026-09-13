"""Backend-agnostic inverse-kinematics infrastructure (Solarian core).

Provides:

* :class:`DifferentialIKBackend` -- the interface a concrete IK solver implements.
* :class:`IKActuatorBus` -- a composite bus that layers Cartesian (IK/FK) control
  on top of a module's joint bus.
* :class:`CartesianIKActuator` -- the actuator type an arm declares for each
  end-effector pose component.
* :class:`FrameOffset` -- rigid tool pose (pos + quat) in a parent frame.
* :class:`MinkIK` -- a declarative spec for the built-in mink/MuJoCo backend.
* :class:`MinkBackend` -- the mink/MuJoCo backend itself (imports ``mink``/
  ``mujoco`` lazily; requires the ``[sim]`` extra).

An arm gets IK by declaring joint actuators plus six ``CartesianIKActuator`` s and
setting a class-level ``ik_backend`` (e.g. ``MinkIK(...)``); the core builds the
backend automatically when connecting.
"""

from solarian.core.ik.base import DifferentialIKBackend
from solarian.core.ik.actuator import CartesianIKActuator
from solarian.core.ik.bus import IKActuatorBus
from solarian.core.ik.spec import FrameOffset, MinkIK


def __getattr__(name):
    # Lazily expose MinkBackend without importing mink/mujoco at package import.
    if name == "MinkBackend":
        from solarian.core.ik.mink_backend import MinkBackend

        return MinkBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DifferentialIKBackend",
    "IKActuatorBus",
    "CartesianIKActuator",
    "FrameOffset",
    "MinkIK",
    "MinkBackend",
]
