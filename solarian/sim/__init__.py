"""MuJoCo-backed simulation framework (part of Solarian core).

Provides the base classes for building simulated robots that behave identically
to real ones -- same observation/action schemas, transport and control loop --
by composing a shared MuJoCo world and proxying each simulated module's I/O to
it.

MuJoCo and other ``[sim]`` extras are imported lazily inside the sim code paths,
so importing :mod:`solarian` (or even :mod:`solarian.sim`) never requires them;
only instantiating a :class:`SimRobot` does.
"""

from solarian.sim.configs import SceneConfig, ModuleAttachment, CameraMount
from solarian.sim.module import SimModule
from solarian.sim.servo import SimServo
from solarian.sim.camera import SimCamera
from solarian.sim.camera_module import SimCameraModule
from solarian.sim.robot import SimRobot

__all__ = [
    "SimRobot",
    "SimModule",
    "SimServo",
    "SimCamera",
    "SimCameraModule",
    "SceneConfig",
    "ModuleAttachment",
    "CameraMount",
]
