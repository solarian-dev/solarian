"""MuJoCo-backed simulation framework (part of Solarian core).

Any robot runs in simulation by flipping a switch: declare ``is_sim = True`` and
a ``scene`` (:class:`SceneConfig`) on the ordinary
:class:`~solarian.core.robot.Robot`. The robot keeps its real modules and
servos; the framework composes a shared MuJoCo world from the modules' MJCF
models and proxies each module's I/O to it, so a simulated robot behaves
identically to a real one -- same observation/action schemas, transport and
control loop.

MuJoCo and other ``[sim]`` extras are imported lazily inside the sim code paths,
so importing :mod:`solarian` (or even :mod:`solarian.sim`) never requires them;
only running a robot with ``is_sim = True`` does.
"""

from solarian.sim.configs import SceneConfig, ModuleAttachment, CameraMount

__all__ = [
    "SceneConfig",
    "ModuleAttachment",
    "CameraMount",
]
