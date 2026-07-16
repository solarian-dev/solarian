"""Base class for simulated modules.

A :class:`SimModule` is an ordinary :class:`~solarian.core.module.Module` whose
components are simulated (``SimServo`` / ``SimCamera``). It carries no extra
runtime behaviour -- control logic is inherited from the concrete module it
mirrors -- but acts as a marker that :class:`~solarian.sim.robot.SimRobot` uses
to enforce that a simulated robot only contains simulated modules.
"""

from __future__ import annotations

from solarian.core.module import Module


class SimModule(Module):
    """Marker base for MuJoCo-backed modules.

    Concrete simulated modules typically subclass both the real module (to
    inherit its ``control_step`` / action handlers) and this class, e.g.::

        class SimArmModule(ArmModule, SimModule):
            shoulder_pan = SimServo()
            ...
    """

    #: Identifies modules that run against a simulated MuJoCo world.
    is_sim = True
