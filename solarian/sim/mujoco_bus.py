"""Placeholder :class:`ComponentBus` for simulated components.

Simulated components (``SimServo``, ``SimCamera``) run under a
:class:`~solarian.sim.robot.SimRobot`, which routes their shared MuJoCo port
through a :class:`~solarian.core.bus_server.BusClient` proxy to a single
:class:`~solarian.sim.mujoco_world.MujocoWorldServer` that owns the composed
scene. In that flow this bus class is never instantiated (the port is
overridden). It exists so that ``Component.bus_class`` is satisfied and to give
a clear error if a simulated module is connected outside a ``SimRobot``.
"""

from __future__ import annotations

from typing import Any, Dict

from solarian.core.component_bus import ComponentBus
from solarian.core.data import Data


_MESSAGE = (
    "MujocoBus is a virtual bus: simulated modules must run under a SimRobot, "
    "which owns the MuJoCo world and proxies component I/O to it. Instantiate "
    "your robot via `solarian run-robot` with a SimRobot subclass instead of "
    "connecting the module directly."
)


class MujocoBus(ComponentBus):
    """Virtual bus marker for simulated components (never opened directly)."""

    @property
    def is_connected(self) -> bool:
        return False

    def connect(self):
        raise NotImplementedError(_MESSAGE)

    def disconnect(self):
        pass

    def read(self, data_type: type[Data], component_names: list[str], normalize: bool = True) -> Dict[str, Any]:
        raise NotImplementedError(_MESSAGE)

    def write(self, values: Dict[str, Data], normalize: bool = True):
        raise NotImplementedError(_MESSAGE)
