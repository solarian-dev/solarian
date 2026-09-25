import abc
from typing import Any, Dict

from solarian.core.component_bus import ComponentBus
from solarian.core.component import Component


class SensorBus(ComponentBus):
    """
    Abstract base class for sensor bus implementations.

    Sensor buses handle communication with sensor hardware
    (cameras, force sensors, etc.)

    To create a custom sensor bus:

    1. Subclass SensorBus and implement all abstract methods
    2. In your Sensor subclass, set bus_class to point to it:

        from my_package.bus import MySensorBus

        class MySensor(Sensor):
            bus_class = MySensorBus

    Sensors that produce frames want :class:`~solarian.core.camera_bus.CameraBus`
    instead, which adds the archive tap: the episode's video is then taken from
    the sensor's own output rather than from whatever survives the trip to a
    controller.
    """
    pass
