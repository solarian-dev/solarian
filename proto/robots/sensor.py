from abc import ABC, abstractmethod
from typing import Any
from proto.common.configs import SensorConfig, ValueName
from proto.robots.component import Component


class Sensor(Component):
    def __init__(self, cfg: SensorConfig, name: str):
        super().__init__(cfg, name)

    @abstractmethod
    def read(self, data_name: ValueName) -> Any:
        """Read a value from the component."""
        pass


