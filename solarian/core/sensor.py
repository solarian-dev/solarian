from abc import ABC, abstractmethod
from typing import Any
from solarian.core.configs import SensorConfig, ValueName
from solarian.core.component import Component
# from solarian.core.joint import Joint
# from solarian.core.body import Body


class Sensor(Component):
    pass
    # def __init__(self, cfg: SensorConfig, name: str):
    # def __init__(self, id: int, model: str):
    #     super().__init__(id, model)

    # @abstractmethod
    # def read(self, data_name: ValueName) -> Any:
    #     """Read a value from the component."""
    #     pass


