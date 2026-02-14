from abc import ABC, abstractmethod
from typing import Any

from solarian.core.actuator import Actuator
from solarian.core.sensor import Sensor
# from solarian.core.configs import ServoConfig
# from solarian.core.joint import Joint


class Servo(Actuator, Sensor):
    pass
    # def __init__(self, cfg: ServoConfig, name: str):
    # def __init__(self, id: int, model: str):
    #     super().__init__(id, model)

        