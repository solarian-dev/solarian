from abc import ABC, abstractmethod
from typing import Any

from proto.robots.actuator import Actuator
from proto.robots.sensor import Sensor
from proto.common.configs import ServoConfig


class Servo(Actuator, Sensor):
    def __init__(self, cfg: ServoConfig, name: str):
        super().__init__(cfg, name)

        