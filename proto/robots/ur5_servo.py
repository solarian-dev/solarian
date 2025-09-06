from proto.robots.servo import Servo
from proto.common.configs import ServoConfig
from typing import Any


class UR5Servo(Servo):
    def __init__(self, cfg: ServoConfig, name: str):
        super().__init__(cfg, name)

    def connect(self):
        pass

    def disconnect(self):
        pass

    def read(self, data_name: str):
        pass

    def write(self, data_name: str, value: Any):
        pass