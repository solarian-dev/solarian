from typing import Any, Dict

from proto.common.configs import CameraConfig
from proto.robots.sensor import Sensor


class Camera(Sensor):
    def __init__(self, cfg: CameraConfig, name: str):
        super().__init__(cfg, name)


    # @abstractmethod
    # def connect(self):
    #     """Start all threads"""
    #     self.running = True
    
    # @abstractmethod
    # def disconnect(self):
    #     """Clean shutdown"""
    #     self.running = False

    # @abstractmethod
    # def read(self, value_name: str):
    #     """Read sensor data"""
    #     pass



        

