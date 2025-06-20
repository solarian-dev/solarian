from abc import ABC, abstractmethod
from typing import Any
from proto.common.configs import ActuatorConfig
from proto.robots.component import Component


class Actuator(Component):
    def __init__(self, cfg: ActuatorConfig, name: str):
        self.cfg = cfg
        self.name = name

    @abstractmethod
    def write(self, value_name: str, value: Any):
        """Write actuator data"""
        
