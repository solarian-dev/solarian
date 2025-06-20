from typing import Dict
from .actuator import Actuator
from .sensor import Sensor
from .body import Body
from .actuator_factory import ActuatorFactory
from proto.common.configs import JointConfig


class Joint:    
    
    def __init__(self, name: str, cfg: JointConfig, parent_body: Body, child_body: Body):
        self.name = name
        self.cfg = cfg

        self.parent_body = parent_body
        self.child_body = child_body

        self.actuators = {}
        self.sensors = {}
        
        for name, actuator_cfg in cfg.actuators.items():
            self.actuators[name] = ActuatorFactory.create_actuator(actuator_cfg, name)

        # if the actuator is also a sensor, eg: a servo that has an encoder, add it to sensors
        for name, actuator in self.actuators.items():
            if isinstance(actuator, Sensor):
                self.sensors[name] = actuator
        
    def get_actuators(self) -> Dict[str, Actuator]:
        return self.actuators
        
    def get_sensors(self) -> Dict[str, Sensor]:
        return self.sensors
        
    def connect_actuators(self):
        for actuator in self.actuators.values():
            actuator.connect()
        