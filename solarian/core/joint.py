from typing import Tuple
from .body import BodyName
from enum import Enum


class JointName(Enum):
    
    def __str__(self):
        return self.value

class JointType(Enum):
    HINGE = "hinge"
    SLIDE = "slide"
    BALL = "ball"
    FIXED = "fixed"
    FREE = "free"
    
    def __str__(self):
        return self.value


class Joint:    
    
    # def __init__(self, name: str, cfg: JointConfig, parent_body: Body, child_body: Body):
    def __init__(self, name: JointName, parent_body: BodyName, child_body: BodyName, pos: Tuple[float, float, float] = (0, 0, 0), axis: Tuple[float, float, float] = (0, 0, 0), type: JointType = JointType.HINGE, range: Tuple[float, float] = (0, 0)):
        # self.cfg = cfg
        self.name = name
        self.parent_body = parent_body
        self.child_body = child_body
        self.axis = axis
        self.type = type
        self.range = range
        self.pos = pos
        
        # self.actuators = {}
        # self.sensors = {}
        
        # for name, actuator_cfg in cfg.actuators.items():
        #     self.actuators[name] = ActuatorFactory.create_actuator(actuator_cfg, name)

        # # if the actuator is also a sensor, eg: a servo that has an encoder, add it to sensors
        # for name, actuator in self.actuators.items():
        #     if isinstance(actuator, Sensor):
        #         self.sensors[name] = actuator
        
    # def get_actuators(self) -> Dict[str, Actuator]:
    #     return self.actuators
        
    # def get_sensors(self) -> Dict[str, Sensor]:
    #     return self.sensors
        
    # def connect_actuators(self):
    #     for actuator in self.actuators.values():
    #         actuator.connect()

    # def add_actuator(self, actuator: Actuator):
    #     self.actuators[actuator.name] = actuator
    #     if isinstance(actuator, Sensor):
    #         self.sensors[actuator.name] = actuator

    # def add_sensor(self, sensor: Sensor):
    #     self.sensors[sensor.name] = sensor
    #     if isinstance(sensor, Actuator):
    #         self.actuators[sensor.name] = sensor

    # def add_servo(self, servo: Servo):
    #     self.add_actuator(servo)
    #     self.add_sensor(servo)