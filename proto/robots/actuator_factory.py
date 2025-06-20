from proto.common.configs import ActuatorConfig
from proto.robots.actuator import Actuator
from proto.robots.feetech_servo import FeetechServo
from proto.robots.dynamixel_servo import DynamixelServo
from enum import Enum

class ActuatorEnum(str, Enum):
    FEETECH = "feetech"
    DYNAMIXEL = "dynamixel"


actuators = {
    ActuatorEnum.FEETECH.value: FeetechServo,
    ActuatorEnum.DYNAMIXEL.value: DynamixelServo,
}

class ActuatorFactory():
    
    @staticmethod
    def create_actuator(actuator_cfg: ActuatorConfig, name: str) -> Actuator:
        if actuator_cfg.type not in actuators:
            raise ValueError(f"Actuator type {actuator_cfg.type} not implemented.")
        return actuators[actuator_cfg.type](actuator_cfg, name)
        