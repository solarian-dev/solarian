from proto.common.configs import ActuatorConfig
from proto.robots.actuator import Actuator
from proto.robots.feetech_servo import FeetechServo
from proto.robots.dynamixel_servo import DynamixelServo
from proto.robots.ur5_servo import UR5Servo
from enum import Enum

class ActuatorEnum(str, Enum):
    FEETECH = "feetech"
    DYNAMIXEL = "dynamixel"
    UR5_JOINT = "ur5_joint"
    UR5_CARTESIAN = "ur5_cartesian"


actuators = {
    ActuatorEnum.FEETECH.value: FeetechServo,
    ActuatorEnum.DYNAMIXEL.value: DynamixelServo,
    ActuatorEnum.UR5_JOINT.value: UR5Servo,
    ActuatorEnum.UR5_CARTESIAN.value: UR5Servo,
}

class ActuatorFactory():
    
    @staticmethod
    def create_actuator(actuator_cfg: ActuatorConfig, name: str) -> Actuator:
        if actuator_cfg.type not in actuators:
            raise ValueError(f"Actuator type {actuator_cfg.type} not implemented.")
        return actuators[actuator_cfg.type](actuator_cfg, name)
        