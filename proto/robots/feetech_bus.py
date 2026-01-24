from typing import Dict, Any
from lerobot.common.motors.feetech import (
    FeetechMotorsBus,
)
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode

from proto.robots.component import Component
from proto.common.configs import ValueName
from proto.robots.actuator_bus import ActuatorBus


WRITE_MAP = {
    ValueName.POS: "Goal_Position",
    ValueName.SPEED: "Goal_Velocity",
    ValueName.TORQUE_ENABLE: "Torque_Enable",
}

READ_MAP = {
    ValueName.POS: "Present_Position",
    ValueName.SPEED: "Present_Velocity",
}

def map_write(data_name: str):
    if data_name in WRITE_MAP:
        return WRITE_MAP[data_name]
    return data_name

def map_read(data_name: str):
    if data_name in READ_MAP:
        return READ_MAP[data_name]
    return data_name

def map_motor_mode(mode:str):
    if mode == "range_0_100":
        return MotorNormMode.RANGE_0_100
    elif mode == "range_m100_100":
        return MotorNormMode.RANGE_M100_100
    elif mode == "degrees":
        return MotorNormMode.DEGREES
    else:
        raise ValueError(f"Unknown motor mode: {mode}")


class FeetechBus(ActuatorBus):

    def __init__(self, port: str, components: Dict[str, Component], calibration: Dict[str, MotorCalibration]):

        super().__init__(port, components, calibration)
        
        self.bus = FeetechMotorsBus(
            port=port,
            motors={
                name: Motor(component.cfg.id, component.cfg.model, map_motor_mode(component.cfg.mode)) for name, component in components.items()
            },
            calibration=calibration
        )

    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self):
        self.bus.connect()
        
    def disconnect(self):
        self.bus.disconnect()

    def read(self, data_name: str, component_names: list[str], normalize: bool = True):
        return self.bus.sync_read(map_read(data_name), motors=component_names, normalize=normalize, num_retry=10)

    def write(self, data_name: str, values: Dict[str, Any], normalize: bool = True):
        self.bus.sync_write(map_write(data_name), values=values, normalize=normalize)

    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def reset_calibration(self, component_names: list[str]):
        self.bus.reset_calibration(component_names)

    def read_calibration(self) -> dict[str, MotorCalibration]:
        return self.bus.read_calibration()
        
    def write_calibration(self, calibration_dict: dict[str, MotorCalibration]) -> dict[str, MotorCalibration]:
        # bound min and max to range of motion
        for name, calibration in calibration_dict.items():
            calibration.range_min = max(calibration.range_min, 0)
            calibration.range_max = min(calibration.range_max, 4095)
        self.bus.write_calibration(calibration_dict)
        return calibration_dict

    def start_calibration_mode(self):
        """Sets the actuators where the arm can be moved manually freely"""
        # self.bus.disable_torque(list(self.components.keys()))
        # self.bus.set_position_mode(list(self.components.keys()))
        self.bus.disable_torque(list(self.components.keys()))
        self.write("Operating_Mode", {name: 0 for name in list(self.components.keys())})

    def end_calibration_mode(self):
        pass
        
    def set_half_turn_homings(self, component_names: list[str]) -> Dict[str, int]:
        """Set half turn homings for the actuators. Read present position and set/store"""
        return self.bus.set_half_turn_homings(component_names)

    def setup_component(self, component_name: str, configs: Dict[str, Any]):
        """Setup config for a component"""

        print("Setting the following configs for component:", component_name)
        print(configs)
        input("Connect the servo to the controller, make sure only the servo to be configured is connected. Press Enter when ready.")
        self.bus.setup_motor(component_name, initial_baudrate=configs["baudrate"], initial_id=configs["id"])
        print("Configuration completed.")