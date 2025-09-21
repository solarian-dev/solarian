# from ur_rtde import rtde_control, rtde_receive
from typing import Any, Dict

import urx
from lerobot.common.motors import MotorCalibration

from proto.common.configs import ValueName
from proto.robots.actuator_bus import ActuatorBus
from proto.robots.component import Component


servo_names = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll"
]


class UR5Bus(ActuatorBus):
    def __init__(self, port: str, components: Dict[str, Component], calibration: Dict[str, MotorCalibration], joint_names2ids: Dict[str, int], cartesian_names2ids: Dict[str, int]):
        super().__init__(port, components, calibration)
        
        self.joint_names2ids = joint_names2ids
        self.cartesian_names2ids = cartesian_names2ids
        self.rtde_c = None
        self.rtde_r = None
        self.connected = False

    def is_connected(self) -> bool:
        """bool: `True` if the underlying serial port is open."""
        return self.connected

    def connect(self):
        self.robot = urx.URRobot(self.port)
        self.connected = True

    def disconnect(self):
        self.connected = False
        self.robot.close()

    def read(self, data_name: str, component_names: list[str], normalize: bool = True) -> Dict[str, Any]:

        values = {}
        # start with joint values
        names = [name for name in component_names if name in self.joint_names2ids.keys()]
        if names:
            joint_values = self.robot.getj()
            values.update({name: joint_values[self.joint_names2ids[name]] for name in names})

        # then with cartesian values
        names = [name for name in component_names if name in self.cartesian_names2ids.keys()]
        if names:
            cartesian_values = self.robot.getl()
            values.update({name: cartesian_values[self.cartesian_names2ids[name]] for name in names})

        if normalize:
            values = self._normalize(values)

        return values


    def write(self, data_name: str, values: Dict[str, Any], normalize: bool = True):
        
        # first joint values
        names = [name for name in values.keys() if name in self.joint_names2ids.keys()]
        if names:
            joint_values = self.robot.getj()
            for name, value in values.items():
                if name in self.joint_names2ids:
                    joint_values[self.joint_names2ids[name]] = value
            try:
                if normalize:
                    joint_values = self._unnormalize(joint_values)
                self.robot.movej(joint_values, acc=0.5, vel=0.5)
            except Exception as e:
                print(e)

        
        # then with cartesian values
        names = [name for name in values.keys() if name in self.cartesian_names2ids.keys()]
        if names:
            cartesian_values = self.robot.getl()
            for name, value in values.items():
                if name in self.cartesian_names2ids:
                    cartesian_values[self.cartesian_names2ids[name]] = value
            try:
                if normalize:
                    cartesian_values = self._unnormalize(cartesian_values)
                self.robot.movel(cartesian_values, acc=0.5, vel=0.5)
            except Exception as e:
                print(e)



    def is_calibrated(self):
        return self.calibration != {}

    
    def reset_calibration(self, component_names: list[str]):
        self.calibration = {}

    def read_calibration(self) -> dict[str, MotorCalibration]:
        return self.calibration
        
    def write_calibration(self, calibration_dict: dict[str, MotorCalibration]) -> None:
        self.calibration = calibration_dict

    def start_calibration_mode(self):
        # Enable freedrive (gravity compensation)
        self.robot.freedrive_mode()

    def end_calibration_mode(self):
        self.robot.end_freedrive_mode()

    def set_half_turn_homings(self, actuator_names: list[str]):
        self.reset_calibration(actuator_names)
        half_turn_homings = {}
        actual_positions = self.read("Present_Position", actuator_names, normalize=False)
        for name, pos in actual_positions.items():
            if name not in self.calibration:
                self.calibration[name] = {}
            self.calibration[name]["homing_offset"] = -pos
            half_turn_homings[name] = -pos
        return half_turn_homings

    
        