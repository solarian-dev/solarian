# from ur_rtde import rtde_control, rtde_receive
from typing import Any, Dict

import urx
from proto.common.configs import ValueName


servo_names = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll"
]


class UR5Bus():
    def __init__(self, port: str, joint_names2ids: Dict[str, int], cartesian_names2ids: Dict[str, int]):
        # port is the IP of the robot
        self.port = port
        self.joint_names2ids = joint_names2ids
        self.cartesian_names2ids = cartesian_names2ids
        self.rtde_c = None
        self.rtde_r = None
        self.connected = False

    def connect(self):
        # self.rtde_c = rtde_control.RTDEControlInterface(self.port)
        # self.rtde_r = rtde_receive.RTDEReceiveInterface(self.port)
        # self.robot = urx.Robot(self.port)
        self.robot = urx.URRobot(self.port)
        self.connected = True

    def disconnect(self):
        self.connected = False
        # self.rtde_c.stopJ(1.0)  # stop gracefully
        # self.rtde_c.disconnect()
        self.robot.close()

    def read(self, data_name: ValueName, component_names: list[str]):

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


        # return values in the same order as component_names
        return [values[name] for name in component_names]


    def write(self, data_name: ValueName, component_names: list[str], values: list[float]):
        
        # first joint values
        names = [name for name in component_names if name in self.joint_names2ids.keys()]
        if names:
            joint_values = self.robot.getj()
            for name, value in zip(component_names, values):
                if name in self.joint_names2ids:
                    joint_values[self.joint_names2ids[name]] = value
            try:
                self.robot.movej(joint_values, acc=0.5, vel=0.5)
            except Exception as e:
                print(e)

        
        # then with cartesian values
        names = [name for name in component_names if name in self.cartesian_names2ids.keys()]
        if names:
            cartesian_values = self.robot.getl()
            for name, value in zip(component_names, values):
                if name in self.cartesian_names2ids:
                    cartesian_values[self.cartesian_names2ids[name]] = value
            try:
                self.robot.movel(cartesian_values, acc=0.5, vel=0.5)
            except Exception as e:
                print(e)



    def is_calibrated(self):
        return True

    @property
    def is_connected(self) -> bool:
        """bool: `True` if the underlying serial port is open."""
        return self.connected