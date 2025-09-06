# from ur_rtde import rtde_control, rtde_receive
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
    def __init__(self, port: str):
        # port is the IP of the robot
        self.port = port
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
        # q = self.rtde_r.getActualQ()
        # q = self.robot.getj()
        q = self.robot.getl()
        return [q[servo_names.index(name)] for name in component_names]

    def write(self, data_name: ValueName, component_names: list[str], values: list[float]):
        # q = self.rtde_r.getActualQ()
        # q = self.robot.getj()
        # for name, value in zip(component_names, values):
        #     q[servo_names.index(name)] = value
        # # self.rtde_c.moveJ(q, speed=1.0, acceleration=1.2)  # rad/s, rad/s^2
        # try:
        #     self.robot.movej(q, acc=0.5, vel=0.5)
        # except Exception as e:
        #     print(e)
        pose = self.robot.getl()
        print("pose", pose, "type", type(pose))
        

        for name, value in zip(component_names, values):
            pose[servo_names.index(name)] = value

        print("pose", pose, "type", type(pose))
            
        try:
            self.robot.movel(pose, acc=0.5, vel=0.5)
        except Exception as e:
            print(e)



    def is_calibrated(self):
        return True

    @property
    def is_connected(self) -> bool:
        """bool: `True` if the underlying serial port is open."""
        return self.connected