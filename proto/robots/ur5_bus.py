import math
from typing import Any, Dict

import urx
import rtde_control
import rtde_receive
from lerobot.common.motors import MotorCalibration

from proto.common.configs import ValueName
from proto.robots.actuator_bus import ActuatorBus
from proto.robots.component import Component


KP = 2.0                             # proportional gain [1/s]; tune 1.0–4.0
# MAX_JOINT_SPEED = 1.0                # rad/s cap (per joint)
MAX_JOINT_SPEED = 0.5                # rad/s cap (per joint)
# ACC = 2.0                            # acceleration cap for speedj
ACC = 1.0                            # acceleration cap for speedj
SAFE_MARGIN = math.radians(2.0)      # keep some distance from joint limits (in rad)
DT = 1.0 / 30                        # time step for P controller

STEP_LIMIT = math.radians(20) * DT    # max per-tick joint change (rad)
LOOKAHEAD = 0.1                       # s (0.03–0.2 typical)
GAIN = 300                            # unitless (start 100–600 on CB3)


# UR5 (CB3/e-Series) soft limits (approx; adjust to your exact model if needed)
UR5_LIMITS = [
    (-2*math.pi,  2*math.pi),        # base
    (-2*math.pi,  2*math.pi),        # shoulder
    (-2*math.pi,  2*math.pi),        # elbow
    (-2*math.pi,  2*math.pi),        # wrist1
    (-2*math.pi,  2*math.pi),        # wrist2
    (-2*math.pi,  2*math.pi),        # wrist3
]

# UR5_LIMITS = [
#     (-math.pi/2,  math.pi/2),        # base
#     (-math.pi/2,  math.pi/2),        # shoulder
#     (-math.pi/2,  math.pi/2),        # elbow
#     (-math.pi/2,  math.pi/2),        # wrist1
#     (-math.pi/2,  math.pi/2),        # wrist2
#     (-math.pi/2,  math.pi/2),        # wrist3
# ]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def clamp_to_limits(q):
    out = []
    for j, qj in enumerate(q):
        lo = UR5_LIMITS[j][0] + SAFE_MARGIN
        hi = UR5_LIMITS[j][1] - SAFE_MARGIN
        out.append(clamp(qj, lo, hi))
    return out


def clamp_joint_value(value: float, joint_id: int):
    lo = UR5_LIMITS[joint_id][0] + SAFE_MARGIN
    hi = UR5_LIMITS[joint_id][1] - SAFE_MARGIN
    return clamp(value, lo, hi)


class UR5Bus(ActuatorBus):
    def __init__(self, port: str, components: Dict[str, Component], calibration: Dict[str, MotorCalibration], joint_names2ids: Dict[str, int], cartesian_names2ids: Dict[str, int]):
        super().__init__(port, components, calibration)
        
        self.joint_names2ids = joint_names2ids
        self.cartesian_names2ids = cartesian_names2ids
        self.connected = False

    def is_connected(self) -> bool:
        """bool: `True` if the underlying serial port is open."""
        return self.connected

    def connect(self):
        self.robot = urx.URRobot(self.port)
        self.rtde_c = rtde_control.RTDEControlInterface(self.port)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.port)
        self.connected = True

    def disconnect(self):
        self.connected = False
        self.robot.close()

    def read(self, data_name: str, component_names: list[str], normalize: bool = True) -> Dict[str, Any]:

        values = {}
        # start with joint values
        names = [name for name in component_names if name in self.joint_names2ids.keys()]
        if names:
            # joint_values = self.robot.getj()
            joint_values = self.rtde_r.getActualQ()
            values.update({name: joint_values[self.joint_names2ids[name]] for name in names})

        # then with cartesian values
        names = [name for name in component_names if name in self.cartesian_names2ids.keys()]
        if names:
            # cartesian_values = self.robot.getl()
            cartesian_values = self.rtde_r.getActualTCPPose()
            values.update({name: cartesian_values[self.cartesian_names2ids[name]] for name in names})

        if normalize:
            values = self._normalize(values)

        return values


    def write(self, data_name: str, values: Dict[str, Any], normalize: bool = True):
        
        if normalize:
            values = self._unnormalize(values)
            
        # first joint values
        names = [name for name in values.keys() if name in self.joint_names2ids.keys()]
        if names:
            # q_now = self.robot.getj()
            q_now = self.rtde_r.getActualQ()
            
            
            # P controller: joint speed = Kp * error
            # err = [clamp_joint_value(values[name], j) - q_now[j] if name in values else 0 for name, j in self.joint_names2ids.items()]
            # qd_cmd = [clamp(KP * e, -MAX_JOINT_SPEED, MAX_JOINT_SPEED) for e in err]

            # dq = [clamp_joint_value(values[name], j) - q_now[j] if name in values else 0 for name, j in self.joint_names2ids.items()]
            # dq = [clamp(KP * d, -STEP_LIMIT, STEP_LIMIT) for d in dq]
            # q_next = [q_now[j] + dq[j] for j in range(6)]

            q_target = [clamp_joint_value(values[name], j) if name in values else q_now[j] for name, j in self.joint_names2ids.items()]
            
            try:
                # # Send velocity command for exactly dt seconds
                # self.robot.speedj(q_next, acc=ACC, min_time=DT)

                # Send a short "hold for DT" servoJ step
                # URScript: servoJ(q, a, v, t, lookahead_time, gain)
                # urx maps keyword args; a,v can be omitted for small steps
                # print(q_next)

                # self.rtde_c.servoJ(q_target,
                #                 speed=MAX_JOINT_SPEED,
                #                 acceleration=ACC,
                #                 time=DT,
                #                 lookahead_time=LOOKAHEAD,
                #                 gain=GAIN)

                self.rtde_c.servoJ(q_target,
                                MAX_JOINT_SPEED,
                                ACC,
                                DT,
                                LOOKAHEAD,
                                GAIN)
                
                
                # gives priority to joint commands, discards cartesian commands
                return
            except Exception as e:
                print(e)

        
        # then with cartesian values
        names = [name for name in values.keys() if name in self.cartesian_names2ids.keys()]
        if names:
            # cartesian_values = self.robot.getl()
            cartesian_values = self.rtde_r.getActualTCPPose()
            for name, value in values.items():
                if name in self.cartesian_names2ids:
                    cartesian_values[self.cartesian_names2ids[name]] = value
            try:
                # if normalize:
                #     cartesian_values = self._unnormalize(cartesian_values)
                # self.robot.movel(cartesian_values, acc=0.5, vel=0.5)
                self.rtde_c.moveL(cartesian_values, speed=0.5, acceleration=0.5)
            except Exception as e:
                print(e)



    def is_calibrated(self):
        return self.calibration != {}

    
    def reset_calibration(self, component_names: list[str]):
        self.calibration = {}

    def read_calibration(self) -> dict[str, MotorCalibration]:
        return self.calibration
        
    def write_calibration(self, calibration_dict: dict[str, MotorCalibration]) -> dict[str, MotorCalibration]:
        self.calibration = calibration_dict
        return calibration_dict

    def start_calibration_mode(self):
        # Enable freedrive (gravity compensation)
        self.robot.set_freedrive(True, 600)

    def end_calibration_mode(self):
        self.robot.set_freedrive(False)

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

    
        