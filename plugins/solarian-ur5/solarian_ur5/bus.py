"""
UR5 Actuator Bus

Bus implementation for communicating with Universal Robots UR5 arms.
Supports both joint-space and Cartesian-space control.
"""

import math
import platform
import sys
from typing import Any, Dict
from solarian.core.data import Data, PosData
try:
    import rtde_control
    import rtde_receive
except ImportError as e:
    if sys.platform == "darwin" and platform.machine() == "arm64":
        raise ImportError(
            "solarian-ur5 requires ur-rtde, which has no pre-built wheel on macOS (Darwin) arm64. "
            "Install it from source: https://sdurobotics.gitlab.io/ur_rtde/installation/installation.html — "
            "clone the repo from https://gitlab.com/sdurobotics/ur_rtde, install Boost (e.g. brew install boost), "
            "then build and install with cmake/make."
        ) from e
    raise

from solarian.core.actuator_bus import ActuatorBus
from solarian.core.actuator import Actuator


KP = 2.0                             # proportional gain [1/s]; tune 1.0–4.0
MAX_JOINT_SPEED = 0.5                # rad/s cap (per joint)
ACC = 1.0                            # acceleration cap for speedj
SAFE_MARGIN = math.radians(2.0)      # keep some distance from joint limits (in rad)
DT = 1.0 / 30                        # time step for P controller
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
    """
    Actuator bus for Universal Robots UR5 arms.
    
    Supports both joint-space and Cartesian-space control.
    Automatically used when UR5JointActuator or UR5CartesianActuator are defined in a module.
    """
    
    def __init__(self, port: str, components: Dict[str, Actuator], calibration: Dict[str, Any] | None = None):
        super().__init__(port, components, calibration)

        from solarian_ur5.actuator import UR5JointActuator, UR5CartesianActuator

        self.joint_names2ids: Dict[str, int] = {}
        self.cartesian_names2ids: Dict[str, int] = {}

        for name, component in components.items():
            if component.cfg is None:
                raise ValueError(f"Component {name} has no configuration")
            if isinstance(component, UR5JointActuator):
                self.joint_names2ids[name] = component.cfg.id
            elif isinstance(component, UR5CartesianActuator):
                self.cartesian_names2ids[name] = component.cfg.id
            else:
                class_name = component.__class__.__name__
                if class_name == 'UR5JointActuator':
                    self.joint_names2ids[name] = component.cfg.id
                elif class_name == 'UR5CartesianActuator':
                    self.cartesian_names2ids[name] = component.cfg.id
                elif hasattr(component, 'cfg') and hasattr(component.cfg, 'id'):
                    self.joint_names2ids[name] = component.cfg.id

        self.connected = False

    @property
    def is_connected(self) -> bool:
        """bool: `True` if the underlying serial port is open."""
        return self.connected

    def connect(self):

        print(f"Connecting rtde control interface to {self.port}") 
        self.rtde_c = rtde_control.RTDEControlInterface(self.port)
        print(f"Connecting rtde receive interface to {self.port}") 
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.port)
        self.connected = True

    def disconnect(self):
        self.connected = False
        self.rtde_c.close()
        self.rtde_r.close()

    def read(self, data_type, component_names: list[str], normalize: bool = True) -> Dict[str, Any]:
        values = {}
        # start with joint values
        names = [name for name in component_names if name in self.joint_names2ids.keys()]
        if names:
            joint_values = self.rtde_r.getActualQ()
            values.update({name: joint_values[self.joint_names2ids[name]] for name in names})

        # then with cartesian values
        names = [name for name in component_names if name in self.cartesian_names2ids.keys()]
        if names:
            cartesian_values = self.rtde_r.getActualTCPPose()
            values.update({name: cartesian_values[self.cartesian_names2ids[name]] for name in names})

        if normalize:
            values = self._normalize(values)

        return values

    def write(self, values: Dict[str, Data], normalize: bool = True):
        raw_values: dict[str, int] = {}
        for name, value in values.items():
            if not isinstance(value, PosData):
                raise ValueError(f"Value {value} must be a PosData, got {type(value)}")
            raw_values[name] = value.value
        if normalize:
            raw_values = self._unnormalize(raw_values)
            
        # first joint values
        names = [name for name in raw_values.keys() if name in self.joint_names2ids.keys()]
        if names:
            q_now = self.rtde_r.getActualQ()
            q_target = [clamp_joint_value(raw_values[name], j) if name in raw_values else q_now[j] for name, j in self.joint_names2ids.items()]
            
            try:
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
        names = [name for name in raw_values.keys() if name in self.cartesian_names2ids.keys()]
        if names:
            cartesian_values = self.rtde_r.getActualTCPPose()
            for name, value in raw_values.items():
                if name in self.cartesian_names2ids:
                    cartesian_values[self.cartesian_names2ids[name]] = value
            try:
                self.rtde_c.moveL(cartesian_values, speed=0.5, acceleration=0.5)
            except Exception as e:
                print(e)

    def is_calibrated(self):
        return self.calibration != {}

    def reset_calibration(self, component_names: list[str]):
        self.calibration = {}

    def read_calibration(self) -> dict[str, Any]:
        return self.calibration
        
    def write_calibration(self, calibration_dict: dict[str, Any]) -> dict[str, Any]:
        self.calibration = calibration_dict
        return calibration_dict

    def start_calibration_mode(self):
        # Enable freedrive (gravity compensation)
        self.rtde_c.freedriveMode()

    def end_calibration_mode(self):
        self.rtde_c.endFreedriveMode()


    def set_half_turn_homings(self, actuator_names: list[str]) -> Dict[str, int]:
        self.reset_calibration(actuator_names)
        half_turn_homings = {}
        actual_positions = self.read(PosData, actuator_names, normalize=False)
        for name, pos in actual_positions.items():
            if name not in self.calibration:
                self.calibration[name] = {}
            self.calibration[name]["homing_offset"] = -pos
            half_turn_homings[name] = -pos
        return half_turn_homings
