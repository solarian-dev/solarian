"""
UR5 Arm Module

Module for Universal Robots UR5 arms with support for both joint-space
and Cartesian-space control.
"""

from solarian.core.module import Module
from solarian.core.configs import ModuleConfig
from solarian_ur5.actuator import UR5JointActuator, UR5CartesianActuator


class UR5ArmConfig(ModuleConfig):
    calibration_file: str | None = None


class UR5ArmModule(Module):
    """
    UR5 robot arm module.
    
    Supports both joint-space control (individual joint control) and
    Cartesian-space control (end-effector position/orientation).
    
    Key features:
        - 6 DOF arm with 6 joints
        - Joint-space control for individual joints
        - Cartesian-space control for end-effector (optional)
        - Uses UR5 actuators from the same package
    """

    config_class = UR5ArmConfig
    
    # Joint-space actuators
    shoulder_pan: UR5JointActuator = UR5JointActuator(id=0, model="ur5")
    shoulder_lift: UR5JointActuator = UR5JointActuator(id=1, model="ur5")
    elbow_flex: UR5JointActuator = UR5JointActuator(id=2, model="ur5")
    wrist_flex: UR5JointActuator = UR5JointActuator(id=3, model="ur5")
    wrist_yaw: UR5JointActuator = UR5JointActuator(id=4, model="ur5")
    wrist_roll: UR5JointActuator = UR5JointActuator(id=5, model="ur5")
    
    # Optional: Cartesian-space actuators (uncomment if needed)
    # x: UR5CartesianActuator = UR5CartesianActuator(id=0, model="ur5")
    # y: UR5CartesianActuator = UR5CartesianActuator(id=1, model="ur5")
    # z: UR5CartesianActuator = UR5CartesianActuator(id=2, model="ur5")
    # rx: UR5CartesianActuator = UR5CartesianActuator(id=3, model="ur5")
    # ry: UR5CartesianActuator = UR5CartesianActuator(id=4, model="ur5")
    # rz: UR5CartesianActuator = UR5CartesianActuator(id=5, model="ur5")

