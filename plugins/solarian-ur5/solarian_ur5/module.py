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
    shoulder_pan: UR5JointActuator
    shoulder_lift: UR5JointActuator
    elbow_flex: UR5JointActuator
    wrist_flex: UR5JointActuator
    wrist_yaw: UR5JointActuator
    wrist_roll: UR5JointActuator
    
    # Optional: Cartesian-space actuators (uncomment if needed)
    x: UR5CartesianActuator
    y: UR5CartesianActuator
    z: UR5CartesianActuator
    rx: UR5CartesianActuator
    ry: UR5CartesianActuator
    rz: UR5CartesianActuator

