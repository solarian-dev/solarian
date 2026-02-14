"""
Solarian-UR5 Plugin

Universal Robots UR5 actuator, bus, and arm module implementation for Solarian robotics framework.
"""

__version__ = "0.1.0"

# Export actuators
from solarian_ur5.actuator import UR5JointActuator, UR5CartesianActuator, UR5Servo

# Export bus
from solarian_ur5.bus import UR5Bus

# Export module
from solarian_ur5.module import UR5ArmModule

__all__ = [
    'UR5JointActuator',
    'UR5CartesianActuator',
    'UR5Servo',  # Backward compatibility alias
    'UR5Bus',
    'UR5ArmModule',
]
