"""
UR5 Actuators

Actuator classes for Universal Robots UR5 arms.
Supports both joint-space and Cartesian-space control.
"""

from typing import ClassVar, Optional
from solarian.core.servo import Servo


class UR5JointActuator(Servo):
    """
    Joint-space actuator for UR5 robot arms.
    
    Controls individual joints (shoulder_pan, shoulder_lift, elbow, wrist1-3).
    
    Retrieve this class from the plugin registry:
    
        from solarian.plugins import actuators
        UR5JointActuator = actuators.get("ur5_joint")
        
        class MyUR5Module(Module):
            shoulder_pan: UR5JointActuator = UR5JointActuator(id=0, model="ur5")
            shoulder_lift: UR5JointActuator = UR5JointActuator(id=1, model="ur5")
    """
    
    _bus_type: ClassVar[str] = "ur5"
    
    model: str = "ur5"
    port: Optional[str] = None  # IP address of the UR5
    mode: str = "range_m100_100"


class UR5CartesianActuator(Servo):
    """
    Cartesian-space actuator for UR5 robot arms.
    
    Controls end-effector position/orientation (x, y, z, rx, ry, rz).
    
    Retrieve this class from the plugin registry:
    
        from solarian.plugins import actuators
        UR5CartesianActuator = actuators.get("ur5_cartesian")
        
        class MyUR5Module(Module):
            x: UR5CartesianActuator = UR5CartesianActuator(id=0, model="ur5")
            y: UR5CartesianActuator = UR5CartesianActuator(id=1, model="ur5")
            z: UR5CartesianActuator = UR5CartesianActuator(id=2, model="ur5")
    """
    
    _bus_type: ClassVar[str] = "ur5"
    
    model: str = "ur5"
    port: Optional[str] = None  # IP address of the UR5
    mode: str = "raw"


# Backward compatibility alias
UR5Servo = UR5JointActuator
