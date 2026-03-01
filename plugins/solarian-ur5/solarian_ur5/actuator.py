"""
UR5 Actuators

Actuator classes for Universal Robots UR5 arms.
Supports both joint-space and Cartesian-space control.
"""

from typing import Optional
from solarian.core.servo import Servo
from solarian.core.data import PosData, SpeedData
from solarian_ur5.bus import UR5Bus



class UR5JointActuator(Servo):
    """
    Joint-space actuator for UR5 robot arms.
    
    Controls individual joints (shoulder_pan, shoulder_lift, elbow, wrist1-3).
    
    Usage:
    
        from solarian_ur5.actuator import UR5JointActuator
        
        class MyUR5Module(Module):
            shoulder_pan: UR5JointActuator = UR5JointActuator(id=0, model="ur5")
            shoulder_lift: UR5JointActuator = UR5JointActuator(id=1, model="ur5")
    """
    
    bus_class = UR5Bus

    pos = PosData()
    speed = SpeedData()

class UR5CartesianActuator(Servo):
    """
    Cartesian-space actuator for UR5 robot arms.
    
    Controls end-effector position/orientation (x, y, z, rx, ry, rz).
    
    Usage:
    
        from solarian_ur5.actuator import UR5CartesianActuator
        
        class MyUR5Module(Module):
            x: UR5CartesianActuator = UR5CartesianActuator(id=0, model="ur5")
            y: UR5CartesianActuator = UR5CartesianActuator(id=1, model="ur5")
            z: UR5CartesianActuator = UR5CartesianActuator(id=2, model="ur5")
    """
    
    bus_class = UR5Bus

    x = PosData()
    y = PosData()
    z = PosData()
    rx = PosData()
    ry = PosData()
    rz = PosData()
    

# Backward compatibility alias
UR5Servo = UR5JointActuator
