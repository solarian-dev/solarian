"""
Koch Leader Arm Module

Leader arm module for teleoperation setups. The leader arm is moved by a human
operator, and its joint positions are read and sent to a follower arm.
"""

from solarian.core.module import Module
from solarian.conf import settings


class KochLeaderArmModule(Module):
    """
    Koch robot arm module configured as a leader (human-controlled arm).
    
    In teleoperation setups, the leader arm is moved by a human operator,
    and its joint positions are read and sent to a follower arm.
    
    Key features:
        - Uses Dynamixel XL330-M077 servos
        - Torque disabled for manual control
        - Position readings are the primary output
    """
    
    def initialize(self):
        """Initialize the arm with torque disabled for manual control."""
        super().initialize()

        # Disable torque for all actuators so operator can move the arm freely
        self.disable_actuators_torque()
        
        POSITION = 3
        EXTENDED_POSITION = 4
        CURRENT_POSITION = 5

        # Configure operating modes for arm joints
        self.write_actuators("Operating_Mode", 
            {
                settings.ActuatorName.SHOULDER_PAN: POSITION, 
                settings.ActuatorName.SHOULDER_LIFT: POSITION, 
                settings.ActuatorName.ELBOW_FLEX: POSITION, 
                settings.ActuatorName.WRIST_FLEX: POSITION, 
                settings.ActuatorName.WRIST_ROLL: POSITION
            })

        # Configure communication
        self.write_actuators("Return_Delay_Time", 0)

        # Gripper uses current position mode
        self.write_actuators("Operating_Mode", {settings.ActuatorName.GRIPPER: CURRENT_POSITION})

        # Enable torque for the gripper and set position
        print(f"Setting gripper open angle...")
        self.enable_actuators_torque([settings.ActuatorName.GRIPPER])
        self.write_actuators_pos({settings.ActuatorName.GRIPPER: self.cfg.gripper_open_degree}, normalize=True)

