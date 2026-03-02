"""
SO100 Follower Arm Module

Follower arm module for teleoperation setups. This arm is controlled by
commands received from a leader arm or controller.
"""

from solarian.core.module import Module

from solarian_feetech.actuator import FeetechServo


class SO100FollowerArmModule(Module):
    """
    SO100 robot arm module configured as a follower (controlled arm).
    
    This module uses Feetech STS3215 servos connected via a standard
    Feetech servo bus. The 'feetech' actuator bus is provided by the
    solarian-feetech plugin.
    
    Key features:
        - 6 DOF arm with shoulder, elbow, wrist, and gripper
        - Uses Feetech STS3215 servos
        - Optimized PID tuning for smooth motion
        - Configurable port via ${actuator_port} variable
    """

    shoulder_pan = FeetechServo()
    shoulder_lift = FeetechServo()
    elbow_flex = FeetechServo()
    wrist_flex = FeetechServo()
    wrist_roll = FeetechServo()
    gripper = FeetechServo()
    
    def initialize(self):
        """Initialize the arm with optimal servo settings."""
        super().initialize()

        # Disable torque to configure settings
        self.write_actuators("Torque_Enable", 0)
        self.write_actuators("Lock", 0)
        
        # Configure servo communication
        self.write_actuators("Return_Delay_Time", 0)
        
        # Set maximum acceleration for responsive movement
        # Note: Maximum_Acceleration is not in the official STS3215 Memory Table
        self.write_actuators("Maximum_Acceleration", 254)
        self.write_actuators("Acceleration", 254)
        
        # Position control mode
        self.set_actuators_position_mode()
        
        # PID tuning for smooth motion (reduced P to avoid shakiness)
        self.write_actuators("P_Coefficient", 12)
        self.write_actuators("I_Coefficient", 0)
        self.write_actuators("D_Coefficient", 32)
        
        # Enable torque and lock settings
        self.write_actuators("Torque_Enable", 1)
        self.write_actuators("Lock", 1)

