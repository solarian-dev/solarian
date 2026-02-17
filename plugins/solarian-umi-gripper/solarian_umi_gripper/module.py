"""
UMI Evo Gripper Module

Gripper module for UMI Evo robot using Feetech STS3215 servos.
"""

from solarian.core.module import Module
from solarian.core.configs import ModuleConfig
from solarian_feetech.actuator import FeetechServo


class UMIGripperConfig(ModuleConfig):
    """Config for UMI Evo gripper module. Extends ModuleConfig with gripper-specific fields."""
    calibration_file: str | None = None   


class UMIGripperModule(Module):
    config_class = UMIGripperConfig
    gripper_actuator = FeetechServo()
    
    def initialize(self):
        """Initialize the gripper with optimal servo settings."""

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
        
        # PID tuning optimized for gripper operation
        self.write_actuators("P_Coefficient", 32)
        self.write_actuators("I_Coefficient", 2)
        self.write_actuators("D_Coefficient", 2)
        
        # Enable torque and lock settings
        self.write_actuators("Torque_Enable", 1)
        self.write_actuators("Lock", 1)

