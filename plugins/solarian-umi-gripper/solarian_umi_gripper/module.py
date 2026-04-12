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
        super().initialize()

        self.disable_torque()

        self.configure_components({
            "Return_Delay_Time": 0,
            "Maximum_Acceleration": 254,
            "Acceleration": 254,
            "Operating_Mode": 0,
            "P_Coefficient": 32,
            "I_Coefficient": 2,
            "D_Coefficient": 2,
        })

        self.enable_torque()

