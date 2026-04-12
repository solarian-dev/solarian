"""
LeKiwi Mobile Base Module

Mobile base module for LeKiwi robot using Feetech STS3215 servos for wheel control.
"""

from solarian.core.module import Module

from solarian_feetech.actuator import FeetechServo


class LekiwiBaseModule(Module):
    """
    LeKiwi mobile base module.
    
    Uses Feetech STS3215 servos for wheel control in velocity mode.
    Includes a front camera sensor for navigation.
    
    Key features:
        - 3-wheel omni-directional base
        - Uses Feetech STS3215 servos (left, right, back wheels)
        - Velocity control mode
        - Front camera sensor
        - Configurable ports via ${actuator_port} and ${front_camera_port} variables
    """

    left_wheel = FeetechServo()
    back_wheel = FeetechServo()
    right_wheel = FeetechServo()
    
    def initialize(self):
        """Initialize the mobile base with optimal servo settings."""
        super().initialize()
        
        # Set maximum acceleration for responsive movement
        # Note: Maximum_Acceleration is not in the official STS3215 Memory Table
        self.write_actuators("Maximum_Acceleration", 254)
        self.write_actuators("Acceleration", 254)

        # Set to velocity mode for wheel control
        self.write_actuators("Operating_Mode", 1)

        # Enable torque and lock settings
        self.write_actuators("Torque_Enable", 1)
        self.write_actuators("Lock", 1)

