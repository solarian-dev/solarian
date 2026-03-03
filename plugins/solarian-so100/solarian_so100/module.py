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

        self.disable_torque()

        self.configure_components({
            "Return_Delay_Time": 0,
            "Maximum_Acceleration": 254,
            "Acceleration": 254,
            "Operating_Mode": 0,
            "P_Coefficient": 12,
            "I_Coefficient": 0,
            "D_Coefficient": 32,
        })

        self.enable_torque()

