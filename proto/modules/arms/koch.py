
from proto.robots.module import Module
from proto.conf import settings

class KochLeaderArmModule(Module):
    def initialize(self):
        super().initialize()

        # Disable torque for all actuators
        self.disable_actuators_torque()
        
        POSITION = 3
        EXTENDED_POSITION = 4
        CURRENT_POSITION = 5

        self.write_actuators("Operating_Mode", 
            {
                settings.ActuatorName.SHOULDER_PAN: POSITION, 
                settings.ActuatorName.SHOULDER_LIFT: POSITION, 
                settings.ActuatorName.ELBOW_FLEX: POSITION, 
                settings.ActuatorName.WRIST_FLEX: POSITION, 
                settings.ActuatorName.WRIST_ROLL: POSITION
            })

        self.write_actuators("Return_Delay_Time", 0)

        self.write_actuators("Operating_Mode", {settings.ActuatorName.GRIPPER: CURRENT_POSITION})

        # Enable torque for the gripper and set position
        print(f"Setting gripper open angle...")
        self.enable_actuators_torque([settings.ActuatorName.GRIPPER])
        self.write_actuators_pos({settings.ActuatorName.GRIPPER: self.cfg.gripper_open_degree}, normalize=True)