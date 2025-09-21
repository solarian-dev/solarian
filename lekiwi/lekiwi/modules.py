from dataclasses import asdict
from lerobot.common.motors import MotorCalibration

from proto.robots.module import Module
from proto.common.configs import ModuleConfig
from proto.robots.component_bus import ComponentBus
from proto.conf import settings


class BaseModule(Module):
    def __init__(self, cfg: ModuleConfig, name: str, component_bus: ComponentBus):
        super().__init__(cfg, name, component_bus)

    def setup(self):
        # disable torque
        self.write_actuators("Torque_Enable", 0)
        self.write_actuators("Lock", 0)

        self.write_actuators("Return_Delay_Time", 0)
        # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
        # the motors. Note: this configuration is not in the official STS3215 Memory Table
        self.write_actuators("Maximum_Acceleration", 254)
        self.write_actuators("Acceleration", 254)

        # set to velocity mode TODO: use enum
        self.write_actuators("Operating_Mode", 1)

        # enable torque
        self.write_actuators("Torque_Enable", 1)
        self.write_actuators("Lock", 1)


    def calibrate(self):
        # no calibration needed
        pass

    def is_calibrated(self) -> bool:
        return True


# class ArmModule(Module):
#     def __init__(self, cfg: ModuleConfig, name: str, component_bus: ComponentBus):
#         super().__init__(cfg, name, component_bus)

#     def setup(self):

#         if not self.is_calibrated():
#             self.calibrate()

#         # Disable torque for each arm motor.
#         self.write_actuators("Torque_Enable", 0)
#         self.write_actuators("Lock", 0)
        
#         self.write_actuators("Return_Delay_Time", 0)
#         # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
#         # the motors. Note: this configuration is not in the official STS3215 Memory Table
#         self.write_actuators("Maximum_Acceleration", 254)
#         self.write_actuators("Acceleration", 254)
        
#         # Mode=0 for Position Control
#         self.set_actuators_position_mode()
#         # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
#         # self.write_actuators("P_Coefficient", 16)
#         # self.write_actuators("P_Coefficient", 8)
#         self.write_actuators("P_Coefficient", 12)
#         # Set I_Coefficient and D_Coefficient to default value 0 and 32
#         self.write_actuators("I_Coefficient", 0)
#         self.write_actuators("D_Coefficient", 32)
        
#         # enable torque
#         self.write_actuators("Torque_Enable", 1)
#         self.write_actuators("Lock", 1)
    


class LeaderArmModule(Module):
    def __init__(self, cfg: ModuleConfig, name: str, component_bus: ComponentBus):
        super().__init__(cfg, name, component_bus)
    
    
    def setup(self):
        if not self.is_calibrated():
            self.calibrate()

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

   
        
