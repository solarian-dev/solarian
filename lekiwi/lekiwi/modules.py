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
        super().setup()
        
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


    # def calibrate(self):
    #     # no calibration needed
    #     pass

    # def is_calibrated(self) -> bool:
    #     return True
