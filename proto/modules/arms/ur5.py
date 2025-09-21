from proto.robots.module import Module
from proto.common.configs import ModuleConfig
from proto.robots.component_bus import ComponentBus
from proto.conf import settings


class UR5FollowerArmModule(Module):
    def __init__(self, cfg: ModuleConfig, name: str, component_bus: ComponentBus):
        super().__init__(cfg, name, component_bus)

    def setup(self):
        # move to home
        # self.write_actuators_pos({
        #     settings.ActuatorName.SHOULDER_PAN: 0,
        #     settings.ActuatorName.SHOULDER_LIFT: -1.57,
        #     settings.ActuatorName.ELBOW_FLEX: 1.57,
        #     settings.ActuatorName.WRIST_FLEX: 0,
        #     settings.ActuatorName.WRIST_YAW: 1.57,
        #     settings.ActuatorName.WRIST_ROLL: 0
        # })
        
        if not self.is_calibrated():
            self.calibrate()
        
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass
        

