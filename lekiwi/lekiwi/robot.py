from proto.common.configs import ValueName
from proto.robots.robot import Robot
from proto.robots.action import Action
from proto.common.configs import RobotConfig
from proto.common.utils import profile


from proto.conf import settings


# Must define robot
class LeKiwiRobot(Robot):
    def __init__(self, cfg: RobotConfig):
        super().__init__(cfg)

    def update(self):
        pass

    @profile
    def process_action(self, action: Action):
        """Process received action"""
        # try:
        for module_name, module_data in action.data.items():
            for actuator_name, actuator_data in module_data.items():
                for data_name, data_value in actuator_data.items():
                    if data_name == ValueName.POS:
                        self.modules[module_name].write_actuators_pos({actuator_name: data_value})
                    elif data_name == ValueName.SPEED:
                        self.modules[module_name].write_actuators_speed({actuator_name: data_value})
        
        # except KeyError:
        #     print("KeyError: ", action.data)
