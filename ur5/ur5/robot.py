from proto.common.configs import ValueName
from proto.robots.robot import Robot
from proto.robots.action import Action
from proto.common.configs import RobotConfig
from proto.common.utils import profile


from proto.conf import settings


JOINT_MAP = {
    settings.ActuatorName.SHOULDER_PAN: {
        "start_pos": -90,
        "end_pos": 90,
    },
    settings.ActuatorName.SHOULDER_LIFT: {
        "start_pos": -0,
        "end_pos": 0,
    },
    settings.ActuatorName.ELBOW_FLEX:{
        "start_pos": 0,
        "end_pos": 180,
    },
    settings.ActuatorName.WRIST_FLEX:{
        "start_pos": 0,
        "end_pos": 180,
    },
    settings.ActuatorName.WRIST_YAW: {
        "start_pos": 0,
        "end_pos": 180,
    },
    settings.ActuatorName.WRIST_ROLL: {
        "start_pos": 0,
        "end_pos": 180,
    },
}


# Must define robot
class UR5Robot(Robot):
    def __init__(self, cfg: RobotConfig):
        super().__init__(cfg)

    def update(self):
        pass

    @profile
    def process_action(self, action: Action):
        """Process received action"""

        print("Action:", action.data)
        try:
            for module_name, module_data in action.data.items():
                for actuator_name, actuator_data in module_data.items():
                    for data_name, data_value in actuator_data.items():
                        if data_name == ValueName.POS:
                            self.modules[module_name].write_actuators_pos({actuator_name: data_value})
                        elif data_name == ValueName.SPEED:
                            self.modules[module_name].write_actuators_speed({actuator_name: data_value})
                        # print(module_name, actuator_name, data_name, data_value)
        
        except KeyError:
            print("KeyError: ", action.data)

        
        
