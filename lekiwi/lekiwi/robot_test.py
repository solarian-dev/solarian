from proto.common.configs import ValueName
from proto.robots.robot import Robot
from proto.robots.action import Action
from proto.common.configs import ProjectConfig
from proto.common.utils import profile


from proto.conf import settings


# Must define robot
class LeKiwiRobot(Robot):
    def __init__(self, cfg: ProjectConfig):
        super().__init__(cfg)

    def update(self):
        pass

    @profile
    def process_action(self, action: Action):
        pass