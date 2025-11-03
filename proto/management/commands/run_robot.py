from omegaconf import DictConfig, OmegaConf

from proto.management import BaseCommand, CommandRegistry
from proto.common.utils import get_class
from proto.conf import settings


@CommandRegistry.register('run-robot', 'robot')
class RunRobotCommand(BaseCommand):
    help = 'Run robot with the specified configuration'

    def get_config_class(self):
        """Return the config class for this command."""
        return get_class(settings.ROBOT_CONFIG_CLASS)

    def handle(self, cfg: DictConfig) -> None:
        # Create robot instance
        RobotClass = get_class(settings.ROBOT_CLASS)
        robot = RobotClass(cfg)
        robot.connect()
        robot.initialize()
        robot.start()


