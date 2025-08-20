from omegaconf import DictConfig, OmegaConf

from proto.management import BaseCommand, CommandRegistry
from proto.common.utils import get_class
from proto.conf import settings


@CommandRegistry.register('run-robot', 'robot')
class RunRobotCommand(BaseCommand):
    help = 'Run robot with the specified configuration'

    def handle(self, cfg: DictConfig) -> None:
        # Get config class
        ConfigClass = get_class(settings.ROBOT_CONFIG_CLASS)
        # Instantiate config
        cfg = ConfigClass(**OmegaConf.to_object(cfg))

        # Create robot instance
        RobotClass = get_class(settings.ROBOT_CLASS)
        robot = RobotClass(cfg)
        robot.connect()
        robot.setup()
        robot.start()


