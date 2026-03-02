from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.configs import RobotConfig
from solarian.utils import get_class

# run-robot is the subcommand, robot is the config name
@CommandRegistry.register('run-robot', 'robot')
class RunRobotCommand(BaseCommand):
    help = 'Run robot with the specified configuration'

    def get_config_class(self):
        """Return the config class for this command."""
        return RobotConfig

    def handle(self, cfg: RobotConfig) -> None:
        from solarian.conf import settings

        # Create robot instance (modules are auto-created from annotations + YAML configs)
        RobotClass = get_class(settings.ROBOT_CLASS)
        robot = RobotClass(cfg)
        robot.connect()
        robot.initialize()
        robot.start()
