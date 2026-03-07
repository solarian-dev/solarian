from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.configs import RobotConfig
from solarian.utils import get_class, StatusBoard, setup_process_logging

# run-robot is the subcommand, robot is the config name
@CommandRegistry.register('run-robot', 'robot')
class RunRobotCommand(BaseCommand):
    help = 'Run robot with the specified configuration'

    def get_config_class(self):
        """Return the config class for this command."""
        return RobotConfig

    def handle(self, cfg: RobotConfig) -> None:
        from solarian.conf import settings

        setup_process_logging(logger_name=None)
        RobotClass = get_class(settings.ROBOT_CLASS)
        robot = RobotClass(cfg)
        try:
            robot.connect()
            robot.initialize()
            robot.start()
        except KeyboardInterrupt:
            StatusBoard.stop()
        except Exception:
            StatusBoard.print_exception()
