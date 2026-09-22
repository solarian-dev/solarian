import logging
from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.configs import RobotConfig
from solarian.utils import get_class, StatusBoard, setup_process_logging

# run-robot is the subcommand, robot is the config name
@CommandRegistry.register('run-robot', 'robot')
class RunRobotCommand(BaseCommand):
    help = "Run the robot: connect its modules and join the relay a controller drives it through"
    description = """
    Builds the project's Robot class, connects every module it declares, and joins
    the relay so a controller or a policy can drive it. Runs until Ctrl-C.

    This is the robot-side process: run it on the machine the hardware is wired
    to, with the deployment profile that names that machine's device ports. It
    holds the actuator buses and cameras open, so calibration and camera tools
    cannot run at the same time.
    """
    examples = (
        "solarian run-robot",
        "solarian run-robot -d pi",
        "solarian run-robot fps=60",
    )

    def get_config_class(self):
        """Return the config class for this command."""
        return RobotConfig

    def handle(self, cfg: RobotConfig) -> None:
        from solarian.conf import settings

        log_level = getattr(logging, getattr(settings, 'LOG_LEVEL', 'INFO'), logging.INFO)
        setup_process_logging(level=log_level, logger_name=None)
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
