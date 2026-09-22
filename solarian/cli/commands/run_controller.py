import logging
import time
from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.configs import ControllerConfig
from solarian.utils import get_class, StatusBoard, setup_process_logging


@CommandRegistry.register('run-controller', 'controller')
class RunTeleopCommand(BaseCommand):
    help = "Teleop client: drive the robot through the relay"
    description = """
    Builds the project's Controller class, connects its modules (leader arm,
    gamepad, keyboard, whatever the project declares) and dials the relay to drive
    the robot. Runs until Ctrl-C.

    This is the operator-side process: the robot (`solarian run-robot`) and the
    relay (`solarian run-server`) have to be up, and all three have to agree on the
    relay address, which is what the deployment profile decides. The ui key selects
    the front end (console, pygame or external).
    """
    examples = (
        "solarian run-controller",
        "solarian run-controller -d pi_relay_mac",
        "solarian run-controller ui=external",
    )

    def get_config_class(self):
        from solarian.conf import settings
        """Return the config class for this command."""
        return ControllerConfig

    def handle(self, cfg: DictConfig) -> None:
        from solarian.conf import settings

        log_level = getattr(logging, getattr(settings, 'LOG_LEVEL', 'INFO'), logging.INFO)
        setup_process_logging(level=log_level, logger_name=None)
        ControllerClass = get_class(settings.CONTROLLER_CLASS)
        controller = ControllerClass(cfg)
        try:
            controller.connect()
            controller.initialize()
            controller.start()
        except KeyboardInterrupt:
            StatusBoard.stop()
        except Exception:
            StatusBoard.print_exception()