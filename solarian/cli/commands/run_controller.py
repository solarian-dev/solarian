import logging
import time
from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.configs import ControllerConfig
from solarian.utils import get_class, StatusBoard, setup_process_logging


@CommandRegistry.register('run-controller', 'controller')
class RunTeleopCommand(BaseCommand):
    help = 'Teleop client that connects to a relay server'

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