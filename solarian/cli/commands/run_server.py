import logging
import time
import hydra
from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.configs import ServerConfig
from solarian.utils import get_class, StatusBoard, setup_process_logging


@CommandRegistry.register('run-server', 'server')
class RunServerCommand(BaseCommand):
    help = 'Run the relay server'

    def get_config_class(self):
        from solarian.conf import settings

        """Return the config class for this command."""
        return ServerConfig

    def handle(self, cfg: DictConfig) -> None:
        from solarian.conf import settings

        log_level = getattr(logging, getattr(settings, 'LOG_LEVEL', 'INFO'), logging.INFO)
        setup_process_logging(level=log_level, logger_name=None)
        ServerClass = get_class(settings.SERVER_CLASS)
        server = ServerClass(cfg)
        try:
            server.start()
        except KeyboardInterrupt:
            StatusBoard.stop()
        except Exception:
            StatusBoard.print_exception()
