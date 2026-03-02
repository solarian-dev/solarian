import time
import hydra
from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.configs import ServerConfig
from solarian.utils import get_class


@CommandRegistry.register('run-server', 'server')
class RunServerCommand(BaseCommand):
    help = 'Run the relay server'

    def get_config_class(self):
        from solarian.conf import settings

        """Return the config class for this command."""
        return ServerConfig

    def handle(self, cfg: DictConfig) -> None:
        from solarian.conf import settings

        # Create server instance
        ServerClass = get_class(settings.SERVER_CLASS)
        server = ServerClass(cfg)
        server.start()
        
