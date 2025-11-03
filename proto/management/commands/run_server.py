import time
import hydra
from omegaconf import DictConfig, OmegaConf

from proto.management import BaseCommand, CommandRegistry
from proto.common.utils import get_class
from proto.conf import settings


@CommandRegistry.register('run-server', 'server')
class RunServerCommand(BaseCommand):
    help = 'Run the relay server'

    def get_config_class(self):
        """Return the config class for this command."""
        return get_class(settings.SERVER_CONFIG_CLASS)

    # @hydra.main(version_base=None, config_path="conf", config_name="server")
    # def handle(self, args: DictConfig) -> None:
    def handle(self, cfg: DictConfig) -> None:
        # Create server instance
        ServerClass = get_class(settings.SERVER_CLASS)
        server = ServerClass(cfg)
        server.start()
        
