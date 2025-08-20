import time
import hydra
from omegaconf import DictConfig, OmegaConf

from proto.management import BaseCommand, CommandRegistry
from proto.common.utils import get_class
from proto.conf import settings


@CommandRegistry.register('run-server', 'server')
class RunServerCommand(BaseCommand):
    help = 'Run the relay server'

    # @hydra.main(version_base=None, config_path="conf", config_name="server")
    # def handle(self, args: DictConfig) -> None:
    def handle(self, cfg: DictConfig) -> None:
        # Get config class
        ConfigClass = get_class(settings.SERVER_CONFIG_CLASS)
        # Instantiate config
        cfg = ConfigClass(**OmegaConf.to_object(cfg))

        # Create server instance
        ServerClass = get_class(settings.SERVER_CLASS)
        server = ServerClass(cfg)
        server.start()
        
