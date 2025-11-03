import time
from omegaconf import DictConfig, OmegaConf

from proto.management import BaseCommand, CommandRegistry
from proto.common.utils import get_class
from proto.conf import settings


@CommandRegistry.register('run-controller', 'controller')
class RunTeleopCommand(BaseCommand):
    help = 'Teleop client that connects to a relay server'

    def get_config_class(self):
        """Return the config class for this command."""
        return get_class(settings.CONTROLLER_CONFIG_CLASS)

    def handle(self, cfg: DictConfig) -> None:
         # Get config class
        # ConfigClass = get_class(settings.CONTROLLER_CONFIG_CLASS)
        # # Instantiate config
        # cfg = ConfigClass(**OmegaConf.to_object(cfg))

        # Create controller instance
        ControllerClass = get_class(settings.CONTROLLER_CLASS)
        controller = ControllerClass(cfg)
        controller.connect()
        controller.initialize()
        controller.start()