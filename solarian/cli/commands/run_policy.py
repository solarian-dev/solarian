import time
from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.utils import get_class


@CommandRegistry.register('run-policy', 'controller')
class RunPolicyCommand(BaseCommand):
    help = 'Policy client that connects to a relay server'

    def handle(self, cfg: DictConfig) -> None:
        from solarian.conf import settings

         # Get config class
        ConfigClass = get_class(settings.CONTROLLER_CONFIG_CLASS)
        # Instantiate config
        cfg = ConfigClass(**OmegaConf.to_object(cfg))

        # Create controller instance
        ControllerClass = get_class(settings.CONTROLLER_CLASS)
        controller = ControllerClass(cfg)
        controller.connect()
        controller.initialize()
        controller.control_loop()