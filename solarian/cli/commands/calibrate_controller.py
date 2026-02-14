import time
from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.utils import get_class



@CommandRegistry.register('calibrate-controller', 'controller')
class CalibrateControllerCommand(BaseCommand):
    help = 'Calibrates controller'

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
        controller.calibrate()
        controller.disconnect()