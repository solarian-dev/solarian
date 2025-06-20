import time
from omegaconf import DictConfig, OmegaConf

from proto.management import BaseCommand, CommandRegistry
from proto.common.utils import get_class
from proto.conf import settings


@CommandRegistry.register('write-controller-calibration', 'controller')
class WriteControllerCalibrationCommand(BaseCommand):
    help = 'Writes controller calibration to the controller'

    def handle(self, cfg: DictConfig) -> None:
         # Get config class
        ConfigClass = get_class(settings.CONTROLLER_CONFIG_CLASS)
        # Instantiate config
        cfg = ConfigClass(**OmegaConf.to_object(cfg))

        # Create controller instance
        ControllerClass = get_class(settings.CONTROLLER_CLASS)
        controller = ControllerClass(cfg)
        controller.connect()
        controller.write_calibration()
        controller.disconnect()