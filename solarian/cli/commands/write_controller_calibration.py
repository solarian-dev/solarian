import time
from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.utils import get_class


@CommandRegistry.register('write-controller-calibration', 'controller')
class WriteControllerCalibrationCommand(BaseCommand):
    category = "Controller"
    help = "Write the calibration in the controller's yamls onto its hardware"
    description = """
    Pushes the calibration already in the controller's module yamls onto the
    operator-side hardware, connecting every module the Controller class declares
    and writing each one's offsets and limits to its actuators.

    The direction is yaml -> hardware, so nothing is measured. Its robot-side
    counterpart is `solarian write-robot-calibration`.
    """
    examples = ("solarian write-controller-calibration",)
    
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
        controller.write_calibration()
        controller.disconnect()