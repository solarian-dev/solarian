import time
from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.utils import get_class



@CommandRegistry.register('calibrate-controller', 'controller')
class CalibrateControllerCommand(BaseCommand):
    category = "Controller"
    help = "Calibrate the operator-side hardware (leader arm, pedals, ...)"
    description = """
    Connects the project's Controller and runs its calibrate() hook, so the
    operator-side hardware gets the same treatment as the robot: a leader arm's
    joint ranges and zero, a pedal's rest position, whatever the controller
    declares.

    Writes to the controller's module yamls. Its robot-side counterpart is
    `solarian calibrate`.
    """
    examples = ("solarian calibrate-controller",)

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