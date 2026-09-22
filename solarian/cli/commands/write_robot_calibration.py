import time
from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.utils import get_class


@CommandRegistry.register('write-robot-calibration', 'robot')
class WriteRobotCalibrationCommand(BaseCommand):
    help = "Write the calibration in the robot's yamls onto its hardware"
    description = """
    Pushes the calibration already in the robot's module yamls onto the hardware,
    connecting every module the Robot class declares and writing each one's
    offsets and limits to its actuators.

    The direction is yaml -> hardware, so nothing is measured here. Use it after
    editing a calibration by hand, or after swapping in a servo that has the
    factory registers, to restore what the project believes without re-running
    `solarian calibrate`.
    """
    examples = (
        "solarian write-robot-calibration",
        "solarian write-robot-calibration -d pi",
    )

    def handle(self, cfg: DictConfig) -> None:
        from solarian.conf import settings

        # Get config class
        ConfigClass = get_class(settings.ROBOT_CONFIG_CLASS)
        # Instantiate config
        cfg = ConfigClass(**OmegaConf.to_object(cfg))

        # Create robot instance
        RobotClass = get_class(settings.ROBOT_CLASS)
        robot = RobotClass(cfg)
        robot.connect()
        robot.write_calibration()
        robot.disconnect()

