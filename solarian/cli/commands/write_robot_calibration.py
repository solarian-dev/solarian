import time
from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.utils import get_class


@CommandRegistry.register('write-robot-calibration', 'robot')
class WriteRobotCalibrationCommand(BaseCommand):
    help = 'Write robot calibration from configuration.'

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

