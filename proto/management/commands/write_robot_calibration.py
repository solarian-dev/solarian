import time
from omegaconf import DictConfig, OmegaConf

from proto.management import BaseCommand, CommandRegistry
from proto.common.utils import get_class
from proto.conf import settings


@CommandRegistry.register('write-robot-calibration', 'robot')
class WriteRobotCalibrationCommand(BaseCommand):
    help = 'Write robot calibration from configuration.'

    def handle(self, cfg: DictConfig) -> None:
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

