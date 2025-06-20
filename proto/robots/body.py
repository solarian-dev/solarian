from typing import Dict
from .sensor_factory import SensorFactory
from proto.common.configs import BodyConfig
from proto.robots.sensor import Sensor


class Body:
    def __init__(self, name: str, cfg: BodyConfig):
        self.name = name
        self.cfg = cfg
        self.sensors = {}
        for sensor_name, sensor_cfg in cfg.sensors.items():
            self.sensors[sensor_name] = SensorFactory.create_sensor(sensor_cfg, sensor_name)

    def get_sensors(self) -> Dict[str, Sensor]:
        return self.sensors
            