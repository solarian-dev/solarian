from typing import Tuple
from enum import Enum



class BodyName(Enum):
    pass
    def __str__(self):
        return self.value


class Body:
    # def __init__(self, name: str, cfg: BodyConfig):
    def __init__(self, name: BodyName, parent: BodyName, pos: Tuple[float, float, float] = (0, 0, 0), quat: Tuple[float, float, float, float] = (0, 0, 0, 1)):
        self.name = name
        self.parent = parent
        self.pos = pos
        self.quat = quat
        
        # self.actuators = {}
        # self.sensors = {}
        # for sensor_name, sensor_cfg in cfg.sensors.items():
        #     self.sensors[sensor_name] = SensorFactory.create_sensor(sensor_cfg, sensor_name)
        # for actuator_name, actuator_cfg in cfg.actuators.items():
        #     self.actuators[actuator_name] = ActuatorFactory.create_actuator(actuator_cfg, actuator_name)

    # def get_actuators(self) -> Dict[str, Actuator]:
    #     return self.actuators

    # def get_sensors(self) -> Dict[str, Sensor]:
    #     return self.sensors

    # def add_actuator(self, actuator: Actuator):
    #     self.actuators[actuator.id] = actuator

    # def add_sensor(self, sensor: Sensor):
    #     self.sensors[sensor.id] = sensor

    # def add_servo(self, servo: Servo):
    #     self.add_actuator(servo)
    #     self.add_sensor(servo)
            