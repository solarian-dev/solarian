from proto.common.configs import SensorConfig, ComponentType
from proto.robots.opencv_camera import OpenCVCamera
from proto.robots.sensor import Sensor


sensors = {
    ComponentType.OPENCV: OpenCVCamera,
}

class SensorFactory:
    
    @staticmethod
    def create_sensor(sensor_cfg: SensorConfig, name: str) -> Sensor:
        if sensor_cfg.type not in sensors:
            raise ValueError(f"Unknown sensor type: {sensor_cfg.type}")
        
        return sensors[sensor_cfg.type](sensor_cfg, name)
