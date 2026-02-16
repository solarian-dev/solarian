from typing import Any, Dict

from solarian.core.sensor_bus import SensorBus
from solarian.core.component import Component


class OpenCVCameraBus(SensorBus):
    """
    Dummy sensor bus that wraps a single OpenCVCamera.

    Since OpenCV cameras don't share a communication bus (each camera
    has its own device), this bus simply delegates all operations
    to the single camera component it contains.
    """

    @property
    def is_connected(self) -> bool:
        camera = next(iter(self.components.values()))
        return camera.is_connected

    def connect(self):
        camera = next(iter(self.components.values()))
        camera.connect()

    def disconnect(self):
        camera = next(iter(self.components.values()))
        camera.disconnect()

    def read(self, data_name: str, component_names: list[str], normalize: bool = True) -> Dict[str, Any]:
        camera = next(iter(self.components.values()))
        camera_name = next(iter(self.components.keys()))
        return {camera_name: camera.read(data_name)}

    def is_calibrated(self) -> bool:
        return True
