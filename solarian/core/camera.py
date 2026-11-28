from typing import Any, Dict

from solarian.core.configs import CameraConfig
from solarian.core.sensor import Sensor


class Camera(Sensor):
    kind: str = "camera"

    def data_shapes(self) -> dict[str, tuple[int, ...]]:
        if self.cfg is None:
            return {}
        from solarian.core.optical_remap import camera_output_hw

        h, w = camera_output_hw(self.cfg)
        return {"rgb_image": (h, w, self.cfg.channels)}
