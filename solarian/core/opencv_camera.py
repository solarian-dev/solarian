import numpy as np

from solarian.core.configs import CameraConfig
from solarian.core.camera import Camera
from solarian.core.opencv_camera_bus import OpenCVCameraBus
from solarian.core.data import ImageData


class OpenCVCamera(Camera):
    bus_class = OpenCVCameraBus

    rgb_image = ImageData()

    def __init__(self, cfg: CameraConfig | None = None, name: str | None = None):
        super().__init__(cfg)
        if cfg is not None:
            from solarian.core.optical_remap import camera_output_hw

            h, w = camera_output_hw(cfg)
            # Mutate the class-level ImageData descriptor so that SensorField
            # references (used by Observation.zeros() to size shared-memory
            # buffers) reflect the actual camera output dimensions.
            cls_rgb = type(self).__dict__['rgb_image']
            cls_rgb.shape = (h, w, cfg.channels)
            cls_rgb.payload = np.zeros((h, w, cfg.channels), dtype=np.uint8)
