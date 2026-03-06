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
            if cfg.resize_width is not None and cfg.resize_height is not None:
                h, w = cfg.resize_height, cfg.resize_width
            elif cfg.rotation in (90, -90):
                h, w = cfg.width, cfg.height
            else:
                h, w = cfg.height, cfg.width
            # Mutate the class-level ImageData descriptor so that SensorField
            # references (used by Observation.zeros() to size shared-memory
            # buffers) reflect the actual camera output dimensions.
            cls_rgb = type(self).__dict__['rgb_image']
            cls_rgb.shape = (h, w, cfg.channels)
            cls_rgb.payload = np.zeros((h, w, cfg.channels), dtype=np.uint8)
