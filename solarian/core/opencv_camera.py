from solarian.core.configs import CameraConfig
from solarian.core.camera import Camera
from solarian.core.opencv_camera_bus import OpenCVCameraBus
from solarian.core.data import ImageData


class OpenCVCamera(Camera):
    bus_class = OpenCVCameraBus

    rgb_image = ImageData(shape=(224, 224, 3))

    def __init__(self, cfg: CameraConfig | None = None, name: str | None = None):
        super().__init__(cfg)
