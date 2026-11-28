from solarian.core.camera import Camera
from solarian.core.opencv_camera_bus import OpenCVCameraBus
from solarian.core.data import ImageData


class OpenCVCamera(Camera):
    bus_class = OpenCVCameraBus

    rgb_image = ImageData()
