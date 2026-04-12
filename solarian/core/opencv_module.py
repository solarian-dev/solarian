"""
OpenCV Camera Module

Module for OpenCV cameras.
"""

from solarian.core.module import Module
from solarian.core.configs import CameraConfig, ModuleConfig
from solarian.core.opencv_camera import OpenCVCamera


class OpenCVCameraConfig(ModuleConfig):
    calibration_file: str | None = None


class OpenCVCameraModule(Module):
    """
    OpenCV camera module.
    """

    config_class = OpenCVCameraConfig
    
    camera = OpenCVCamera()


