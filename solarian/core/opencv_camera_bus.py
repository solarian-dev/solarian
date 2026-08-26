from __future__ import annotations

import sys
from typing import Any, Dict, TYPE_CHECKING
from enum import Enum

import numpy as np

from solarian.core.sensor_bus import SensorBus
from solarian.core.component import Component
from solarian.core.configs import CameraConfig
from solarian.core.data import Data, ImageData

if TYPE_CHECKING:
    import cv2
    from solarian.core.opencv_camera import OpenCVCamera


import logging
logger = logging.getLogger(f"opencv_camera_bus")

_BACKEND_MAP = {
    "linux": "CAP_V4L2",
    "darwin": "CAP_AVFOUNDATION",
    "win32": "CAP_MSMF",
}

def _get_backend():
    import cv2
    return getattr(cv2, _BACKEND_MAP[sys.platform])


class Rotation(int, Enum):
    NO_ROTATION = 0
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = -90


class ColorMode(str, Enum):
    RGB = "rgb"
    BGR = "bgr"


class OpenCVCameraBus(SensorBus):
    """
    Sensor bus for OpenCV cameras.

    Since OpenCV cameras don't share a communication bus (each camera
    has its own device), this bus manages a single camera per instance.
    """

    def __init__(self, port: str, components: Dict[str, OpenCVCamera], calibration=None):
        super().__init__(port, components, calibration)
        self.capture = None
        self.components = components
        self._optical_remap = None

    @property
    def is_connected(self) -> bool:
        return self.capture is not None and self.capture.isOpened()

    def connect(self):
        import cv2

        # logger.info(f"OpenCVCameraBus connecting {self.components.keys()} on port {self.port}")
        cfg: CameraConfig = next(iter(self.components.values())).cfg  # type: ignore[assignment]
        # logger.info(f"Config: {cfg}, class: {type(cfg)}")

        if isinstance(self.port, str) and self.port.startswith('/dev/'):
            port = self.port
        else:
            port = int(self.port)

        backend = _get_backend()
        # logger.info(f"OpenCVCameraBus opening camera on port {port} with backend {backend}")
        self.capture = cv2.VideoCapture(port, backend)

        # logger.info("Camera opened")

        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*cfg.video_format))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(cfg.width))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(cfg.height))
        self.capture.set(cv2.CAP_PROP_FPS, float(cfg.fps))

        if not self.capture.isOpened():
            logger.error(f"Error: Could not open camera with platform backend: {self.port}")
            logger.error("Trying with default backend...")
            self.capture = cv2.VideoCapture(port)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open camera: {self.port}. "
                "Ensure the device is connected and codecs are installed."
            )

        fps = self.capture.get(cv2.CAP_PROP_FPS)
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        from solarian.core.optical_remap import load_bus_remap

        self._optical_remap = load_bus_remap(cfg)
        # logger.info(f"Camera opened: port={self.port}, {width}x{height} @ {fps:.1f} fps")

    def disconnect(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            self._optical_remap = None

    def read(self, data_type: type[Data], component_names: list[str], normalize: bool = True) -> Dict[str, Any]:
        import cv2

        if not issubclass(data_type, ImageData):
            raise ValueError(f"OpenCVCameraBus only supports ImageData, got {data_type}")

        if self.capture is None:
            raise RuntimeError("Camera not connected")

        ret, frame = self.capture.read()
        if not ret:
            raise ValueError("Could not read frame from camera")

        if isinstance(frame, bytes):
            frame = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode frame from camera")

        cfg: CameraConfig = next(iter(self.components.values())).cfg  # type: ignore[assignment]
        processed = frame

        if cfg.color_mode == ColorMode.RGB.value:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        if cfg.rotation == Rotation.ROTATE_90.value:
            processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
        elif cfg.rotation == Rotation.ROTATE_180.value:
            processed = cv2.rotate(processed, cv2.ROTATE_180)
        elif cfg.rotation == Rotation.ROTATE_270.value:
            processed = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)

        from solarian.core.optical_remap import finish_camera_frame

        processed = finish_camera_frame(processed, cfg, self._optical_remap)

        if not isinstance(processed, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(processed)}")

        camera_name = next(iter(self.components.keys()))
        return {camera_name: processed}

    def is_calibrated(self) -> bool:
        return True
