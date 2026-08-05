from __future__ import annotations

import time
from typing import Any, Dict, TYPE_CHECKING
from enum import Enum

import numpy as np

from solarian.core.sensor_bus import SensorBus
from solarian.core.configs import CameraConfig
from solarian.core.data import Data, ImageData

if TYPE_CHECKING:
    from solarian.core.picamera import PiCamera


import logging
logger = logging.getLogger("picamera_bus")


class Rotation(int, Enum):
    NO_ROTATION = 0
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = -90


class ColorMode(str, Enum):
    RGB = "rgb"
    BGR = "bgr"


class PiCameraBus(SensorBus):
    """Sensor bus for Raspberry Pi CSI cameras via picamera2.

    ``port`` is the libcamera CSI index (``"0"`` = cam0, ``"1"`` = cam1),
    not a ``/dev/video*`` path. ``picamera2`` is a system package
    (``apt install python3-picamera2``); use a venv created with
    ``--system-site-packages`` so it is importable.
    """

    def __init__(self, port: str, components: Dict[str, PiCamera], calibration=None):
        super().__init__(port, components, calibration)
        self.picam2 = None
        self.components = components

    @property
    def is_connected(self) -> bool:
        return self.picam2 is not None

    def connect(self):
        if isinstance(self.port, str) and self.port.startswith("/dev/"):
            raise ValueError(
                f"PiCameraBus port must be a CSI camera index (e.g. '0' or '1'), "
                f"not a V4L2 device path: {self.port!r}. "
                f"CSI cameras on Raspberry Pi use libcamera via picamera2."
            )

        try:
            camera_num = int(self.port)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"PiCameraBus port must be an integer CSI index, got {self.port!r}"
            ) from e

        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise ImportError(
                "picamera2 is required for PiCameraBus. On Raspberry Pi OS install "
                "with `sudo apt install python3-picamera2` and create your venv with "
                "`--system-site-packages` so the system package is importable."
            ) from e

        cfg: CameraConfig = next(iter(self.components.values())).cfg  # type: ignore[assignment]

        # Instantiate only here (after fork/reconnect in the module child process).
        self.picam2 = Picamera2(camera_num=camera_num)
        main = {"format": "RGB888", "size": (int(cfg.width), int(cfg.height))}
        config = self.picam2.create_video_configuration(
            main=main,
            controls={"FrameRate": float(cfg.fps)},
        )
        self.picam2.configure(config)
        self.picam2.start()
        # Let AE/AWB begin to converge before the first read.
        time.sleep(0.5)

    def disconnect(self):
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:  # noqa: BLE001
                logger.exception("Error closing picamera2")
            self.picam2 = None

    def read(self, data_type: type[Data], component_names: list[str], normalize: bool = True) -> Dict[str, Any]:
        import cv2

        if not issubclass(data_type, ImageData):
            raise ValueError(f"PiCameraBus only supports ImageData, got {data_type}")

        if self.picam2 is None:
            raise RuntimeError("Camera not connected")

        frame = self.picam2.capture_array("main")
        if frame is None:
            raise ValueError("Could not read frame from camera")

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[..., :3]

        # picamera2 RGB888 returns a buffer that is already B,G,R for OpenCV
        # (UMI observed quirk). Treat as BGR so color_mode handling matches
        # OpenCVCameraBus.
        processed = np.ascontiguousarray(frame)

        cfg: CameraConfig = next(iter(self.components.values())).cfg  # type: ignore[assignment]

        if cfg.color_mode == ColorMode.RGB.value:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        if cfg.rotation == Rotation.ROTATE_90.value:
            processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
        elif cfg.rotation == Rotation.ROTATE_180.value:
            processed = cv2.rotate(processed, cv2.ROTATE_180)
        elif cfg.rotation == Rotation.ROTATE_270.value:
            processed = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if cfg.resize_width is not None and cfg.resize_height is not None:
            if processed.shape[0] != cfg.resize_height or processed.shape[1] != cfg.resize_width:
                processed = cv2.resize(
                    processed,
                    (cfg.resize_width, cfg.resize_height),
                    interpolation=cv2.INTER_AREA,
                )

        if not isinstance(processed, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(processed)}")

        camera_name = next(iter(self.components.keys()))
        return {camera_name: processed}

    def is_calibrated(self) -> bool:
        return True
