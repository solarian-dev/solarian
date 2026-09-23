from __future__ import annotations

import threading
import time
from typing import Any, Dict, TYPE_CHECKING
from enum import Enum

import numpy as np

from solarian.core.sensor_bus import SensorBus
from solarian.core.configs import PiCameraSensorConfig
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


# CSI cameras (IMX477 HQ especially) drop frames or stall on ribbon/power
# glitches. libcamera's dequeue watchdog is ~1s; after that capture_array
# hangs or raises and the pipeline is dead until Picamera2 is recreated.
_CAPTURE_TIMEOUT_S = 1.5
_AE_SETTLE_S = 0.5
_RECONNECT_SETTLE_S = 0.0
_RECONNECT_MIN_DELAY_S = 0.05
_RECONNECT_MAX_DELAY_S = 1.0
_CLOSE_TIMEOUT_S = 2.0
_BUFFER_COUNT = 6


class PiCameraBus(SensorBus):
    """Sensor bus for Raspberry Pi CSI cameras via picamera2.

    ``port`` is the libcamera CSI index (``"0"`` = cam0, ``"1"`` = cam1),
    not a ``/dev/video*`` path. ``picamera2`` is a system package
    (``apt install python3-picamera2``); use a venv created with
    ``--system-site-packages`` so it is importable.

    Capture uses a timeout. On stall, empty frame, or libcamera error the
    bus closes the handle and recreates ``Picamera2`` until the camera
    comes back, so a CSI glitch does not kill the module process.
    """

    def __init__(self, port: str, components: Dict[str, PiCamera], calibration=None):
        super().__init__(port, components, calibration)
        self.picam2 = None
        self.components = components
        self._optical_remap = None
        self._optical_remap_loaded = False
        self.capture_timeout_s = _CAPTURE_TIMEOUT_S
        self.ae_settle_s = _AE_SETTLE_S
        self.reconnect_settle_s = _RECONNECT_SETTLE_S
        self.reconnect_min_delay_s = _RECONNECT_MIN_DELAY_S
        self.reconnect_max_delay_s = _RECONNECT_MAX_DELAY_S
        self.close_timeout_s = _CLOSE_TIMEOUT_S
        self.buffer_count = _BUFFER_COUNT

    @property
    def is_connected(self) -> bool:
        return self.picam2 is not None

    def _cfg(self) -> PiCameraSensorConfig:
        return next(iter(self.components.values())).cfg  # type: ignore[return-value]

    def _camera_num(self) -> int:
        if isinstance(self.port, str) and self.port.startswith("/dev/"):
            raise ValueError(
                f"PiCameraBus port must be a CSI camera index (e.g. '0' or '1'), "
                f"not a V4L2 device path: {self.port!r}. "
                f"CSI cameras on Raspberry Pi use libcamera via picamera2."
            )
        try:
            return int(self.port)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"PiCameraBus port must be an integer CSI index, got {self.port!r}"
            ) from e

    @staticmethod
    def _picamera2_class():
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise ImportError(
                "picamera2 is required for PiCameraBus. On Raspberry Pi OS install "
                "with `sudo apt install python3-picamera2` and create your venv with "
                "`--system-site-packages` so the system package is importable."
            ) from e
        return Picamera2

    def _open(self, *, settle: float) -> None:
        camera_num = self._camera_num()
        Picamera2 = self._picamera2_class()
        cfg = self._cfg()
        # Instantiate only here (after fork/reconnect in the module child process).
        self.picam2 = Picamera2(camera_num=camera_num)
        main = {"format": "RGB888", "size": (int(cfg.width), int(cfg.height))}
        controls = {"FrameRate": float(cfg.fps)}
        if cfg.exposure_value is not None:
            controls["ExposureValue"] = float(cfg.exposure_value)
        if cfg.brightness is not None:
            controls["Brightness"] = float(cfg.brightness)
        if cfg.contrast is not None:
            controls["Contrast"] = float(cfg.contrast)
        if cfg.shutter is not None:
            controls["ExposureTime"] = int(cfg.shutter)
        if cfg.gain is not None:
            controls["AnalogueGain"] = float(cfg.gain)
        if cfg.shutter is not None and cfg.gain is not None:
            controls["AeEnable"] = False
        try:
            config = self.picam2.create_video_configuration(
                main=main,
                controls=controls,
                buffer_count=int(self.buffer_count),
            )
        except TypeError:
            config = self.picam2.create_video_configuration(
                main=main,
                controls=controls,
            )
        self.picam2.configure(config)
        self.picam2.start()
        if not self._optical_remap_loaded:
            from solarian.core.optical_remap import load_bus_remap

            self._optical_remap = load_bus_remap(cfg)
            self._optical_remap_loaded = True
        if settle > 0:
            time.sleep(settle)

    def connect(self):
        if self.picam2 is not None:
            return
        self._open(settle=self.ae_settle_s)

    @staticmethod
    def _close_picam2(picam2) -> None:
        try:
            cancel = getattr(picam2, "cancel_all_and_flush", None)
            if cancel is not None:
                cancel()
        except Exception:  # noqa: BLE001
            logger.debug("picamera2 cancel_all_and_flush failed", exc_info=True)
        try:
            picam2.stop()
            picam2.close()
        except Exception:  # noqa: BLE001
            logger.exception("Error closing picamera2")

    def disconnect(self):
        picam2 = self.picam2
        self.picam2 = None
        if picam2 is None:
            return
        closer = threading.Thread(
            target=self._close_picam2,
            args=(picam2,),
            name="picamera-close",
            daemon=True,
        )
        closer.start()
        closer.join(timeout=self.close_timeout_s)
        if closer.is_alive():
            logger.warning(
                "picamera2 close timed out on CSI %s; abandoning handle",
                self.port,
            )

    def _capture_frame(self):
        if self.picam2 is None:
            raise RuntimeError("Camera not connected")
        timeout = self.capture_timeout_s
        try:
            frame = self.picam2.capture_array("main", wait=timeout)
        except TypeError:
            # Older picamera2: wait is bool/None only.
            frame = self.picam2.capture_array("main")
        if frame is None:
            raise RuntimeError("Could not read frame from camera")
        return frame

    def _reconnect(self, reason: BaseException | str) -> None:
        logger.warning(
            "PiCamera CSI %s dropped (%s); reconnecting",
            self.port,
            reason,
        )
        self.disconnect()
        delay = self.reconnect_min_delay_s
        attempt = 0
        while True:
            attempt += 1
            if delay > 0:
                time.sleep(delay)
            try:
                self._open(settle=self.reconnect_settle_s)
                logger.warning(
                    "PiCamera CSI %s reconnected after %d attempt(s)",
                    self.port,
                    attempt,
                )
                return
            except Exception as exc:  # noqa: BLE001
                delay = (
                    min(delay * 2.0, self.reconnect_max_delay_s)
                    if delay > 0
                    else self.reconnect_min_delay_s
                )
                logger.warning(
                    "PiCamera CSI %s reconnect attempt %d failed: %s; retry in %.2fs",
                    self.port,
                    attempt,
                    exc,
                    delay,
                )

    def _capture_or_reconnect(self):
        while True:
            try:
                return self._capture_frame()
            except Exception as exc:  # noqa: BLE001
                self._reconnect(exc)

    def read(self, data_type: type[Data], component_names: list[str], normalize: bool = True) -> Dict[str, Any]:
        import cv2

        if not issubclass(data_type, ImageData):
            raise ValueError(f"PiCameraBus only supports ImageData, got {data_type}")

        frame = self._capture_or_reconnect()

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[..., :3]

        # picamera2 RGB888 returns a buffer that is already B,G,R for OpenCV
        # (UMI observed quirk). Treat as BGR so color_mode handling matches
        # OpenCVCameraBus.
        processed = np.ascontiguousarray(frame)

        cfg = self._cfg()

        if cfg.color_mode == ColorMode.RGB.value:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        if cfg.rotation == Rotation.ROTATE_90.value:
            processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
        elif cfg.rotation == Rotation.ROTATE_180.value:
            processed = cv2.rotate(processed, cv2.ROTATE_180)
        elif cfg.rotation == Rotation.ROTATE_270.value:
            processed = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)

        camera_name = next(iter(self.components.keys()))
        self.archive_capture(camera_name, processed, cfg)

        from solarian.core.optical_remap import finish_camera_frame

        processed = finish_camera_frame(processed, cfg, self._optical_remap)

        if not isinstance(processed, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(processed)}")

        return {camera_name: processed}

    def is_calibrated(self) -> bool:
        return True
