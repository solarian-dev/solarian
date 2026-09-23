from __future__ import annotations

import abc
from typing import Any, Dict, Mapping, TYPE_CHECKING

from solarian.core.clock_sync import synced_time
from solarian.core.data import Data
from solarian.core.calibration import MotorCalibration

if TYPE_CHECKING:
    import numpy as np

    from solarian.core.component import Component
    from solarian.core.recording.video_writer import ArchiveSink

class ComponentBus(abc.ABC):
    """
    Abstract base class for per-port component buses.

    Component buses handle communication with specific hardware types
    on a single port (e.g., a serial port, IP address, or device index).

    Subclasses:
        - ActuatorBus: for actuator hardware (motors, servos, etc.)
        - SensorBus: for sensor hardware (cameras, force sensors, etc.)
    """

    def __init__(self, port: str, components: Mapping[str, Component], calibration=None):
        self.port = port
        self.components = components
        #: Where archive frames go, by component name. Empty unless the robot
        #: wired an encoder to this bus, which it only does for cameras.
        self.archive_sinks: Dict[str, "ArchiveSink"] = {}
        #: When each component's most recent sample was taken. The module reads
        #: this instead of stamping on return, so the observation and the
        #: archive carry the same instant.
        self.capture_timestamps: Dict[str, float] = {}
        #: Known delay between the physical sample and this bus seeing it.
        self.capture_latency: float = 0.0

    def archive_capture(self, name: str, frame: "np.ndarray", cfg) -> float:
        """Send *frame* to the archive encoder and return its capture stamp.

        A camera bus calls this between rotation and the optical remap. The
        remap is a calibration artifact rather than data: kept out of the
        archive, a later recalibration re-derives every dataset already
        recorded; baked in, it could never be undone.

        The stamp returned is also the one the observation must carry. The
        encoder and the recorder each place the frame on the episode's grid by
        computing from it independently, and they can only agree because there
        is exactly one of it.
        """
        captured_at = synced_time() - self.capture_latency
        self.capture_timestamps[name] = captured_at

        sink = self.archive_sinks.get(name)
        if sink is not None:
            sink.write_frame(archive_frame(frame, cfg), captured_at)
        return captured_at

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """True if the bus is currently connected to the hardware."""
        

    @abc.abstractmethod
    def connect(self):
        """Establish connection to the hardware."""
        pass

    @abc.abstractmethod
    def disconnect(self):
        """Close the connection to the hardware."""
        pass

    @abc.abstractmethod
    def read(self, data_type: type[Data], component_names: list[str], normalize: bool = True) -> Dict[str, Any]:
        """Read values from components on this bus."""
        pass

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration]) -> dict[str, MotorCalibration]:
        """Write calibration parameters to the actuators and cache them.

        Args:
            calibration_dict (dict[str, MotorCalibration]): parameters per actuator

        Returns:
            dict[str, MotorCalibration]: parameters per actuator, it might bound the ranges of motion
        """
        raise NotImplementedError(f"{type(self).__name__} does not support write_calibration")

    def start_calibration_mode(self):
        """Sets the actuators where the arm can be moved manually freely"""
        raise NotImplementedError(f"{type(self).__name__} does not support start_calibration_mode")

    def end_calibration_mode(self):
        raise NotImplementedError(f"{type(self).__name__} does not support end_calibration_mode")

    def write(self, values: Dict[str, Data], normalize: bool = True):
        """Write values to components on this bus. Override in writable buses."""
        raise NotImplementedError(f"{type(self).__name__} does not support write operations")

    def is_calibrated(self) -> bool:
        """Check if the components on this bus are calibrated."""
        return True

    def read_calibration(self) -> dict[str, Any]:
        """Read calibration parameters from the components."""
        return {}

    def configure(self, params: dict[str, int], component_names: list[str]):
        """Write raw hardware-level configuration parameters to components.

        Each bus subclass implements this with hardware-specific logic
        (e.g., writing servo registers, setting camera parameters).

        Args:
            params: Mapping of parameter names to values (hardware-specific).
            component_names: Components on this bus to configure.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support configure operations")

    def enable_torque(self, component_names: list[str]):
        """Enable torque on the specified components. Override in actuator buses."""
        raise NotImplementedError(f"{type(self).__name__} does not support torque operations")

    def disable_torque(self, component_names: list[str]):
        """Disable torque on the specified components. Override in actuator buses."""
        raise NotImplementedError(f"{type(self).__name__} does not support torque operations")

    def setup_component(self, component_name: str, configs: Dict[str, Any]):
        """Setup a component on this bus. Override if supported."""
        raise NotImplementedError(f"{type(self).__name__} does not support setup_component")


def archive_frame(frame: "np.ndarray", cfg) -> "np.ndarray":
    """Downscale a capture to the archive resolution, if one is configured.

    Unset archives at whatever resolution the sensor produced. This is the
    knob for encoder CPU on a Raspberry Pi, and it deliberately does not touch
    the sensor mode: selecting a smaller mode would crop the field of view and
    invalidate the camera's calibration.
    """
    if cfg.archive_width is None or cfg.archive_height is None:
        return frame
    if frame.shape[:2] == (int(cfg.archive_height), int(cfg.archive_width)):
        return frame

    import cv2

    return cv2.resize(
        frame,
        (int(cfg.archive_width), int(cfg.archive_height)),
        interpolation=cv2.INTER_AREA,
    )
