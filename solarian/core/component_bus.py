from __future__ import annotations

import abc
from typing import Any, Dict, Mapping, TYPE_CHECKING

from solarian.core.data import Data
from solarian.core.calibration import MotorCalibration

if TYPE_CHECKING:
    from solarian.core.component import Component

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
