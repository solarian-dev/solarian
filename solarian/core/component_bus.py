import abc
from typing import Any, Dict

from solarian.core.component import Component
from solarian.core.data import Data

class ComponentBus(abc.ABC):
    """
    Abstract base class for per-port component buses.

    Component buses handle communication with specific hardware types
    on a single port (e.g., a serial port, IP address, or device index).

    Subclasses:
        - ActuatorBus: for actuator hardware (motors, servos, etc.)
        - SensorBus: for sensor hardware (cameras, force sensors, etc.)
    """

    def __init__(self, port: str, components: Dict[str, Component]):
        self.port = port
        self.components = components

    @classmethod
    def create(cls, port: str, components: Dict[str, Component], calibration=None) -> "ComponentBus":
        """
        Factory method to create a bus instance.

        Override this in subclasses that need special construction logic.

        Args:
            port: The communication port (e.g., '/dev/ttyUSB0' or IP address)
            components: Dict of component name to Component
            calibration: Optional calibration data (used by ActuatorBus subclasses)

        Returns:
            An instance of the ComponentBus subclass
        """
        return cls(port=port, components=components)

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """True if the bus is currently connected to the hardware."""
        pass

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

    def write(self, values: Dict[str, Data], normalize: bool = True):
        """Write values to components on this bus. Override in writable buses."""
        raise NotImplementedError(f"{type(self).__name__} does not support write operations")

    def is_calibrated(self) -> bool:
        """Check if the components on this bus are calibrated."""
        return True

    def read_calibration(self) -> dict[str, Any]:
        """Read calibration parameters from the components."""
        return {}

    def setup_component(self, component_name: str, configs: Dict[str, Any]):
        """Setup a component on this bus. Override if supported."""
        raise NotImplementedError(f"{type(self).__name__} does not support setup_component")
