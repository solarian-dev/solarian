from typing import Any, Dict
from enum import Enum
import time

from solarian.core.component import Component
from solarian.core.component_bus import ComponentBus
from solarian.core.data import Data
from solarian.utils import manual_profile


class TorqueMode(int, Enum):
    ENABLED = 1
    DISABLED = 0


class ModuleBus():
    """Bus of components across all ports for a module."""

    def __init__(self):
        self.components = []
        self.components_by_port: Dict[str, Dict[str, Component]] = {}
        self.component_bus_by_port: Dict[str, ComponentBus] = {}
        
    def add_components(self, components: Dict[str, Component]):
        for name, component in components.items():
            self.components.append(component)
            if component.cfg is None:
                continue
            if component.cfg.port not in self.components_by_port:
                self.components_by_port[component.cfg.port] = {}
            self.components_by_port[component.cfg.port][name] = component

    def connect(self, calibration: Dict[str, Any] = {}):
        self.component_bus_by_port = {}
        for port, components in self.components_by_port.items():
            # get the first component to determine bus type
            first_component = next(iter(components.values()))
            
            # Get the bus class directly from the component class
            bus_class = first_component.bus_class
            # Create the bus instance
            self.component_bus_by_port[port] = bus_class(port=port, components=components, calibration=calibration.get(port, None))

            try:
                print(f"Connecting {self.component_bus_by_port[port]} on port", port)
                self.component_bus_by_port[port].connect()
            except Exception as e:
                print(f"Failed to connect to component bus on port {port}: {e}")
                print("Check if the device is connected and powered.")
                exit(1)
    

    def disconnect(self):
        for bus in self.component_bus_by_port.values():
            if not bus.is_connected:
                continue
            bus.disconnect()

    def write(self, values: Dict[str, Data], normalize: bool = True):
        """Write values to components."""
        for port, component_bus in self.component_bus_by_port.items():
            # group component writes by port
            values_bus = {name: value for name, value in values.items() if name in self.components_by_port[port]}
            if values_bus:
                component_bus.write(values_bus, normalize)
    
    def read(self, data_type: type[Data], component_names: list[str], normalize: bool = True) -> Dict[str, Any]:
        """Read values from components."""
        result = {}
        for port, component_bus in self.component_bus_by_port.items():
            # group component reads by port
            names_bus = [name for name in component_names if name in self.components_by_port[port]]
            if names_bus:
                data = component_bus.read(data_type, names_bus, normalize=normalize)
                result.update(data)
        return result

    # def write_pos(self, values: Dict[str, Any]):
    #     """Write a value to the actuator group."""
    #     self.write(ValueName.POS, values)

    # def write_speed(self, values: Dict[str, Any]):
    #     """Write a value to the actuator group."""
    #     self.write(ValueName.SPEED, values)

    # def read_pos(self, component_names: list[str]):
    #     """Read a value from the actuator group."""
    #     return self.read(ValueName.POS, component_names)

    # def read_speed(self, component_names: list[str]):
    #     """Read a value from the actuator group."""
    #     return self.read(ValueName.SPEED, component_names)

    # def enable_torque(self, component_names: list[str]):
    #     """Enable torque for the actuator."""
    #     self.write(ValueName.TORQUE_ENABLE, {name: TorqueMode.ENABLED for name in component_names})
        
    # def disable_torque(self, component_names: list[str]):
    #     """Disable torque for the actuator."""
    #     self.write(ValueName.TORQUE_ENABLE, {name: TorqueMode.DISABLED for name in component_names})
        
    # TODO, remove hardcoded strings
    # def set_velocity_mode(self, component_names: list[str]):
    #     self.write("Operating_Mode", {name: 1 for name in component_names})

    # def set_position_mode(self, component_names: list[str]):
    #     self.write("Operating_Mode", {name: 0 for name in component_names})
        
    def is_calibrated(self) -> bool:
        for component_bus in self.component_bus_by_port.values():
            if not component_bus.is_calibrated():
                return False
        return True

    def read_calibration(self) -> dict[str, Any]:
        calibration = {}
        for component_bus in self.component_bus_by_port.values():
            calibration.update(component_bus.read_calibration())
        return calibration

    def setup_component(self, component_name: str, component_port: str, configs: Dict[str, Any]):
        component_bus = self.component_bus_by_port[component_port]
        component_bus.setup_component(component_name, configs)
