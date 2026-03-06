from typing import Any, Dict

from solarian.core.component import Component
from solarian.core.component_bus import ComponentBus
from solarian.core.data import Data

import logging
logger = logging.getLogger(f"module_bus")


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

            # logger.info(f"Connecting {components.keys()} on port {port} bus class {bus_class.__name__}")
            # Create the bus instance
            self.component_bus_by_port[port] = bus_class(port=port, components=components, calibration=calibration.get(port, None))

            
            logger.info(f"Connecting {self.component_bus_by_port[port].__class__.__name__} on port {port}")
            self.component_bus_by_port[port].connect()
            
    

    def disconnect(self):
        for bus in self.component_bus_by_port.values():
            if not bus.is_connected:
                continue
            bus.disconnect()
        self.component_bus_by_port = {}

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

    def configure(self, params: dict[str, int], component_names: list[str]):
        """Write raw hardware-level configuration parameters to components."""
        for port, component_bus in self.component_bus_by_port.items():
            names_bus = [name for name in component_names if name in self.components_by_port[port]]
            if names_bus:
                component_bus.configure(params, names_bus)

    def enable_torque(self, component_names: list[str]):
        """Enable torque on the specified components."""
        for port, component_bus in self.component_bus_by_port.items():
            names_bus = [name for name in component_names if name in self.components_by_port[port]]
            if names_bus:
                component_bus.enable_torque(names_bus)

    def disable_torque(self, component_names: list[str]):
        """Disable torque on the specified components."""
        for port, component_bus in self.component_bus_by_port.items():
            names_bus = [name for name in component_names if name in self.components_by_port[port]]
            if names_bus:
                component_bus.disable_torque(names_bus)

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
