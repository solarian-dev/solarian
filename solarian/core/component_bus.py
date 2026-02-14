from typing import Any, Dict
from enum import Enum
import time

# from lerobot.common.motors import MotorCalibration
# # from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode

from solarian.core.opencv_camera import OpenCVCamera
from solarian.core.component import Component
from solarian.core.configs import ValueName
from solarian.utils import manual_profile
from solarian.plugins import actuator_buses


class TorqueMode(int, Enum):
    ENABLED = 1
    DISABLED = 0

DATA_NAME_READ_MAP = {
    # FeetechMotorsBus: {
    #     ValueName.POS: "Present_Position",
    #     ValueName.SPEED: "Present_Speed",
    # },
    # DynamixelMotorsBus: {
    #     ValueName.POS: "Present_Position",
    #     ValueName.SPEED: "Present_Speed",
    # },
    OpenCVCamera: {
        ValueName.IMAGE: ValueName.IMAGE,
    },
    # UR5Bus: {
    #     ValueName.POS: "Present_Position",
    #     ValueName.SPEED: "Present_Speed",
    # },
}

# DATA_NAME_WRITE_MAP = {
#     FeetechMotorsBus: {
#         ValueName.POS: "Goal_Position",
#         ValueName.SPEED: "Goal_Velocity",
#         ValueName.TORQUE_ENABLE: "Torque_Enable",
#     },
#     DynamixelMotorsBus: {
#         ValueName.POS: "Goal_Position",
#         ValueName.SPEED: "Goal_Speed",
#         ValueName.TORQUE_ENABLE: "Torque_Enable",
#     },
#     UR5Bus: {
#         ValueName.POS: "Goal_Position",
#         ValueName.SPEED: "Goal_Speed",
#         ValueName.TORQUE_ENABLE: "Torque_Enable",
#     }
# }

# def map_write(data_name: str, component_bus):
#     if data_name in DATA_NAME_WRITE_MAP[type(component_bus)]:
#         return DATA_NAME_WRITE_MAP[type(component_bus)][data_name]
#     return data_name

# def map_read(data_name: str, component_bus):
#     if data_name in DATA_NAME_READ_MAP[type(component_bus)]:
#         return DATA_NAME_READ_MAP[type(component_bus)][data_name]
#     return data_name

# def map_motor_mode(mode:str):
#     if mode == "range_0_100":
#         return MotorNormMode.RANGE_0_100
#     elif mode == "range_m100_100":
#         return MotorNormMode.RANGE_M100_100
#     elif mode == "degrees":
#         return MotorNormMode.DEGREES
#     else:
#         raise ValueError(f"Unknown motor mode: {mode}")

class ComponentBus():
    """Bus of components"""

    def __init__(self):
        self.components = []
        self.components_by_port: Dict[str, Dict[str, Component]] = {}
        self.component_bus_by_port: Dict[str, ComponentBus] = {}
        
    def add_components(self, components: Dict[str, Component]):
        for name, component in components.items():
            self.components.append(component)
            if component.cfg.port not in self.components_by_port:
                self.components_by_port[component.cfg.port] = {}
            self.components_by_port[component.cfg.port][name] = component

    def connect(self, calibration: Dict[str, Any] = {}):
        self.component_bus_by_port = {}
        for port, components in self.components_by_port.items():
            # get the first component to determine bus type
            first_component = next(iter(components.values()))
            
            # Special handling for cameras (they don't share ports)
            if isinstance(first_component, OpenCVCamera):
                self.component_bus_by_port[port] = first_component
            else:
                # Get bus type from the actuator CLASS itself
                bus_type = self._get_bus_type_for_component(first_component)
                
                # Look up the bus class from the registry
                bus_cls = actuator_buses.get(bus_type)
                if bus_cls is None:
                    available = ', '.join(actuator_buses.available())
                    raise ValueError(
                        f"No actuator bus registered for type '{bus_type}'. "
                        f"Available buses: [{available}]. "
                        f"Register a bus via entry points in pyproject.toml."
                    )
                
                # Create the bus instance
                self.component_bus_by_port[port] = bus_cls.create(
                    port=port,
                    components=components,
                    calibration=calibration.get(port, None)
                )

            try:
                self.component_bus_by_port[port].connect()
            except Exception as e:
                print(f"Failed to connect to component bus on port {port}: {e}")
                print("Check if the device is connected and powered.")
                exit(1)
    
    def _get_bus_type_for_component(self, component: Component) -> str:
        """
        Determine the bus type for a component.
        
        Tries to get bus type from the component class first (new approach),
        falls back to config.type for backward compatibility.
        """
        from solarian.core.actuator import Actuator
        
        # New approach: get bus type from the actuator class
        if isinstance(component, Actuator):
            try:
                return component.get_bus_type()
            except NotImplementedError:
                pass
        
        # Fallback: use config.type for backward compatibility with YAML-based configs
        if hasattr(component, 'cfg') and hasattr(component.cfg, 'type'):
            return component.cfg.type
        
        raise ValueError(
            f"Cannot determine bus type for component {component}. "
            f"Either define _bus_type on the actuator class or specify 'type' in config."
        )

    def disconnect(self):
        for bus in self.component_bus_by_port.values():
            if not bus.is_connected:
                continue
            bus.disconnect()

    def write(self, data_name: str, values: Dict[str, Any], normalize: bool = True):
        """Write values to components."""
        for port, component_bus in self.component_bus_by_port.items():
            # group component writes by port
            values_bus = {name: value for name, value in values.items() if name in self.components_by_port[port]}
            if values_bus:
                component_bus.write(data_name, values_bus, normalize=normalize)
    
    def read(self, data_name: str, component_names: list[str], normalize: bool = True) -> Dict[str, Any]:
        """Read values from components."""
        result = {}
        for port, component_bus in self.component_bus_by_port.items():
            # group component reads by port
            names_bus = [name for name in component_names if name in self.components_by_port[port]]
            if names_bus:
                if isinstance(component_bus, OpenCVCamera):
                    # data_name = map_read(data_name, component_bus)
                    data = {names_bus[0]: component_bus.read(data_name)}
                else:
                    data = component_bus.read(data_name, names_bus, normalize=normalize)
                    
                result.update(data)
        return result

    def write_pos(self, values: Dict[str, Any]):
        """Write a value to the actuator group."""
        self.write(ValueName.POS, values)

    def write_speed(self, values: Dict[str, Any]):
        """Write a value to the actuator group."""
        self.write(ValueName.SPEED, values)

    def read_pos(self, component_names: list[str]):
        """Read a value from the actuator group."""
        return self.read(ValueName.POS, component_names)

    def read_speed(self, component_names: list[str]):
        """Read a value from the actuator group."""
        return self.read(ValueName.SPEED, component_names)

    def enable_torque(self, component_names: list[str]):
        """Enable torque for the actuator."""
        self.write(ValueName.TORQUE_ENABLE, {name: TorqueMode.ENABLED for name in component_names})
        # self.write("Lock", {name: 1 for name in component_names})
        
    def disable_torque(self, component_names: list[str]):
        """Disable torque for the actuator."""
        self.write(ValueName.TORQUE_ENABLE, {name: TorqueMode.DISABLED for name in component_names})
        # self.write("Lock", {name: 0 for name in component_names})
        
    # TODO, remove hardcoded strings
    def set_velocity_mode(self, component_names: list[str]):
        # self.write("Lock", {name: 0 for name in component_names})
        self.write("Operating_Mode", {name: 1 for name in component_names})
        # self.write("Lock", {name: 1 for name in component_names})

    def set_position_mode(self, component_names: list[str]):
        # self.write("Lock", {name: 0 for name in component_names})
        self.write("Operating_Mode", {name: 0 for name in component_names})
        # self.write("Lock", {name: 1 for name in component_names})
        
    def is_calibrated(self) -> bool:
        for component_bus in self.component_bus_by_port.values():
            if not component_bus.is_calibrated():
                return False
        return True

    def read_calibration(self) -> dict[str, Any]:
        calibration = {}
        for component_bus in self.component_bus_by_port.values():
            if not isinstance(component_bus, OpenCVCamera):
                calibration.update(component_bus.read_calibration())
        return calibration


    def setup_component(self, component_name: str, component_port: str, configs: Dict[str, Any]):
        component_bus = self.component_bus_by_port[component_port]
        component_bus.setup_component(component_name, configs)

    