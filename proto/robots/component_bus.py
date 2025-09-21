from typing import Any, Dict
from enum import Enum
import time

from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
# from lerobot.common.motors.dynamixel import (
#     DriveMode,
#     DynamixelMotorsBus,
#     OperatingMode,
# )
# from lerobot.common.motors.feetech import (
#     FeetechMotorsBus,
#     OperatingMode,
# )


from proto.robots.opencv_camera import OpenCVCamera
from proto.robots.ur5_bus import UR5Bus
from proto.robots.component import Component
from proto.common.configs import ValueName, ComponentType
from proto.common.utils import manual_profile
from proto.robots.dynamixel_bus import DynamixelBus
from proto.robots.feetech_bus import FeetechBus


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

    def connect(self, calibration: Dict[str, MotorCalibration]):
        self.component_bus_by_port = {}
        for port, components in self.components_by_port.items():
            # get the first component type
            first_component = next(iter(components.values()))
            component_type = first_component.cfg.type
            
            if component_type == ComponentType.FEETECH:
                # self.component_bus_by_port[port] = FeetechMotorsBus(
                #     port=port,
                #     motors={
                #         name: Motor(component.cfg.id, component.cfg.model, map_motor_mode(component.cfg.mode)) for name, component in components.items()
                #     },
                #     calibration=calibration.get(port, None)
                # )
                self.component_bus_by_port[port] = FeetechBus(
                    port=port,
                    components=components,
                    calibration=calibration.get(port, None)
                )
                
            elif component_type == ComponentType.DYNAMIXEL:
                # self.component_bus_by_port[port] = DynamixelMotorsBus(
                #     port=port,
                #     motors={
                #         name: Motor(component.cfg.id, component.cfg.model, map_motor_mode(component.cfg.mode)) for name, component in components.items()
                #     },
                #     calibration=calibration.get(port, None)
                # )
                self.component_bus_by_port[port] = DynamixelBus(
                    port=port,
                    components=components,
                    calibration=calibration.get(port, None)
                )
            elif component_type == ComponentType.UR5_JOINT or component_type == ComponentType.UR5_CARTESIAN:
                self.component_bus_by_port[port] = UR5Bus(
                    port=port,
                    components=components,
                    calibration=calibration.get(port, None),
                    joint_names2ids={name: component.cfg.id for name, component in components.items() if component.cfg.type == ComponentType.UR5_JOINT},
                    cartesian_names2ids={name: component.cfg.id for name, component in components.items() if component.cfg.type == ComponentType.UR5_CARTESIAN}
                )
            elif component_type == ComponentType.OPENCV:
                # cameras don't share ports
                self.component_bus_by_port[port] = first_component
            else:
                raise ValueError(f"Unknown component bus type: {component_type}")

            self.component_bus_by_port[port].connect()

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
                # if isinstance(component_bus, FeetechMotorsBus) or isinstance(component_bus, DynamixelMotorsBus):
                if isinstance(component_bus, FeetechBus) or isinstance(component_bus, DynamixelBus):
                    # component_bus.sync_write(map_write(data_name, component_bus), values=values_bus, normalize=normalize)
                    component_bus.write(data_name, values_bus, normalize=normalize)
                elif isinstance(component_bus, UR5Bus):
                    component_bus.write(data_name, values_bus, normalize=False)
                else:
                    raise ValueError(f"Unknown component bus type: {component_bus.__class__}")
    
    def read(self, data_name: str, component_names: list[str], normalize: bool = True) -> Dict[str, Any]:
        """Read values from components."""
        result = {}
        for port, component_bus in self.component_bus_by_port.items():
            # group component reads by port
            names_bus = [name for name in component_names if name in self.components_by_port[port]]
            if names_bus:
                # if isinstance(component_bus, FeetechMotorsBus) or isinstance(component_bus, DynamixelMotorsBus):
                if isinstance(component_bus, FeetechBus) or isinstance(component_bus, DynamixelBus):
                    # data_name = map_read(data_name, component_bus)
                    # data = component_bus.read(data_name, motor_names=names_bus)
                    # data = component_bus.sync_read(data_name, motors=names_bus, normalize=normalize).values()
                    data = component_bus.read(data_name, names_bus, normalize=normalize)
                elif isinstance(component_bus, UR5Bus):
                    # data_name = map_read(data_name, component_bus)
                    data = component_bus.read(data_name, names_bus, normalize=False)
                elif isinstance(component_bus, OpenCVCamera):
                    # data_name = map_read(data_name, component_bus)
                    data = {names_bus[0]: component_bus.read(data_name)}
                else:
                    raise ValueError(f"Unknown component bus type: {component_bus.__class__}")
                result.update(data)
        return result

    def write_pos(self, values: Dict[str, Any]):
        """Write a value to the actuator group."""
        self.write(ValueName.POS, values)

    def write_speed(self, values: Dict[str, Any]):
        """Write a value to the actuator group."""
        self.write(ValueName.SPEED, values)

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

    