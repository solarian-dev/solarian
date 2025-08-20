import json
import os
import time
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Any
from lerobot.common.motors import Motor, MotorCalibration

from proto.robots.joint import Joint
from proto.robots.body import Body
from proto.robots.actuator import Actuator
from proto.robots.sensor import Sensor
from proto.robots.component_bus import ComponentBus
from proto.common.configs import ModuleConfig
from proto.robots.observation import Observation
from proto.common.configs import ValueName
from proto.conf import settings
from proto.common.utils import manual_profile, profile


# TODO: remove speed from calibration required
CALIBRATION_REQUIRED = [ValueName.POS]

class Module(ABC):
    def __init__(self, cfg: ModuleConfig, name: str, component_bus: ComponentBus):
        self.name = name
        self.cfg = cfg

        self.component_bus = component_bus

        self.bodies: Dict[str, Body] = {}
        self.joints: Dict[str, Joint] = {}

        for body_name, body_cfg in cfg.bodies.items():
            body = Body(body_name, body_cfg)
            self.bodies[body_name] = body

        for joint_name, joint_cfg in cfg.joints.items():
            joint = Joint(joint_name, joint_cfg, self.bodies[joint_cfg.parent_body], self.bodies[joint_cfg.child_body])
            self.joints[joint_name] = joint
    
        self.actuators: Dict[str, Actuator] = {}
        for joint in self.joints.values():
            self.actuators.update(joint.get_actuators())
            self.component_bus.add_components(joint.get_actuators())

        self.sensors: Dict[str, Sensor] = {}
        for body in self.bodies.values():
            self.sensors.update(body.get_sensors())
            self.component_bus.add_components(body.get_sensors())

        for joint in self.joints.values():
            self.sensors.update(joint.get_sensors())
            self.component_bus.add_components(joint.get_sensors())

        self._load_calibration()
        
            
    def get_calibration(self) -> dict:
        "returns calibrations grouped by port"
        calibration = {}
        for actuator_name in self.calibration.keys():
            port = self.actuators[actuator_name].cfg.port
            if port not in calibration:
                calibration[port] = {}
            calibration[port][actuator_name] = self.calibration[actuator_name]
        return calibration
            
    def get_sensors(self) -> Dict[str, Sensor]:
        """Get all sensors in the module."""
        return self.sensors

    def get_actuators(self) -> Dict[str, Actuator]:
        """Get all actuators in the module."""
        return self.actuators

    def get_actuator_models(self) -> List[str]:
        """Get all actuator models in the module."""
        return [actuator.cfg.model for actuator in self.actuators.values()]

    def get_actuator_names(self) -> List[str]:
        """Get all actuator names in the module."""
        return list(self.actuators.keys())

    def read_sensors(self, data_name: str, component_names: list[str] | None = None, normalize: bool = True) -> Dict[str, Any]:
        """Read calibrated values from the actuators."""
        if component_names is None:
            component_names = list(self.sensors.keys())

        values_dict = self.component_bus.read(data_name, component_names, normalize)
        values = values_dict.values()

        # Perform calibration transformations if required
        # if data_name in CALIBRATION_REQUIRED and self.calibrator is not None:
        #     values = np.array(list(values))
        #     values = self.calibrator.avoid_rotation_reset(values, component_names, data_name)
        #     values = self.calibrator.apply_calibration_autocorrect(values, component_names).tolist()

        return {name: {data_name: value} for name, value in zip(component_names, values)}

    def connect_sensors(self):
        for sensor in self.sensors.values():
            sensor.connect()

    def read_actuators_np(self, data_name: str, component_names: list[str] | None = None) -> np.ndarray:
        """Read raw values from the actuators."""
        return np.array(list(self.component_bus.read(data_name, component_names).values()))
    
    def write_actuators(self, data_name: str, data: int | float | Dict[str, Any], normalize: bool = True):
        """Write calibrated values to the actuators."""
        if isinstance(data, (int, float)):
            values = np.array([data] * len(self.actuators))
            component_names = list(self.actuators.keys())
        else:
            values = np.array(list(data.values()))
            component_names = list(data.keys())

        # Perform reverse calibration transformations if required
        # if data_name in CALIBRATION_REQUIRED and self.calibrator is not None:
        #     values = self.calibrator.revert_calibration(values, component_names)

        values_dict = {name: value for name, value in zip(component_names, values)}
        self.component_bus.write(data_name, values_dict, normalize)

    def write_actuators_pos(self, values: int | float | Dict[str, Any], normalize: bool = True):
        """Write positions to actuators."""
        self.write_actuators(ValueName.POS, values, normalize)        
    
    def write_actuators_speed(self, values: int | float | Dict[str, Any], normalize: bool = True):
        """Write speeds to actuators."""
        self.write_actuators(ValueName.SPEED, values, normalize)
    
    def enable_actuators_torque(self, actuator_names: list[str] | None = None):
        """Enable torque for the actuator."""
        if actuator_names is None:
            actuator_names = list(self.actuators.keys())
        self.component_bus.enable_torque(actuator_names)

    def disable_actuators_torque(self, actuator_names: list[str] | None = None):
        """Disable torque for the actuator."""
        if actuator_names is None:
            actuator_names = list(self.actuators.keys())
        self.component_bus.disable_torque(actuator_names)

    def set_actuators_velocity_mode(self, actuator_names: list[str] | None = None):
        """Set actuators to velocity mode."""
        if actuator_names is None:
            actuator_names = list(self.actuators.keys())
        self.component_bus.set_velocity_mode(actuator_names)

    def set_actuators_position_mode(self, actuator_names: list[str] | None = None):
        """Set actuators to position mode."""
        if actuator_names is None:
            actuator_names = list(self.actuators.keys())
        self.component_bus.set_position_mode(actuator_names)
        

    @abstractmethod
    def calibrate(self):
        """
        Calibrates the bus. Attempts to load an existing calibration file;
        if not found, runs manual calibration and saves the result.
        """

    def is_calibrated(self):
        return self.component_bus.is_calibrated()

    @abstractmethod
    def setup(self):
        """Setup module"""
        
    def write_calibration(self) -> None:
        print(f"\nWriting calibration of: {self.name}")
        # group actuators by port
        actuators_by_port = {}
        for actuator_name, actuator in self.actuators.items():
            if actuator.cfg.port not in actuators_by_port:
                actuators_by_port[actuator.cfg.port] = []
            actuators_by_port[actuator.cfg.port].append(actuator_name)
        
        print("Disabling torque...")
        self.component_bus.disable_torque(self.actuators.keys())

        print("Setting position mode...")
        self.component_bus.set_position_mode(self.actuators.keys())

        self._load_calibration()

        for port, actuators in actuators_by_port.items():
            port_calibration = {}
            for actuator_name in actuators:
                port_calibration[actuator_name] = self.calibration[actuator_name]
            self.component_bus.component_bus_by_port[port].write_calibration(port_calibration)
        print("Calibration written to:", self.name)

    
    def get_observation(self) -> Observation:
        """Create an observation with current module state."""

        point_0 = time.time_ns()
        
        # group sensors by data name, to optimize reading in batches
        sensors_by_data_name = {}
        for sensor_name, (data_name, data_shape) in settings.OBSERVATION[self.name].items():
            if data_name not in sensors_by_data_name:
                sensors_by_data_name[data_name] = []
            sensors_by_data_name[data_name].append(sensor_name)

        point_1 = time.time_ns()

        # read sensors in batches
        data = {}
        for data_name, sensor_names in sensors_by_data_name.items():
            # print("Reading sensors:", sensor_names)
            data.update(self.read_sensors(data_name, sensor_names))

        point_2 = time.time_ns()

        manual_profile("get observation 1", (point_1 - point_0))
        manual_profile("get observation 2", (point_2 - point_1))
        
        return Observation(data={self.name: data})

    def _save_calibration(self, calibration: dict):
        with open(self.cfg.calibration_file, "w") as f:
            json.dump(calibration, f, indent=4)
            print(f"Calibration saved for module {self.name}")

    def _load_calibration(self):
        if not os.path.exists(self.cfg.calibration_file):
            print(f"Calibration file not found for module {self.name}")
            self.calibration = {}
            return

        with open(self.cfg.calibration_file, "r") as f:
            self.calibration = json.load(f)
            self.calibration = {actuator_name: MotorCalibration(**calibration) for actuator_name, calibration in self.calibration.items()}
            print(f"Calibration loaded for module {self.name}")
        