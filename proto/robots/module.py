import json
import os
import time
from dataclasses import dataclass, asdict
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
from proto.common.configs import ValueName, DataComputer
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
        self.sensors: Dict[str, Sensor] = {}
        
        # add actuators and sensors from joints
        for joint in self.joints.values():
            self.actuators.update(joint.get_actuators())
            self.component_bus.add_components(joint.get_actuators())

            self.sensors.update(joint.get_sensors())
            self.component_bus.add_components(joint.get_sensors())

        # add actuators and sensors from bodies
        for body in self.bodies.values():
            self.actuators.update(body.get_actuators())
            self.component_bus.add_components(body.get_actuators())

            self.sensors.update(body.get_sensors())
            self.component_bus.add_components(body.get_sensors())

        # group actuators by port
        self.actuators_by_port = {}
        for actuator_name, actuator in self.actuators.items():
            if actuator.cfg.port not in self.actuators_by_port:
                self.actuators_by_port[actuator.cfg.port] = []
            self.actuators_by_port[actuator.cfg.port].append(actuator_name)

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

    def is_calibrated(self) -> bool:
        if self.calibration is None:
            return False
        return self.calibration == self.read_calibration()

    def read_calibration(self) -> dict[str, MotorCalibration]:
        if self.cfg.calibration_file is None:
            print(f"No calibration required for module {self.name}")
            return {}

        """Read calibration parameters from the actuators."""
        # actuator_names = list(self.actuators.keys())
        # offsets = self.component_bus.read("Homing_Offset", actuator_names, normalize=False)
        # mins = self.component_bus.read("Min_Position_Limit", actuator_names, normalize=False)
        # maxes = self.component_bus.read("Max_Position_Limit", actuator_names, normalize=False)
        # drive_modes = self.component_bus.read("Drive_Mode", actuator_names, normalize=False)
        
        # calibration = {}
        # for actuator_name in actuator_names:
        #     calibration[actuator_name] = MotorCalibration(
        #         id=self.actuators[actuator_name].cfg.id,
        #         drive_mode=drive_modes[actuator_name],
        #         homing_offset=offsets[actuator_name],
        #         range_min=mins[actuator_name],
        #         range_max=maxes[actuator_name],
        #     )

        data = self.component_bus.read_calibration()
        calibration = {}
        for actuator_name, actuator in self.actuators.items():
            calibration[actuator_name] = data[actuator_name]
        return calibration

    def write_calibration(self) -> None:
        if self.cfg.calibration_file is None:
            print(f"No calibration required for module {self.name}")
            return

        print(f"\nWriting calibration of: {self.name}")
        # # group actuators by port
        # actuators_by_port = {}
        # for actuator_name, actuator in self.actuators.items():
        #     if actuator.cfg.port not in actuators_by_port:
        #         actuators_by_port[actuator.cfg.port] = []
        #     actuators_by_port[actuator.cfg.port].append(actuator_name)
        
        # print("Disabling torque...")
        # self.component_bus.disable_torque(self.actuators.keys())

        # print("Setting position mode...")
        # self.component_bus.set_position_mode(self.actuators.keys())

        self.start_calibration_mode()

        self._load_calibration()

        for port, actuators in self.actuators_by_port.items():
            port_calibration = {}
            for actuator_name in actuators:
                port_calibration[actuator_name] = self.calibration[actuator_name]
            self.component_bus.component_bus_by_port[port].write_calibration(port_calibration)
        print("Calibration written to:", self.name)
        self.end_calibration_mode()

    
    def get_observation(self) -> Observation:
        """Create an observation with current module state."""

        point_0 = time.time_ns()
        
        # group sensors by data name, to optimize reading in batches
        sensors_by_data_name = {}
        for sensor_name, sensor_spec in settings.OBSERVATION[self.name].items():
            for data_name, data_spec in sensor_spec.items():
                if data_spec.computer == DataComputer.ROBOT:
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
        if self.cfg.calibration_file is None:
            print(f"No calibration required for module {self.name}")
            return
        
        if not os.path.exists(os.path.dirname(self.cfg.calibration_file)):
            os.makedirs(os.path.dirname(self.cfg.calibration_file))
        with open(self.cfg.calibration_file, "w") as f:
            data = {actuator_name: asdict(motor_calib) for actuator_name, motor_calib in calibration.items()}
            json.dump(data, f, indent=4)
            print(f"Calibration saved for module {self.name}")

    def _load_calibration(self):
        if self.cfg.calibration_file is None:
            print(f"No calibration required for module {self.name}")
            self.calibration = {}
            return
        
        if not os.path.exists(self.cfg.calibration_file):
            print(f"Calibration file not found for module {self.name}")
            self.calibration = None
            return

        with open(self.cfg.calibration_file, "r") as f:
            data = json.load(f)
            self.calibration = {actuator_name: MotorCalibration(**calibration) for actuator_name, calibration in data.items()}
            print(f"Calibration loaded for module {self.name}")
        
    def calibrate(self) -> None:
        if self.cfg.calibration_file is None:
            print(f"No calibration required for module {self.name}")
            return

        print(f"\nRunning calibration of {self}")

        self.start_calibration_mode()

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = {}
        for port, actuator_names in self.actuators_by_port.items():
            homing_offsets.update(self.component_bus.component_bus_by_port[port].set_half_turn_homings(actuator_names))

        print(
            f"Move all joints sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )

        range_mins = {}
        range_maxes = {}
       
        # get ranges of motion for each actuator
        for port, actuator_names in self.actuators_by_port.items():
            range_mins, range_maxes = self.component_bus.component_bus_by_port[port].record_ranges_of_motion(actuator_names)
            range_mins.update(range_mins)
            range_maxes.update(range_maxes)
            

        self.calibration = {}
        for port, actuator_names in self.actuators_by_port.items():
            port_calibration = {}
            for actuator_name in actuator_names:
                port_calibration[actuator_name] = MotorCalibration(
                    id=self.actuators[actuator_name].cfg.id,
                    drive_mode=0,
                    homing_offset=homing_offsets[actuator_name],
                    range_min=range_mins[actuator_name],
                    range_max=range_maxes[actuator_name],
                )
            self.component_bus.component_bus_by_port[port].write_calibration(port_calibration)
            self.calibration.update(port_calibration)

        self._save_calibration(self.calibration)
        print("Calibration saved to", self.cfg.calibration_file)
        self.end_calibration_mode()

    def start_calibration_mode(self):
        for port, actuator_names in self.actuators_by_port.items():
            self.component_bus.component_bus_by_port[port].start_calibration_mode()

    def end_calibration_mode(self):
        for port, actuator_names in self.actuators_by_port.items():
            self.component_bus.component_bus_by_port[port].end_calibration_mode()

    
    def setup(self):
        """Setup module"""
        if self.calibration is None:
            self.calibrate()
        if not self.is_calibrated():
            self.write_calibration()
        


    def __str__(self):
        return f"{self.name} ({self.cfg.type})"