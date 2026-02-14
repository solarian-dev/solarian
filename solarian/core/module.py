import json
import os
import time
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Union
from multiprocessing import Process, Event
from solarian.core.calibration import MotorCalibration

from solarian.core.joint import Joint
from solarian.core.actuator import Actuator
from solarian.core.sensor import Sensor
from solarian.core.servo import Servo
from solarian.core.component_bus import ComponentBus
from solarian.core.action import Action, ActionType, action_handler
from solarian.core.configs import ModuleConfig, InterpolatorType
from solarian.core.observation import ModuleObservation
from solarian.core.configs import ValueName, DataComputer
from solarian.utils import manual_profile, profile
from solarian.utils.shared_memory.shared_memory_queue import Empty
from solarian.utils.shared_memory.action_queue import ActionQueue
from solarian.utils.shared_memory.observation_ring_buffer import ObservationRingBuffer
from solarian.utils.trajectory_interpolator import (
    TrajectoryInterpolator,
    ScalarTrajectoryInterpolator,
    JointTrajectoryInterpolator,
    PoseTrajectoryInterpolator,
)
from solarian.utils.precise_sleep import precise_wait

# TODO: remove speed from calibration required
CALIBRATION_REQUIRED = [ValueName.POS]


class Module(Process):
    """
    Base class for robot modules.
    
    Modules run as separate processes and communicate via shared memory if the start() method is called.
    Otherwise, they run in the same process as the caller, eg: to just read sensors or write actuators without time sensitive operations.
    They support an extensible action system where subclasses can register
    custom action handlers using the @action_handler decorator.
    
    Subclasses might typically:

    - extend the initialize method if further initialization is needed
    - define a `config_class` pointing to a ModuleConfig subclass for module-specific config fields
    
    
    
    Standard actions (STOP, SCHEDULE_WAYPOINT, RESTART_PUT) are handled
    by default. Subclasses can override these or add new actions.
    """

    # Override in subclasses to use a custom config class (must extend ModuleConfig)
    config_class: type = ModuleConfig

    def __init_subclass__(cls, **kwargs):
        """Collect action handlers from the class hierarchy."""
        super().__init_subclass__(**kwargs)
        # Build action handler registry for this class
        cls._action_handlers = {}
        
        # Collect handlers from all parent classes and this class
        for klass in cls.__mro__:
            for name, method in vars(klass).items():
                if callable(method) and hasattr(method, '_handles_action'):
                    action_type = method._handles_action
                    # Don't override if already registered by a subclass
                    if action_type not in cls._action_handlers:
                        cls._action_handlers[action_type] = name
      
    def __init__(self, cfg: ModuleConfig):
        super().__init__()

        self.cfg = cfg
        self.component_bus = None
        self.shm_manager = None
        self.observation_buffer = None
        
        self.action_handlers: dict[int, str] = {}

        # self.bodies: Dict[str, Body] = {}
        # self.joints: Dict[str, Joint] = {}
        self.verbose = True
        
        # Runtime state for the control loop
        self.interp: TrajectoryInterpolator | None = None
        self.last_waypoint_time: float = 0.0
        self.speed_kwargs: dict[str, Any] = {}

        # for body_name, body_cfg in cfg.bodies.items():
        #     body = Body(body_name, body_cfg)
        #     self.bodies[body_name] = body

        # for joint_name, joint_cfg in cfg.joints.items():
        #     joint = Joint(joint_name, joint_cfg, self.bodies[joint_cfg.parent_body], self.bodies[joint_cfg.child_body])
        #     self.joints[joint_name] = joint
    
        # self.actuators: Dict[str, Actuator] = {}
        # self.sensors: Dict[str, Sensor] = {}
        
        
        self.ready_event = Event()
        self.launch_timeout = cfg.launch_timeout
        self.commands_per_cycle = cfg.commands_per_cycle
        self.frequency = cfg.frequency

    
    # stores its name when created as a class attribute, eg in a Robot class
    def __set_name__(self, owner, name):
        self.name = name
    
    
    def _get_fields_of_type(self, typ: type):
        result = {}
        for name, field in self.model_fields.items():
            if field.annotation is typ:
                result[name] = getattr(self, name)
        return result

    @property
    def servos(self) -> Dict[str, Servo]:
        return self._get_fields_of_type(Servo)

    @property
    def joints(self) -> Dict[str, Joint]:
        return self._get_fields_of_type(Joint)

    @property
    def actuators(self) -> Dict[str, Actuator]:
        return self._get_fields_of_type(Actuator)

    @property
    def sensors(self) -> Dict[str, Sensor]:
        return self._get_fields_of_type(Sensor)
        
    def get_calibration(self) -> dict:
        "returns calibrations grouped by port"
        if self.calibration is None:
            return self.calibration
            
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

        self.start_calibration_mode()

        self._load_calibration()

        for port, actuators in self.actuators_by_port.items():
            port_calibration = {}
            for actuator_name in actuators:
                port_calibration[actuator_name] = self.calibration[actuator_name]
            self.component_bus.component_bus_by_port[port].write_calibration(port_calibration)
        print("Calibration written to:", self.name)
        self.end_calibration_mode()

    
    

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
            port_calibration = self.component_bus.component_bus_by_port[port].write_calibration(port_calibration)
            self.calibration.update(port_calibration)

        self._save_calibration(self.calibration)
        print("Calibration saved to", self.cfg.calibration_file)
        self.write_calibration()
        print("Calibration written to actuators")
        self.end_calibration_mode()

    def start_calibration_mode(self):
        for port, actuator_names in self.actuators_by_port.items():
            self.component_bus.component_bus_by_port[port].start_calibration_mode()

    def end_calibration_mode(self):
        for port, actuator_names in self.actuators_by_port.items():
            self.component_bus.component_bus_by_port[port].end_calibration_mode()

    
    def initialize(self, component_bus: ComponentBus, observation_buffer: ObservationRingBuffer, action_queue: ActionQueue):
        """Initialize module"""

        self.component_bus = component_bus
        self.observation_buffer = observation_buffer
        self.action_queue = action_queue

        # add actuators and sensors from joints
        for actuator in self.actuators.values():
            self.component_bus.add_components(actuator)

        # add actuators and sensors from bodies
        for sensor in self.sensors.values():
            self.component_bus.add_components(sensor)

        # group actuators by port
        self.actuators_by_port = {}
        for actuator_name, actuator in self.actuators.items():
            if actuator.port not in self.actuators_by_port:
                self.actuators_by_port[actuator.port] = []
            self.actuators_by_port[actuator.port].append(actuator_name)

        self._load_calibration()

        if self.calibration is None:
            self.calibrate()
        
        if not self.is_calibrated():
            self.write_calibration()
        
    def setup(self):

        # List all available components that need setup
        print("\n=== Available Components ===")
        idx = 0
        components = []

        # List all actuators and sensors that need setup
        for actuator_name, actuator in self.actuators.items():
            if actuator.needs_setup() and not actuator in components:
                idx += 1
                print(f"{idx}. {actuator_name} ({actuator.cfg.type} - {actuator.cfg.model})")
                components.append(actuator)
        
        for sensor_name, sensor in self.sensors.items():
            if sensor.needs_setup() and not sensor in components:
                idx += 1
                print(f"{idx}. {sensor_name} ({sensor.cfg.type} - {sensor.cfg.model})")
                components.append(sensor)
        
        # Ask user to select a component
        print("\n=== Select Component to Setup ===")
        while True:
            user_input = input(f"Enter component name or number (1-{len(components)}): ").strip()
            
            # Check if input is a number
            if user_input.isdigit():
                idx_cfg = int(user_input)
                if 1 <= idx_cfg <= len(components):
                    break
                else:
                    print(f"Invalid number. Please enter a number between 1 and {len(components)}")
            else:
                print(f"Invalid number. Please enter a number between 1 and {len(components)}")
        
        component = components[idx_cfg - 1]
        print(f"\nSetting up component: {component}")
        self.component_bus.setup_component(component.name, component_port=component.cfg.port, configs=component.cfg.configs)
        
        
    def __str__(self):
        return f"{self.name} ({self.cfg.type})"

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[Module] {self.name} process spawned at {self.pid}")

    def stop(self, wait=True):
        """Stop the module process."""
        self.process_action(Action.stop())
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


    # ========= receive APIs =============
    def get_observation(self, k: int = None) -> Union[ModuleObservation, list]:
        """
        Get observation(s) from the ring buffer.
        
        Args:
            k: If None, return the most recent observation. 
               If specified, return the last k observations.
               
        Returns:
            ModuleObservation if k is None, otherwise list of ModuleObservations (oldest first)
        """
        if k is None:
            return self.observation_buffer.get()
        else:
            return self.observation_buffer.get_last_k(k=k)
    
    def get_all_observations(self) -> list:
        """
        Get all available observations from the ring buffer (up to get_max_k).
        
        Returns:
            List of ModuleObservation instances (oldest first)
        """
        return self.observation_buffer.get_all()


    def update_observations(self):
        """Update observations with current module state."""
        from solarian.conf import settings

        point_0 = time.time_ns()
        
        # group sensors by data name, to optimize reading in batches
        sensors_by_data_name = {}
        for sensor_name, sensor_spec in settings.PROJECT.observation_class[self.name].items():
            for data_name, data_spec in sensor_spec.items():
                if data_spec.computer == DataComputer.ROBOT:
                    if data_name not in sensors_by_data_name:
                        sensors_by_data_name[data_name] = []
                    sensors_by_data_name[data_name].append(sensor_name)

        point_1 = time.time_ns()

        # read sensors in batches
        data = {}
        for data_name, sensor_names in sensors_by_data_name.items():
            data.update(self.read_sensors(data_name, sensor_names))

        point_2 = time.time_ns()
        manual_profile("get observation 1", (point_1 - point_0))
        manual_profile("get observation 2", (point_2 - point_1))

        # add timestamp
        data[settings.SensorName.CLOCK] = {ValueName.TIMESTAMP: time.time() - self.cfg.receive_latency}

        observation = ModuleObservation(data={self.name: data})
        self.observation_buffer.put(observation)


    def reset(self):
        """Reset the module. Eg: set to home position"""
        pass
        

    # ========= interpolator factory ============
    def _create_interpolator(self, times: np.ndarray, values: np.ndarray) -> TrajectoryInterpolator:
        """
        Factory method to create the appropriate interpolator based on module config.
        
        Override this method in subclasses for custom interpolation behavior.
        
        Args:
            times: Array of waypoint times
            values: Array of waypoint values (shape depends on interpolator type)
            
        Returns:
            A TrajectoryInterpolator instance
        """
        interp_type = self.cfg.interpolator_type
        
        if interp_type == InterpolatorType.SCALAR:
            return ScalarTrajectoryInterpolator(times, values)
        elif interp_type == InterpolatorType.JOINT:
            return JointTrajectoryInterpolator(times, values)
        elif interp_type == InterpolatorType.POSE:
            return PoseTrajectoryInterpolator(times, values)
        else:
            raise ValueError(f"Unknown interpolator type: {interp_type}")
    
    def _get_max_speed_kwargs(self) -> dict:
        """
        Get the appropriate max speed kwargs for schedule_waypoint based on interpolator type.
        
        Returns:
            Dict with the appropriate speed limit kwargs for the interpolator
        """
        interp_type = self.cfg.interpolator_type
        max_speed = self.cfg.max_speed
        
        if interp_type == InterpolatorType.SCALAR:
            return {'max_speed': max_speed if isinstance(max_speed, (int, float)) else max_speed[0]}
        elif interp_type == InterpolatorType.JOINT:
            return {'max_joint_speed': max_speed}
        elif interp_type == InterpolatorType.POSE:
            # For pose, we need both position and rotation speed
            if isinstance(max_speed, (int, float)):
                return {'max_pos_speed': max_speed, 'max_rot_speed': max_speed}
            else:
                # Assume first element is pos speed, second is rot speed
                return {'max_pos_speed': max_speed[0], 'max_rot_speed': max_speed[1] if len(max_speed) > 1 else max_speed[0]}
        else:
            raise ValueError(f"Unknown interpolator type: {interp_type}")

    
    def get_initial_action_values(self) -> np.ndarray:
        """
        Get the initial action values for the interpolator.
        
        Override in subclass to return the appropriate initial values
        based on the module's current state.
        
        Returns:
            np.ndarray: Initial values (scalar, joint array, or 6D pose)
        """
        return np.array([0.0] * len(self.actuators))

    # def process_action(self, action: Action):
    #     """
    #     Utility method to write action data to actuators.
        
    #     Subclasses can use this in their exec_waypoint implementation
    #     after converting interpolated values to an Action.
        
    #     Args:
    #         action: Action with actuator data to write
    #     """
    #     if self.name not in action.data:
    #         return
    #     for actuator_name, actuator_data in action.data[self.name].items():
    #         for data_name, data_value in actuator_data.items():
    #             if data_name == ValueName.POS:
    #                 self.write_actuators_pos({actuator_name: data_value})
    #             elif data_name == ValueName.SPEED:
    #                 self.write_actuators_speed({actuator_name: data_value})
    #             elif data_name == ValueName.TORQUE_ENABLE:
    #                 self.write_actuators_torque_enable({actuator_name: data_value})

    # def schedule_waypoint(self, target_values: np.ndarray, target_time: float):
    #     """Schedule a waypoint action to be executed at a target time."""
    #     action = Action.schedule_waypoint(target_values, target_time)
    #     self.process_action(action)

    def process_action(self, action: Action):
        """Send an action to the module's input queue."""
        self.action_queue.put(action)
    
    def _create_example_action(self) -> Action:
        """
        Create an example Action to define the input queue schema.
        
        Override in subclass if you need custom data fields in actions.
        The example action should have the same structure as actions
        that will be sent to this module.
        
        Returns:
            Example Action with all fields that will be used
        """
        from solarian.conf import settings

        # Default example with target_values for waypoint scheduling
        # Subclasses should override this to match their actual data shape

        example_data = {}
        for actuator_name, actuator_spec in settings.PROJECT.action_class[self.name].items():
            for data_name, data_spec in actuator_spec.items():
                if data_name in [ValueName.POS, ValueName.SPEED, ValueName.TORQUE_ENABLE]:
                    example_data[data_name] = 0.0
                else:
                    raise ValueError(f"Unknown data type: {data_name}")

        return Action(
            action_type=ActionType.SCHEDULE_WAYPOINT,
            target_time=0.0,
            data=example_data
        )

    

    # ========= action dispatch system ============
    def _dispatch_action(self, action: Action, t_now: float) -> bool:
        """
        Dispatch an action to its registered handler.
        
        Args:
            action: The Action object from the queue
            t_now: Current monotonic time
            
        Returns:
            bool: True to continue running, False to stop
        """
        handler_name = self._action_handlers.get(action.action_type)
        
        if handler_name:
            handler = getattr(self, handler_name)
            return handler(action, t_now)
        else:
            # Unknown action - log warning but continue
            if self.verbose:
                print(f"[Module] {self.name}: Unknown action type {action.action_type}, ignoring")
            return True

    # ========= standard action handlers ============
    @action_handler(ActionType.STOP)
    def _handle_stop(self, action: Action, t_now: float) -> bool:
        """Handle STOP action - stops the module."""
        if self.verbose:
            print(f"[Module] {self.name}: Received STOP action")
        return False  # Signal to stop
    
    @action_handler(ActionType.SCHEDULE_WAYPOINT)
    def _handle_schedule_waypoint(self, action: Action, t_now: float) -> bool:
        """Handle SCHEDULE_WAYPOINT action - schedules a new waypoint."""
        target_values = self._extract_target_values(action)
        target_time = action.target_time
        # translate global time to monotonic time
        target_time = time.monotonic() - time.time() + target_time
        
        self._interp = self._interp.schedule_waypoint(
            value=target_values,
            time=target_time,
            curr_time=t_now,
            last_waypoint_time=self._last_waypoint_time,
            **self._speed_kwargs
        )
        self._last_waypoint_time = target_time
        return True
    
    @action_handler(ActionType.RESTART_PUT)
    def _handle_restart_put(self, action: Action, t_now: float) -> bool:
        """Handle RESTART_PUT action - restarts the put timing."""
        self._t_start = action.target_time - time.time() + time.monotonic()
        self._iter_idx = 1
        return True

    # ========= main loop in process ============
    def run(self):
        """Main control loop running in a separate process."""
        try:
            self.reset()

            # get initial values based on interpolator type
            curr_values = self.get_initial_action_values()
            curr_t = time.monotonic()
            
            # Initialize runtime state
            self._interp = self._create_interpolator(
                times=[curr_t],
                values=[curr_values]
            )
            self._last_waypoint_time = curr_t
            self._speed_kwargs = self._get_max_speed_kwargs()
            self._t_start = time.monotonic()
            self._iter_idx = 0
            
            keep_running = True

            while keep_running:
                t_now = time.monotonic()

                # Execute current waypoint from interpolator
                self.exec_waypoint(t_now)
                
                # update observations
                self.update_observations()

                # fetch actions from queue
                try:
                    actions = self.action_queue.get_k(self.commands_per_cycle)
                except Empty:
                    actions = []
                
                # dispatch actions via handler registry
                for action in actions:
                    keep_running = self._dispatch_action(action, t_now)
                    if not keep_running:
                        break
                    
                # first loop successful, ready to receive actions
                if self._iter_idx == 0:
                    self.ready_event.set()
                self._iter_idx += 1
                
                # regulate frequency
                dt = 1 / self.frequency
                t_end = self._t_start + dt * self._iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)
                
        finally:
            self.ready_event.set()
            if self.verbose:
                print(f"[Module] {self.name} process terminated")

    # # ========= abstract methods for subclasses ============
    # @abstractmethod
    # def exec_waypoint(self, timestamp: float):
    #     """
    #     Execute the current interpolated waypoint values.
        
    #     This method is called on every iteration of the control loop.
    #     Subclasses must implement this to convert interpolated values
    #     to actuator commands and execute them.
        
    #     Args:
    #         values: Interpolated values from the trajectory interpolator
    #         timestamp: Current timestamp (monotonic time)
    #     """
    
    # @abstractmethod
    # def _extract_target_values(self, action: Action) -> np.ndarray:
    #     """
    #     Extract target values from an action for scheduling waypoints.
        
    #     Override in subclass to customize how actions are parsed.
        
    #     Args:
    #         action: Action object from input queue
            
    #     Returns:
    #         Target values array for the interpolator
    #     """
