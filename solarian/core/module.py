import json
import os
import sys
import time
import inspect
import threading
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Union
import traceback as _tb
from multiprocessing import Process, Event, Queue as MpQueue, get_start_method
from solarian.core.calibration import MotorCalibration

from solarian.core.joint import Joint
from solarian.core.actuator import Actuator
from solarian.core.sensor import Sensor
from solarian.core.servo import Servo
from solarian.core.data import Data, PosData, DataComputer, TimestampData
from solarian.core.clock_sync import synced_time, sync_with_server, get_offset
from solarian.core.module_bus import ModuleBus
from solarian.core.action import Action, ActionType, ModuleAction, action_handler
from solarian.core.configs import InterpolatorType, ModuleConfig
from solarian.core.observation import Observation, ModuleObs
from solarian.utils import manual_profile, profile, setup_process_logging, get_log_queue
from solarian.utils.profiling import get_stats_snapshot, merge_stats as merge_profiling_stats
from solarian.utils.shared_memory.shared_memory_queue import Empty
from solarian.utils.shared_memory.action_queue import ModuleActionQueue
from solarian.utils.shared_memory.observation_ring_buffer import ObservationRingBuffer
from solarian.utils.precise_sleep import precise_wait
from solarian.utils.trajectory_interpolator import (
    JointTrajectoryInterpolator,
    PoseTrajectoryInterpolator,
    ScalarTrajectoryInterpolator,
    TrajectoryInterpolator,
)
from solarian.utils import get_class
from solarian.utils.status_board import StatusBoard
# TODO: remove speed from calibration required
# CALIBRATION_REQUIRED = [ValueName.POS]

from solarian.utils import profile, manual_profile

import logging
logger = logging.getLogger(f"module")


class Module(Process, ABC):
    """
    Base class for robot modules.
    
    Modules run as separate processes and communicate via shared memory if the start() method is called.
    Otherwise, they run in the same process as the caller, eg: to just read sensors or write actuators without time sensitive operations.
    They support an extensible action system where subclasses can register
    custom action handlers using the @action_handler decorator.
    
    Subclasses might typically:

    - extend the initialize method if further initialization is needed
    - define a `config_class` pointing to a ModuleConfig subclass for module-specific config fields
    
    
    
    Standard actions (STOP, SCHEDULE_WAYPOINT) are handled by default.
    :class:`ArmModule` and :class:`GripperModule` override waypoint handling
    and add ``SET_HOME``. Subclasses can override these or add new actions.
    """

    # Override in subclasses to use a custom config class (must extend ModuleConfig)
    config_class: type = ModuleConfig

    field_name = "component"

    allowed_attr_type = Sensor | Actuator

    # def __init_subclass__(cls, **kwargs):
    #     super().__init_subclass__(**kwargs)

    #     for name, value in cls.__dict__.items():
    #         if name.startswith("_") or inspect.isroutine(value):
    #             continue

    #         if not isinstance(value, cls.allowed_attr_type):
    #             raise TypeError(
    #                 f"{cls.__name__}.{name} must be of type {cls.allowed_attr_type}, "
    #                 f"got {type(value)}"
    #             )
      
    @classmethod
    def _component_names(cls, component_type):
        # Walk the MRO (not just cls.__dict__) so components declared on a base
        # class -- e.g. Cartesian actuators inherited from a base arm module --
        # are included, matching how __init__ builds the module's
        # actuators/sensors.
        names = []
        seen = set()
        for klass in cls.__mro__:
            for name in klass.__dict__:
                if name in seen:
                    continue
                if isinstance(getattr(cls, name, None), component_type):
                    names.append(name)
                    seen.add(name)
        return names

    @classmethod
    def actuator_names(cls):
        return cls._component_names(Actuator)
    
    @classmethod
    def sensor_names(cls):
        return cls._component_names(Sensor)

    @classmethod
    def validate_obs(cls, data: dict):
        for sensor_name, sensor_data in data.items():
            if sensor_name == 'timestamp':
                continue
            if sensor_name not in cls.sensor_names():
                raise ValueError(f"Invalid sensor name: '{sensor_name}' for module '{cls.__name__}'")
            sensor = getattr(cls, sensor_name)
            sensor.validate_obs(sensor_data)

    @classmethod
    def validate_act(cls, data: dict):
        for actuator_name, actuator_data in data.items():
            if actuator_name == 'target_time':
                continue
            if actuator_name not in cls.actuator_names():
                raise ValueError(f"Invalid actuator name: '{actuator_name}' for module '{cls.__name__}'")
            actuator = getattr(cls, actuator_name)
            actuator.validate_act(actuator_data)

    @classmethod
    def __class_getitem__(cls, item):
        return getattr(cls, item)

    def __getitem__(self, item):
        return getattr(self, item)

    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner):
        object.__setattr__(self, 'parent', owner)
        return self    


    def __init__(self, cfg: ModuleConfig | None = None, name: str | None = None, observation_buffer: ObservationRingBuffer | None = None, action_queue: ModuleActionQueue | None = None):
        super().__init__()

        if cfg is None:
            return

        self.cfg = cfg
        self.name = name if name is not None else self.__class__.__name__
        
        self.observation_buffer = observation_buffer
        self.action_queue = action_queue

        self._register_action_handlers()

        self.verbose = True
        self.is_connected = False
        self.calibration = None

        # Interpolator runtime (pose / scalar / joint). Sensor-only modules
        # keep interpolator_type JOINT and never seed ``interp``.
        self.interp: TrajectoryInterpolator | None = None
        self.last_waypoint_time: float = 0.0
        self.speed_kwargs: dict[str, Any] = {}
        self._late_waypoint_warn_t: float = 0.0
        self._late_waypoint_suppressed: int = 0
        self._last_action_time: float | None = None
        self._action_interval_ema: float = float(
            getattr(self.cfg, "drive_horizon_min_s", 1.0 / 30.0)
        )

        self.ready_event = Event()
        self._error_queue: MpQueue = MpQueue()
        self._profiling_queue: MpQueue = MpQueue()

        self.launch_timeout = self.cfg.launch_timeout
        self.commands_per_cycle = self.cfg.commands_per_cycle
        self.frequency = self.cfg.frequency

        # create actuators and sensors from class-level attributes
        self.actuators = {}
        self.sensors = {}
        self.actuators_by_port = {}
        class_attrs = {}
        for cls in reversed(type(self).__mro__):
            for name, value in cls.__dict__.items():
                if isinstance(value, (Actuator, Sensor)):
                    class_attrs[name] = value
        for name, value in class_attrs.items():
            if isinstance(value, Actuator):
                # create actuator
                actuator_class = value.__class__
                self.actuators[name] = actuator_class(self.cfg.actuators[name])
                setattr(self, name, self.actuators[name])
                
                # group actuators by port
                if self.cfg.actuators[name].port not in self.actuators_by_port:
                    self.actuators_by_port[self.cfg.actuators[name].port] = []
                self.actuators_by_port[self.cfg.actuators[name].port].append(name)
                
                # if component is a servo, it is also a sensor
                if isinstance(value, Sensor):
                    self.sensors[name] = self.actuators[name]
            elif isinstance(value, Sensor):
                # create sensor
                sensor_class = value.__class__
                self.sensors[name] = sensor_class(self.cfg.sensors[name])
                setattr(self, name, self.sensors[name])

        self._log_queue = get_log_queue()

        self.module_bus = ModuleBus()
        self.module_bus.add_components(self.actuators)
        self.module_bus.add_components(self.sensors)
        # Lets subclasses layer extra behavior (e.g. Cartesian IK on ArmModule)
        # over a port's joint bus.
        self.module_bus.port_bus_wrapper = self.wrap_port_bus

        module_obs_spec = Observation.get_module_obs_spec()
        if self.name in module_obs_spec:
            self._module_obs_cls = module_obs_spec[self.name].__class__
            self._robot_processors = self._module_obs_cls.create_processors(DataComputer.ROBOT)
        else:
            self._module_obs_cls = None
            self._robot_processors = {}


        
    def get_calibration(self) -> dict | None:
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
            
    # def sensors(self) -> tuple[str, Sensor]:
    #     """Get all sensors in the module."""
    #     return self.sensors

    # def actuators(self) -> Dict[str, Actuator]:
    #     """Get all actuators in the module."""
    #     return self.actuators.items()

    def get_actuator_models(self) -> List[str]:
        """Get all actuator models in the module."""
        return [actuator.cfg.model for actuator in self.actuators.values()]

    def get_actuator_names(self) -> List[str]:
        """Get all actuator names in the module."""
        return list(self.actuators.keys())

    def read_sensors(self, data_type: type, component_names: list[str] | None = None, normalize: bool = True) -> Dict[str, Any]:
        """Read calibrated values from the actuators."""
        
        if component_names is None:
            component_names = list(self.sensors.keys())

        if self.module_bus is None:
            return {}

        values_dict = self.module_bus.read(data_type, component_names, normalize)
        values = values_dict.values()
        return {name: {'value': value, 'timestamp': time.time()} for name, value in zip(component_names, values)}

    def connect_sensors(self):
        for sensor in self.sensors.values():
            sensor.connect()

    def connect(self, calibration: dict | None = None):
        """Connect this module's hardware bus."""
        if self.is_connected:
            return
        self.module_bus.connect(calibration or {})
        self.is_connected = True

    def disconnect(self):
        """Disconnect this module's hardware bus."""
        if not self.is_connected:
            return
        self.module_bus.disconnect()
        self.is_connected = False
    # def read_actuators_np(self, data_type: type, component_names: list[str] | None = None) -> np.ndarray:
    #     """Read raw values from the actuators."""
    #     if self.module_bus is None:
    #         return np.array([])
    #     return np.array(list(self.module_bus.read(data_type, component_names).values()))
    
    def write_actuators(self, values: Dict[str, Data], normalize: bool = True):
        """Write calibrated values to the actuators."""
        self.module_bus.write(values, normalize)

    def configure_components(self, params: dict[str, int], component_names: list[str] | None = None):
        """Write raw hardware-level configuration parameters to components.

        Args:
            params: Mapping of parameter names to values (hardware-specific).
            component_names: Components to configure. Defaults to all components.
        """
        if component_names is None:
            component_names = list(self.actuators.keys()) + [
                name for name in self.sensors.keys() if name not in self.actuators
            ]
        self.module_bus.configure(params, component_names)

    def enable_torque(self, component_names: list[str] | None = None):
        """Enable torque on the specified components (actuators by default)."""
        if component_names is None:
            component_names = list(self.actuators.keys())
        self.module_bus.enable_torque(component_names)

    def disable_torque(self, component_names: list[str] | None = None):
        """Disable torque on the specified components (actuators by default)."""
        if component_names is None:
            component_names = list(self.actuators.keys())
        self.module_bus.disable_torque(component_names)

    def is_calibrated(self) -> bool:
        if self.calibration is None:
            return False
        return self.calibration == self.read_calibration()

    def read_calibration(self) -> dict[str, MotorCalibration]:
        if self.cfg.calibration_file is None:
            logger.error(f"No calibration required for module {self.name}")
            return {}

        """Read calibration parameters from the actuators."""

        data = self.module_bus.read_calibration()
        calibration = {}
        for actuator_name, actuator in self.actuators.items():
            # Virtual actuators (e.g. ik_cartesian x/y/z/rx/ry/rz) are not
            # hardware and have no calibration to read back; the bus only
            # reports the real motors, so skip anything it doesn't return.
            if actuator_name not in data:
                continue
            calibration[actuator_name] = data[actuator_name]
        return calibration

    def write_calibration(self) -> None:
        if self.cfg.calibration_file is None:
            logger.error(f"No calibration required for module {self.name}")
            return

        logger.info(f"\nWriting calibration of: {self.name}")

        self.start_calibration_mode()

        self.load_calibration()

        if self.calibration is None:
            # raise ValueError(f"No calibration found for module {self.name}")
            logger.error(f"No calibration found for module {self.name}")
            

        for port, actuators in self.actuators_by_port.items():
            port_calibration = {}
            for actuator_name in actuators:
                # Skip virtual actuators (e.g. ik_cartesian) that have no
                # hardware calibration entry.
                if actuator_name not in self.calibration:
                    continue
                port_calibration[actuator_name] = self.calibration[actuator_name]
            self.module_bus.component_bus_by_port[port].write_calibration(port_calibration)
        logger.info(f"Calibration written to: {self.name}")
        self.end_calibration_mode()
    
    def _save_calibration(self, calibration: dict):
        if self.cfg.calibration_file is None:
            logger.error(f"No calibration required for module {self.name}")
            return
        
        if not os.path.exists(os.path.dirname(self.cfg.calibration_file)):
            os.makedirs(os.path.dirname(self.cfg.calibration_file))
        with open(self.cfg.calibration_file, "w") as f:
            data = {actuator_name: asdict(motor_calib) for actuator_name, motor_calib in calibration.items()}
            json.dump(data, f, indent=4)
            logger.info(f"Calibration saved for module {self.name}")

    def load_calibration(self):
        if self.cfg.calibration_file is None:
            logger.error(f"No calibration required for module {self.name}")
            self.calibration = {}
            return
        
        if not os.path.exists(self.cfg.calibration_file):
            logger.error(f"Calibration file not found for module {self.name}")
            self.calibration = None
            return

        with open(self.cfg.calibration_file, "r") as f:
            data = json.load(f)
            self.calibration = {actuator_name: MotorCalibration(**calibration) for actuator_name, calibration in data.items()}
            logger.info(f"Calibration loaded for module {self.name}")
        
    def calibrate(self) -> None:
        """Calibrate this module. Actuator modules override with a range sweep."""
        pass

    def start_calibration_mode(self):
        print(f"Starting calibration mode for {self.name}")
        for port, actuator_names in self.actuators_by_port.items():
            self.module_bus.component_bus_by_port[port].start_calibration_mode()

    def end_calibration_mode(self):
        print(f"Ending calibration mode for {self.name}")
        for port, actuator_names in self.actuators_by_port.items():
            self.module_bus.component_bus_by_port[port].end_calibration_mode()


    # ========= inverse kinematics ============
    def wrap_port_bus(self, port, inner_bus, joint_components, cartesian_components):
        """Optionally wrap a port's joint bus (e.g. Cartesian IK on ArmModule).

        The default returns ``inner_bus`` unchanged. :class:`ArmModule` overrides
        this to layer inverse kinematics when Cartesian actuators are present.
        """
        return inner_bus

    def initialize(self):
        pass

        
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
        self.module_bus.setup_component(component.name, component_port=component.cfg.port, configs=component.cfg.configs)
        
        
    def __str__(self):
        return f"{self.name} module"

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            logger.info(f"[Module] {self.name} process spawned at {self.pid}")

    def stop(self, wait=True):
        """Stop the module process."""
        # self.process_action(Action.stop())
        self._dispatch_action(ModuleAction(ActionType.STOP), time.time())
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        if not self.is_alive():
            error = self.get_error()
            if error:
                mod_name, tb_str = error
                raise RuntimeError(
                    f"Module '{mod_name}' process died during startup:\n{tb_str}"
                )
            raise RuntimeError(
                f"Module '{self.name}' process died during startup "
                f"(exit code {self.exitcode}). Check stderr for details."
            )
        if not self.ready_event.is_set():
            # Alive but never reached the first control-loop iteration — usually
            # a post-fork deadlock (ZMQ / StatusBoard threads in the parent).
            raise RuntimeError(
                f"Module '{self.name}' process (pid {self.pid}) did not become "
                f"ready within launch_timeout={self.launch_timeout}s. The child "
                f"is likely stuck after fork; restart and check that the parent "
                f"is not multithreaded before module.start()."
            )
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get_error(self) -> tuple[str, str] | None:
        """Return (module_name, traceback_string) if the process crashed, else None.
        
        Safe to call from the parent process only.
        """
        try:
            return self._error_queue.get_nowait()
        except Exception:
            return None

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


    # ========= receive APIs =============
    def get_observation(self, k: int | None = None) -> Union[ModuleObs, list, None]:
        """
        Get observation(s) from the observation buffer.
        
        Args:
            k: If None, return the next observation (FIFO).
               If specified, return up to k observations.
               
        Returns:
            ModuleObs if k is None, otherwise list of ModuleObs (oldest first).
            Returns None if the buffer is empty.
        """
        
        if self.observation_buffer is None or self.observation_buffer.count == 0:
            return None
        elif k is None:
            return self.observation_buffer.get()
        else:
            return self.observation_buffer.get_last_k(k=k)
    
    def get_all_observations(self) -> list[ModuleObs]:
        """
        Get all available observations from the observation buffer.
        
        Returns:
            List of ModuleObs instances (newest first)
        """
        if self.observation_buffer is None or self.observation_buffer.count == 0:
            return []
        return self.observation_buffer.get_last_k(k=self.observation_buffer.count)


    def enrich_observation(self, obs: ModuleObs) -> ModuleObs:
        for field_name, proc in self._robot_processors.items():
            obs[field_name] = proc(obs)
        return obs


    def update_observations(self):

        if self.observation_buffer is None:
            raise ValueError("Observation buffer is not initialized.")

        if self._module_obs_cls is None:
            return

        sensor_fields = self._module_obs_cls.get_sensor_fields_spec()

        data = {}
        fields_by_data_class = {}

        for field_name, field in sensor_fields.items():
            if field.data_class not in fields_by_data_class:
                fields_by_data_class[field.data_class] = []
            fields_by_data_class[field.data_class].append((field_name, field.sensor_name, field.data_name))

        for data_class, field_list in fields_by_data_class.items():
            sensor_names = [sensor_name for _, sensor_name, _ in field_list]
            sensor_data = self.read_sensors(data_class, sensor_names)
            for field_name, sensor_name, _ in field_list:
                sensor_data[sensor_name]['timestamp'] = synced_time() - self.cfg.receive_latency
                data[field_name] = sensor_data[sensor_name]

        obs = self._module_obs_cls(data)
        obs = self.enrich_observation(obs)

        self.observation_buffer.put(obs)

    def reset(self):
        """Reset the module. Eg: set to home position"""
        pass

    def _coalesce_actions(self, actions: list) -> list:
        """Keep only the latest SCHEDULE_WAYPOINT in a drained burst.

        Backlogged teleop waypoints with stale stamps cause interpolator thrash
        (and used to be discarded). Other action types keep their order.
        """
        if len(actions) <= 1:
            return actions
        last_sw = None
        for i, action in enumerate(actions):
            if int(action.action_type) == int(ActionType.SCHEDULE_WAYPOINT):
                last_sw = i
        if last_sw is None:
            return actions
        return [
            action
            for i, action in enumerate(actions)
            if int(action.action_type) != int(ActionType.SCHEDULE_WAYPOINT) or i == last_sw
        ]

    def _register_action_handlers(self) -> None:
        """Scan the MRO for ``@action_handler`` methods (inherited included)."""
        self.action_handlers: dict[int, str] = {}
        for cls in reversed(type(self).__mro__):
            for attr_name, value in cls.__dict__.items():
                if callable(value) and hasattr(value, "_handles_action"):
                    self.action_handlers[int(getattr(value, "_handles_action"))] = attr_name

    # ========= interpolator helpers ============
    def _is_pose_interpolator(self) -> bool:
        return self.cfg.interpolator_type == InterpolatorType.POSE

    def _is_scalar_interpolator(self) -> bool:
        return self.cfg.interpolator_type == InterpolatorType.SCALAR

    def _is_joint_interpolator(self) -> bool:
        return self.cfg.interpolator_type == InterpolatorType.JOINT

    def _warn_late_waypoint(
        self, target_time: float, t_now: float, *, clamped: bool = False
    ) -> None:
        late_ms = (t_now - target_time) * 1000.0
        now_wall = time.time()
        if now_wall - self._late_waypoint_warn_t < 1.0:
            self._late_waypoint_suppressed += 1
            return
        suppressed = self._late_waypoint_suppressed
        self._late_waypoint_warn_t = now_wall
        self._late_waypoint_suppressed = 0
        extra = f" ({suppressed} more suppressed)" if suppressed else ""
        action = "clamping" if clamped else "discarding"
        logger.warning(
            "Module '%s': %s late SCHEDULE_WAYPOINT (%.1f ms late)%s",
            self.name,
            action,
            late_ms,
            extra,
        )

    def _resolve_waypoint_time(self, target_time: float, t_now: float) -> float:
        if target_time > t_now:
            return float(target_time)
        self._warn_late_waypoint(target_time, t_now, clamped=True)
        horizon = 1.0 / max(float(self.frequency), 1.0)
        return float(t_now + horizon)

    def _adaptive_drive_horizon(self, t_now: float) -> float:
        min_h = float(getattr(self.cfg, "drive_horizon_min_s", 1.0 / 30.0))
        max_h = float(getattr(self.cfg, "drive_horizon_max_s", 0.12))
        alpha = float(getattr(self.cfg, "drive_horizon_ema_alpha", 0.2))
        mult = float(getattr(self.cfg, "drive_horizon_multiplier", 1.5))
        if self._last_action_time is not None:
            dt = t_now - self._last_action_time
            if dt > 0:
                self._action_interval_ema = (
                    alpha * dt + (1.0 - alpha) * self._action_interval_ema
                )
        self._last_action_time = t_now
        horizon = min(max(self._action_interval_ema * mult, min_h), max_h)
        StatusBoard.set(
            f"{self.name} drive horizon (ms)",
            round(horizon * 1000.0, 1),
        )
        return horizon

    def _drop_timestamps(self, action: ModuleAction) -> bool:
        flag = getattr(action, "drop_timestamps", None)
        if flag is not None:
            return bool(flag)
        return bool(getattr(self.cfg, "drive_waypoints", False))

    def _create_interpolator(
        self, times: np.ndarray, values: np.ndarray
    ) -> TrajectoryInterpolator:
        interp_type = self.cfg.interpolator_type
        if interp_type == InterpolatorType.SCALAR:
            return ScalarTrajectoryInterpolator(times, values)
        if interp_type == InterpolatorType.JOINT:
            return JointTrajectoryInterpolator(times, values)
        if interp_type == InterpolatorType.POSE:
            return PoseTrajectoryInterpolator(times, values)
        raise ValueError(f"Unknown interpolator type: {interp_type}")

    def _get_max_speed_kwargs(self) -> dict:
        interp_type = self.cfg.interpolator_type
        max_speed = self.cfg.max_speed
        if interp_type == InterpolatorType.SCALAR:
            return {
                "max_speed": max_speed if isinstance(max_speed, (int, float)) else max_speed[0]
            }
        if interp_type == InterpolatorType.JOINT:
            return {"max_joint_speed": max_speed}
        if interp_type == InterpolatorType.POSE:
            if isinstance(max_speed, (int, float)):
                return {"max_pos_speed": max_speed, "max_rot_speed": max_speed}
            return {
                "max_pos_speed": max_speed[0],
                "max_rot_speed": max_speed[1] if len(max_speed) > 1 else max_speed[0],
            }
        raise ValueError(f"Unknown interpolator type: {interp_type}")

    def _scalar_actuator_name(self) -> str | None:
        explicit = getattr(type(self), "scalar_actuator", None)
        if explicit:
            return str(explicit)
        from solarian.core.ik.actuator import CartesianIKActuator

        names = [
            n
            for n, act in self.actuators.items()
            if not isinstance(act, CartesianIKActuator)
        ]
        return names[0] if names else None

    def _scalar_to_actuator(self, value: float) -> float:
        return float(value)

    def _actuator_to_scalar(self, value: float) -> float:
        return float(value)

    def _read_current_scalar(self) -> float:
        name = self._scalar_actuator_name()
        if not name:
            return 0.0
        readings = self.read_sensors(PosData, [name])
        entry = readings[name]
        raw = float(entry["value"] if isinstance(entry, dict) else entry.value)
        return self._actuator_to_scalar(raw)

    def _ensure_scalar_interpolator(self, t_now: float | None = None) -> None:
        if self.interp is not None:
            return
        t = float(t_now if t_now is not None else time.monotonic())
        try:
            value = self._read_current_scalar()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Module '%s': could not seed scalar interpolator (%s); using 0.",
                self.name,
                exc,
            )
            value = 0.0
        self.interp = self._create_interpolator(
            np.array([t]), np.array([value], dtype=np.float64)
        )
        self.last_waypoint_time = t

    def _extract_scalar_waypoint(
        self, action: ModuleAction
    ) -> tuple[float, float] | None:
        name = self._scalar_actuator_name()
        if not name:
            return None
        fields = action.values()
        if name not in fields:
            if len(fields) == 1:
                field = next(iter(fields.values()))
                return float(field.value), float(field.timestamp)
            logger.warning(
                "Module '%s': SCHEDULE_WAYPOINT missing scalar field %r; ignoring.",
                self.name,
                name,
            )
            return None
        field = fields[name]
        return float(field.value), float(field.timestamp)

    def _schedule_scalar_waypoint(
        self,
        value: float,
        target_time_wall: float,
        t_now: float,
        drop_timestamps: bool = False,
    ) -> None:
        self._ensure_scalar_interpolator(t_now)
        assert self.interp is not None
        if drop_timestamps:
            target_time = t_now + self._adaptive_drive_horizon(t_now)
            self.interp = self.interp.drive_to_waypoint(
                value=float(value),
                time=target_time,
                curr_time=t_now,
                **self._get_max_speed_kwargs(),
            )
            self.last_waypoint_time = target_time
            return

        target_time = time.monotonic() - synced_time() + float(target_time_wall)
        target_time = self._resolve_waypoint_time(target_time, t_now)
        self.interp = self.interp.schedule_waypoint(
            value=float(value),
            time=target_time,
            curr_time=t_now,
            last_waypoint_time=self.last_waypoint_time,
            **self._get_max_speed_kwargs(),
        )
        self.last_waypoint_time = target_time

    def _write_scalar(self, value: float) -> None:
        name = self._scalar_actuator_name()
        if not name:
            return
        self.write_actuators({name: PosData(value=self._scalar_to_actuator(value))})

    # ========= action dispatch system ============
    def _dispatch_action(self, action: ModuleAction, t_now: float) -> bool:
        """
        Dispatch an action to its registered handler.
        
        Args:
            action: The ModuleAction object from the queue
            t_now: Current monotonic time
            
        Returns:
            bool: True to continue running, False to stop
        """
        handler_name = self.action_handlers.get(int(action.action_type))
        
        if handler_name:
            handler = getattr(self, handler_name)
            return handler(action, t_now)
        else:
            # Unknown action - log warning but continue
            if self.verbose:
                print(f"[Module] {self.name}: Unknown action type {action.action_type}, ignoring")
            return True

    # ========= observation thread ============
    def _observation_loop(self, stop_event: threading.Event):
        """Continuously read sensors and publish observations in a dedicated thread.

        Runs as fast as the underlying I/O allows so that the main action
        loop is never blocked by slow serial-bus reads.
        """
        while not stop_event.is_set():
            try:
                self.update_observations()
            except Exception as e:
                if self.verbose:
                    print(f"[Module] {self.name}: observation error: {e}")
                break

    @action_handler(ActionType.SCHEDULE_WAYPOINT)
    def _handle_schedule_waypoint(self, action: ModuleAction, t_now: float) -> bool:
        """Schedule a timestamped waypoint into the module interpolator.

        * ``interpolator_type: pose`` — Cartesian ``x..rz`` (ArmModule).
        * ``interpolator_type: scalar`` — single actuator (grippers).
        Other interpolator types ignore the waypoint.
        """
        drop_timestamps = self._drop_timestamps(action)
        if self._is_pose_interpolator():
            extract = getattr(self, "_extract_pose_waypoint", None)
            schedule = getattr(self, "_schedule_pose_waypoint", None)
            if extract is None or schedule is None:
                return True
            extracted = extract(action)
            if extracted is None:
                return True
            pose, target_time_wall = extracted
            schedule(pose, target_time_wall, t_now, drop_timestamps=drop_timestamps)
            return True

        if self._is_scalar_interpolator():
            extracted = self._extract_scalar_waypoint(action)
            if extracted is None:
                return True
            value, target_time_wall = extracted
            self._schedule_scalar_waypoint(
                value, target_time_wall, t_now, drop_timestamps=drop_timestamps
            )
            return True

        return True

    def control_step(self):
        """Sample pose/scalar interpolators when configured.

        Subclasses may override; call ``super().control_step()`` to keep
        upsampling, or replace entirely (ArmModule joint motion).
        """
        if self._is_pose_interpolator():
            ensure = getattr(self, "_ensure_pose_interpolator", None)
            write = getattr(self, "_write_pose", None)
            if ensure is not None and write is not None:
                ensure()
                if self.interp is not None:
                    write(self.interp(time.monotonic()))
            return
        if self._is_scalar_interpolator():
            self._ensure_scalar_interpolator()
            if self.interp is not None:
                self._write_scalar(float(self.interp(time.monotonic())))

    # ========= main loop in process ============
    def run(self):
        """Main control loop running in a separate process."""

        status_mp_q = getattr(self, '_status_mp_q', None)
        if status_mp_q is not None:
            StatusBoard.init_child(status_mp_q)

        logger.info(f"Module {self.name} process start method: {get_start_method()}")
        setup_process_logging(logger_name=None, log_queue=self._log_queue, install_fallback=False)

        logger.info(f"Starting module {self.name} process")
        self._register_action_handlers()

        # Reconnect hardware in the subprocess — device handles inherited
        # from the parent (via fork) are not reliably usable (e.g. AVFoundation cameras).
        # Keep per-port calibration so normalized reads work immediately.
        calibration = self.get_calibration() or {}
        self.disconnect()
        logger.info(f"Module {self.name} process disconnected")
        self.connect(calibration)

        logger.info(f"Module {self.name} process connected")

        endpoint = getattr(self, "_clock_sync_endpoint", None)
        if endpoint is not None:
            try:
                sync_with_server(endpoint[0], int(endpoint[1]))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Module '%s': clock sync failed (%s); using local wall "
                    "clock (offset=%.1f ms)",
                    self.name,
                    exc,
                    get_offset() * 1000.0,
                )

        try:
            self.reset()


            self._t_start = time.monotonic()
            self._iter_idx = 0
            
            keep_running = True

            # obs_thread.start()


            t_last_control_step = time.monotonic()


            while keep_running:
                t_now = time.monotonic()

                # Drain actions first so control_step() sees the latest target
                action = None
                try:
                    if self.action_queue is None:
                        logger.error("Action queue is not initialized.")
                    elif not self.action_queue.empty():
                        actions = self._coalesce_actions(self.action_queue.get_all())
                        for action in actions:
                            keep_running = self._dispatch_action(action, t_now)

                except Empty:
                    pass

                manual_profile("control_step", time.monotonic() - t_last_control_step)
                t_last_control_step = time.monotonic()
                self.control_step()

                self.update_observations()

                
                # first loop successful, ready to receive actions
                if self._iter_idx == 0:
                    self.ready_event.set()
                self._iter_idx += 1
                
                # regulate frequency
                dt = 1 / self.frequency
                t_end = self._t_start + dt * self._iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)
                
        except Exception as _run_exc:
            tb_str = "".join(
                _tb.format_exception(type(_run_exc), _run_exc, _run_exc.__traceback__)
            )
            logger.exception(
                "Module '%s' crashed: %s", self.name, _run_exc
            )
            self._error_queue.put_nowait((self.name, tb_str))
        finally:
            self.ready_event.set()
            try:
                self._profiling_queue.put_nowait(get_stats_snapshot())
            except Exception:
                pass
            if self.verbose:
                print(f"[Module] {self.name} process terminated")

    