import json
import os
import sys
import time
import inspect
import threading
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Union
from multiprocessing import get_start_method
from solarian.core.calibration import MotorCalibration

from solarian.core.joint import Joint
from solarian.core.actuator import Actuator, merge_actuator_config
from solarian.core.sensor import Sensor
from solarian.core.servo import Servo
from solarian.core.data import Data, TimestampData
from solarian.core.clock_sync import synced_time, sync_with_server, get_offset
from solarian.core.module_bus import ModuleBus
from solarian.core.action import Action, ActionType, ModuleAction, action_handler
from solarian.core.configs import ModuleConfig
from solarian.core.observation import Observation, ModuleObs
from solarian.core.event import Event
from solarian.core.execution import ExecutionMode, ExecutionPlace, WorkerExit
from solarian.core.runner import Runner, make_runner
from solarian.utils import manual_profile, profile, setup_process_logging, get_log_queue
from solarian.utils.shared_memory.shared_memory_queue import Empty
from solarian.utils.shared_memory.action_queue import ModuleActionQueue
from solarian.utils.shared_memory.observation_buffer import ObservationBuffer
from solarian.utils import get_class
from solarian.utils.status_board import StatusBoard
# TODO: remove speed from calibration required
# CALIBRATION_REQUIRED = [ValueName.POS]

from solarian.utils import profile, manual_profile

import logging
logger = logging.getLogger(f"module")


class Module(ABC):
    """
    Base class for robot modules.

    A module is a *logical* device: it produces observations, may emit events,
    and consumes actions. How it is scheduled is a separate, operational
    decision expressed by :attr:`execution_mode` and carried out by a
    :class:`~solarian.core.runner.Runner` — so the same class can run in its
    own process on a robot and inline inside a test.

    Calling :meth:`start` hands the module to its runner. Without it the module
    is simply usable in-process, e.g. to read sensors or write actuators
    without time-sensitive operations.

    They support an extensible action system where subclasses can register
    custom action handlers using the @action_handler decorator.
    
    Subclasses might typically:

    - extend the initialize method if further initialization is needed
    - define a `config_class` pointing to a ModuleConfig subclass for module-specific config fields
    
    
    
    STOP is handled by default. Motion types (``MOVE_*``, ``SERVO_*``,
    ``SCHEDULE_*``) live on :class:`~solarian.core.arm_module.ArmModule` and
    :class:`~solarian.core.gripper_module.GripperModule`. Subclasses can
    override these or add new actions.
    """

    #: What this module *is*, for a consumer of a recorded session that knows
    #: nothing about the class. Declared on the class rather than inferred from
    #: the MRO so a plugin naming a new kind needs no change in the core.
    kind: str = "module"

    # Override in subclasses to use a custom config class (must extend ModuleConfig)
    config_class: type = ModuleConfig

    #: Config to use when the project ships no ``conf/<name>.yaml``. Set this
    #: on modules that have nothing to configure (input devices), so a project
    #: does not have to carry an empty YAML file for them. ``None`` keeps the
    #: YAML mandatory.
    default_config: dict | None = None

    #: How this module is scheduled by default. Hardware with its own timing
    #: requirements wants a process; cheap event sources can run inline.
    #: Overridable per instance and via ``execution:`` in the module YAML.
    execution_mode: ExecutionMode = ExecutionMode.PROCESS

    #: Event types this module can emit, for discovery and documentation.
    #: Declaring an event type here does not by itself emit anything; the
    #: module calls :meth:`emit` when the thing actually happens.
    events: tuple = ()

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


    def __init__(
        self,
        cfg: ModuleConfig | None = None,
        name: str | None = None,
        observation_buffer: ObservationBuffer | None = None,
        action_queue: ModuleActionQueue | None = None,
        execution_mode: ExecutionMode | str | None = None,
    ):
        # Class-level declarations (``right_arm = So106ArmModule()``) are
        # placeholders resolved later against the module YAML.
        self._runner: Runner | None = None

        if cfg is None:
            return

        self.cfg = cfg
        self.name = name if name is not None else self.__class__.__name__
        
        self.observation_buffer = observation_buffer
        self.action_queue = action_queue
        #: Archive encoders for this module's cameras, by sensor name. Set by
        #: the robot before the module starts, so the sink crosses the process
        #: boundary the same way the observation buffer does.
        self.archive_sinks: dict = {}

        # Explicit argument beats YAML, which beats the class default.
        resolved = execution_mode or getattr(cfg, "execution", None)
        if resolved is not None:
            self.execution_mode = ExecutionMode(resolved)

        self._register_action_handlers()

        self.verbose = True
        self.is_connected = False
        self.calibration = None

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

        from solarian.core.ik.actuator import CartesianIKActuator

        hardware_port = None
        for name, value in class_attrs.items():
            if not isinstance(value, Actuator) or isinstance(value, CartesianIKActuator):
                continue
            yaml_cfg = self.cfg.actuators.get(name)
            if yaml_cfg is not None and yaml_cfg.port:
                hardware_port = yaml_cfg.port
                break

        for name, value in class_attrs.items():
            if isinstance(value, Actuator):
                yaml_cfg = self.cfg.actuators.get(name)
                merged = merge_actuator_config(value, name, yaml_cfg, hardware_port)
                self.cfg.actuators[name] = merged
                self.actuators[name] = value.__class__(
                    merged,
                    model=value._declared_model,
                    is_continuous=value.is_continuous,
                    mode=value._declared_mode,
                    id=value._declared_id,
                )
                setattr(self, name, self.actuators[name])

                port = merged.port
                if port not in self.actuators_by_port:
                    self.actuators_by_port[port] = []
                self.actuators_by_port[port].append(name)

                if isinstance(value, Sensor):
                    self.sensors[name] = self.actuators[name]
            elif isinstance(value, Sensor):
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
            self._robot_processors = self._module_obs_cls.create_processors(ExecutionPlace.ROBOT)
        else:
            self._module_obs_cls = None
            self._robot_processors = {}

    def bind(self, cfg, name):
        """Rebuild this class-level placeholder against the module YAML."""
        return type(self)(cfg, name)

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

    def require_calibration(self) -> dict:
        """Port-grouped calibration, or raise if this module has none."""
        calibration = self.get_calibration()
        if calibration:
            return calibration
        raise RuntimeError(
            f"No calibration for module '{self.name}'. "
            f"Complete the sweep when prompted, or run `solarian calibrate`."
        )
            
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
        return {
            name: {"value": values_dict[name], "timestamp": time.time()}
            for name in component_names
            if name in values_dict
        }

    def connect_sensors(self):
        for sensor in self.sensors.values():
            sensor.connect()

    def connect(self, calibration: dict | None = None):
        """Connect this module's hardware bus."""
        if self.is_connected:
            return
        self.module_bus.archive_sinks = self.archive_sinks
        self.module_bus.capture_latency = self.cfg.receive_latency
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

    def needs_motor_calibration(self) -> bool:
        """True when this module has hardware servos that use MotorCalibration."""
        return any(
            getattr(act, "uses_motor_calibration", True)
            for act in self.actuators.values()
        )

    def _module_yaml_path(self) -> str | None:
        from solarian.core.module_yaml import module_yaml_path

        return module_yaml_path(self.name)

    def is_calibrated(self) -> bool:
        if self.calibration is None:
            return False
        return self.calibration == self.read_calibration()

    def read_calibration(self) -> dict[str, MotorCalibration]:
        if not self.needs_motor_calibration():
            return {}

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
        if not self.needs_motor_calibration():
            return

        logger.info(f"\nWriting calibration of: {self.name}")

        self.start_calibration_mode()

        self.load_calibration()

        if self.calibration is None:
            logger.error(f"No calibration found for module {self.name}")
            self.end_calibration_mode()
            return

        for port, actuators in self.actuators_by_port.items():
            port_calibration = {}
            for actuator_name in actuators:
                if actuator_name not in self.calibration:
                    continue
                port_calibration[actuator_name] = self.calibration[actuator_name]
            if port_calibration:
                self.module_bus.component_bus_by_port[port].write_calibration(
                    port_calibration
                )
        logger.info(f"Calibration written to: {self.name}")
        self.end_calibration_mode()

    def _apply_calibration_to_cfg(self, calibration: dict) -> None:
        for name, motor_calib in calibration.items():
            act = self.actuators.get(name)
            if act is None or act.cfg is None:
                continue
            act.cfg.real_range_min = int(motor_calib.range_min)
            act.cfg.real_range_max = int(motor_calib.range_max)
            act.cfg.center_offset = int(motor_calib.homing_offset)
            act.cfg.drive_mode = int(motor_calib.drive_mode)
            self.cfg.actuators[name] = act.cfg

    def _save_calibration(self, calibration: dict):
        self.calibration = calibration
        self._apply_calibration_to_cfg(calibration)
        yaml_path = self._module_yaml_path()
        if yaml_path is None:
            logger.info(f"Calibration kept in memory for module {self.name} (no yaml)")
            return
        from solarian.core.module_yaml import patch_actuator_fields

        updates = {
            name: {
                "real_range_min": int(mc.range_min),
                "real_range_max": int(mc.range_max),
                "center_offset": int(mc.homing_offset),
                "drive_mode": int(mc.drive_mode),
            }
            for name, mc in calibration.items()
        }
        patch_actuator_fields(yaml_path, updates)
        logger.info(f"Calibration saved to {yaml_path}")

    def _save_home(self, home: dict[str, float]) -> None:
        self.cfg.home = {name: float(value) for name, value in home.items()}
        yaml_path = self._module_yaml_path()
        if yaml_path is None:
            logger.info(f"Home kept in memory for module {self.name} (no yaml)")
            return
        from solarian.core.module_yaml import patch_top_level_mapping

        patch_top_level_mapping(yaml_path, "home", self.cfg.home)
        logger.info(f"Home saved to {yaml_path}")

    def _load_legacy_calibration_json(self) -> dict[str, MotorCalibration]:
        path = self.cfg.calibration_file
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            data = json.load(f)
        return {
            actuator_name: MotorCalibration(**entry)
            for actuator_name, entry in data.items()
        }

    def load_calibration(self):
        if not self.needs_motor_calibration():
            self.calibration = {}
            return

        calib: dict[str, MotorCalibration] = {}
        for name, act in self.actuators.items():
            if not getattr(act, "uses_motor_calibration", True) or act.cfg is None:
                continue
            cfg = act.cfg
            if cfg.real_range_min is None or cfg.real_range_max is None:
                continue
            calib[name] = MotorCalibration(
                id=int(cfg.id),
                drive_mode=int(cfg.drive_mode or 0),
                homing_offset=int(cfg.center_offset or 0),
                range_min=int(cfg.real_range_min),
                range_max=int(cfg.real_range_max),
            )

        hard_stop = [
            name
            for name, act in self.actuators.items()
            if getattr(act, "uses_motor_calibration", True)
            and act.cfg is not None
            and not act.is_continuous
        ]
        if any(name not in calib for name in hard_stop):
            for name, motor_calib in self._load_legacy_calibration_json().items():
                calib.setdefault(name, motor_calib)
        if any(name not in calib for name in hard_stop):
            self.calibration = None
            return
        self.calibration = calib if calib else None
        if self.calibration:
            self._apply_calibration_to_cfg(self.calibration)
            logger.info(f"Calibration loaded for module {self.name}")
        
    def calibrate(self) -> None:
        """Calibrate this module. Actuator modules override with a range sweep."""
        pass

    def start_calibration_mode(self):
        print(f"Starting calibration mode for {self.name}\n")
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
                print(
                    f"{idx}. {actuator_name} "
                    f"({actuator.__class__.__name__} - {actuator.cfg.model})"
                )
                components.append(actuator)
        
        for sensor_name, sensor in self.sensors.items():
            if sensor.needs_setup() and not sensor in components:
                idx += 1
                print(f"{idx}. {sensor_name} ({sensor.__class__.__name__})")
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
    @property
    def runner(self) -> Runner:
        """The scheduler for this module, created on first use."""
        if self._runner is None:
            self._runner = make_runner(self)
        return self._runner

    def start(self, wait=True):
        """Hand this module to its runner (process, thread, or inline host)."""
        self.runner.start(wait=wait)
        if self.verbose:
            logger.info(
                f"[Module] {self.name} started ({self.execution_mode.value})"
            )

    def stop(self, wait=True):
        """Ask the runner to stop the module."""
        self.runner.stop(wait=wait)

    def terminate(self):
        self.runner.terminate()

    def kill(self):
        self.runner.kill()

    def is_alive(self) -> bool:
        return self.runner.is_alive()

    def join(self, timeout: float | None = None):
        self.runner.join(timeout)

    @property
    def pid(self) -> int | None:
        """OS pid when this module runs in its own process, else ``None``."""
        return self.runner.pid

    @property
    def exitcode(self) -> int | None:
        return self.runner.exitcode

    @property
    def is_ready(self):
        return self.runner.is_ready

    def get_error(self) -> tuple[str, str] | None:
        """Return (module_name, traceback_string) if the module crashed, else None.

        Safe to call from the host process only.
        """
        return self.runner.get_error()

    def get_profiling_stats(self) -> dict | None:
        """Return the worker's profiling snapshot, if it published one."""
        return self.runner.get_profiling_stats()

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

    @action_handler(ActionType.START_SERVICE)
    def start_service(self, action: ModuleAction, t_now: float) -> bool:
        """Start the processor behind one of this module's calculated fields.

        ``params`` arrived off the wire, hence the ``.get``s.
        """
        field = str(action.params.get("field") or "")
        proc = self._robot_processors.get(field)
        if proc is None:
            logger.warning(
                "Module '%s': no robot-side processor for field '%s'", self.name, field
            )
            return True
        proc.start(dict(action.params.get("params") or {}))
        return True

    @action_handler(ActionType.STOP_SERVICE)
    def stop_service(self, action: ModuleAction, t_now: float) -> bool:
        proc = self._robot_processors.get(str(action.params.get("field") or ""))
        if proc is not None:
            proc.stop()
        return True


    def update_observations(self):
        """Publish this module's current state into its observation buffer.

        The one way a module reports state. Called by the runner on every
        tick, whichever execution mode that runner implements, so a skill
        reads the same buffer whether this module sits in its own process on
        the robot or inline on the controller's loop.

        The base handles modules that have an entry in the project's
        ``Observation`` schema, reading each declared sensor field. A module
        with no schema entry has nothing typed to publish and overrides this
        to report whatever it does have.
        """
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
                # A bus that reports its own capture time wins: an archived
                # frame and the observation of it have to name the same
                # instant, or the episode table points at the wrong frame.
                sensor_data[sensor_name]['timestamp'] = (
                    self.module_bus.capture_timestamps.get(
                        sensor_name, synced_time() - self.cfg.receive_latency
                    )
                )
                data[field_name] = sensor_data[sensor_name]

        obs = self._module_obs_cls(data)
        obs = self.enrich_observation(obs)

        self.observation_buffer.put(obs)

    # ========= events ============
    def emit(self, event_type, **data) -> None:
        """Queue an :class:`~solarian.core.event.Event` for the host to drain.

        Emitting never runs a handler: the event lands in the runner's sink
        and the host pulls it on its next control tick. That keeps handlers on
        the control thread even when the event originated on a device callback
        thread or in another process.
        """
        sink = self.runner.event_sink
        if sink is None:
            return
        event = Event(type=int(event_type), source=self.name, data=dict(data))
        try:
            sink.put_nowait(event) if hasattr(sink, "put_nowait") else sink.append(event)
        except Exception:  # noqa: BLE001 - a dropped event must not stall the loop
            logger.debug("Module '%s': dropped event %r", self.name, event)

    def drain_events(self) -> list[Event]:
        """Pop every event this module has emitted since the last drain.

        A module with no runner yet cannot have emitted anything, and building
        one just to read an empty sink would allocate queues for every module
        that never emits.
        """
        if self._runner is None:
            return []
        return self._runner.drain_events()

    def reset(self):
        """Reset the module. Eg: set to home position"""
        pass

    _DRIVE_ACTION_TYPES = frozenset({
        int(ActionType.MOVE_L),
        int(ActionType.MOVE_J),
        int(ActionType.SERVO_L),
        int(ActionType.SERVO_J),
        int(ActionType.STOP_L),
        int(ActionType.STOP_J),
    })

    def _is_drive_waypoint(self, action) -> bool:
        return int(action.action_type) in self._DRIVE_ACTION_TYPES

    def _coalesce_actions(self, actions: list) -> list:
        """Keep only the latest ``MOVE_*`` / ``SERVO_*``; keep every ``SCHEDULE_*``.

        A drive action means "be here as soon as you can", so a backlogged
        burst of them is stale and only thrashes the interpolator. A
        timestamped waypoint names its own moment instead, and a policy chunk
        is seconds of trajectory expressed as several of them, so collapsing
        those would flatten the whole motion onto its final setpoint. Other
        action types keep their order.
        """
        if len(actions) <= 1:
            return actions
        last_drive = None
        for i, action in enumerate(actions):
            if self._is_drive_waypoint(action):
                last_drive = i
        if last_drive is None:
            return actions
        return [
            action
            for i, action in enumerate(actions)
            if i == last_drive or not self._is_drive_waypoint(action)
        ]

    def _register_action_handlers(self) -> None:
        """Scan the MRO for ``@action_handler`` methods (inherited included)."""
        self.action_handlers: dict[int, str] = {}
        for cls in reversed(type(self).__mro__):
            for attr_name, value in cls.__dict__.items():
                if callable(value) and hasattr(value, "_handles_action"):
                    self.action_handlers[int(getattr(value, "_handles_action"))] = attr_name

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
        logger.warning(
            "Module '%s': unknown action type %s, ignoring",
            self.name,
            action.action_type,
        )
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

    @abstractmethod
    def control_step(self):
        """Drive the module's outputs once, between action drain and observation
        publish.

        Called every iteration at ``self.frequency`` Hz. Sensor-only modules
        (cameras) have nothing to drive and implement this as a no-op;
        :class:`~solarian.core.actuator_module.ActuatorModule` samples its
        trajectory interpolator here.
        """

    # ========= worker protocol (driven by the runner) ============
    def worker_setup(self, reconnect: bool) -> None:
        """Prepare this module to run in its execution context.

        Args:
            reconnect: True when the worker body executes in a different
                process than the one that built the module. Device handles,
                log routing, and processor graphs do not survive that boundary
                and are rebuilt here; an inline module already owns valid ones.
        """
        if reconnect:
            status_mp_q = getattr(self, '_status_mp_q', None)
            if status_mp_q is not None:
                StatusBoard.init_child(status_mp_q)

            logger.info(f"Module {self.name} process start method: {get_start_method()}")
            setup_process_logging(
                logger_name=None, log_queue=self._log_queue, install_fallback=False
            )

        logger.info(f"Starting module {self.name}")
        self._register_action_handlers()

        if reconnect:
            # Spawn pickles processor ``inputs`` into copies whose ``id()`` no
            # longer matches class-level SensorField sources. Rebuild from the
            # schema in this process so CalculatedFields resolve.
            if getattr(self, "_module_obs_cls", None) is not None:
                self._robot_processors = self._module_obs_cls.create_processors(
                    ExecutionPlace.ROBOT
                )

            # Device handles inherited from the parent (via fork) are not
            # reliably usable (e.g. AVFoundation cameras). Keep per-port
            # calibration so normalized reads work immediately.
            calibration = self.get_calibration() or {}
            self.disconnect()
            logger.info(f"Module {self.name} process disconnected")
            self.connect(calibration)
            logger.info(f"Module {self.name} process connected")

        # Only a robot-side module gets one; a controller-side module shares the
        # host process's offset.
        connection = getattr(self, "_clock_sync_connection", None)
        if connection is not None:
            try:
                sync_with_server(
                    connection.ip,
                    int(connection.clock_sync_port),
                    connection=connection,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Module '%s': clock sync failed (%s); using local wall "
                    "clock (offset=%.1f ms)",
                    self.name,
                    exc,
                    get_offset() * 1000.0,
                )

        self.reset()
        self._t_last_control_step = time.monotonic()

    def worker_tick(self) -> bool:
        """One control iteration: drain actions, drive outputs, publish state.

        Returns:
            False when a STOP action asked the module to shut down.
        """
        t_now = time.monotonic()

        # Drain actions first so control_step() sees the latest target
        keep_running = True
        try:
            # No queue is normal for a module that only produces: a keyboard
            # or a leader arm on the controller is never sent actions.
            if self.action_queue is not None and not self.action_queue.empty():
                actions = self._coalesce_actions(self.action_queue.get_all())
                for action in actions:
                    keep_running = self._dispatch_action(action, t_now)
        except Empty:
            pass

        manual_profile("control_step", time.monotonic() - self._t_last_control_step)
        self._t_last_control_step = time.monotonic()
        self.control_step()

        self.update_observations()
        return keep_running

    def worker_teardown(self, exit_reason: WorkerExit) -> None:
        # The parent drops the bus before the fork. The handle opened in
        # worker_setup lives only in this process, and only this process
        # can close it.
        self.disconnect()
        if self.verbose:
            print(f"[Module] {self.name} terminated ({exit_reason.value})")

    