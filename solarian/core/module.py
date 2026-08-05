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
from solarian.core.actuator_bus import ActuatorBus
from solarian.core.sensor import Sensor
from solarian.core.servo import Servo
from solarian.core.data import Data, PosData, DataComputer, TimestampData
from solarian.core.module_bus import ModuleBus
from solarian.core.action import Action, ActionType, action_handler, ModuleAction
from solarian.core.configs import ModuleConfig, InterpolatorType
from solarian.core.observation import Observation, ModuleObs
from solarian.utils import manual_profile, profile, setup_process_logging, get_log_queue
from solarian.utils.profiling import get_stats_snapshot, merge_stats as merge_profiling_stats
from solarian.utils.shared_memory.shared_memory_queue import Empty
from solarian.utils.shared_memory.action_queue import ModuleActionQueue
from solarian.utils.shared_memory.observation_ring_buffer import ObservationRingBuffer
from solarian.utils.trajectory_interpolator import (
    TrajectoryInterpolator,
    ScalarTrajectoryInterpolator,
    JointTrajectoryInterpolator,
    PoseTrajectoryInterpolator,
)
from solarian.utils.precise_sleep import precise_wait
from solarian.utils import get_class
from solarian.utils.status_board import StatusBoard
# TODO: remove speed from calibration required
# CALIBRATION_REQUIRED = [ValueName.POS]

from solarian.utils import profile, manual_profile

# Pose component order for Cartesian IK modules (matches CartesianIKActuator cfg.id 0..5).
_POSE_AXES = ("x", "y", "z", "rx", "ry", "rz")


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
    
    
    
    Standard actions (STOP, SCHEDULE_WAYPOINT, RESTART_PUT) are handled
    by default. Subclasses can override these or add new actions.
    """

    # Override in subclasses to use a custom config class (must extend ModuleConfig)
    config_class: type = ModuleConfig

    # Optional IK backend for arms that declare Cartesian IK actuators. Set this
    # on the subclass to a declarative spec (e.g. ``MinkIK(...)``), an object
    # exposing ``.build()``, or a ready :class:`DifferentialIKBackend`. The core
    # resolves it automatically when connecting a port with Cartesian IK
    # actuators, so no ``build_ik_backend`` method is required.
    ik_backend = None

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

        # Build action handler registry from @action_handler decorated methods
        self.action_handlers: dict[int, str] = {}
        for attr_name in dir(self):
            method = getattr(type(self), attr_name, None)
            if callable(method) and hasattr(method, '_handles_action'):
                self.action_handlers[getattr(method, '_handles_action')] = attr_name

        self.verbose = True
        self.is_connected = False
        
        # Runtime state for the control loop
        self.interp: TrajectoryInterpolator | None = None
        self.last_waypoint_time: float = 0.0
        self.speed_kwargs: dict[str, Any] = {}
        
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
        # Lets modules layer extra behavior (e.g. Cartesian IK) over a port's
        # joint bus. Default wrap_port_bus is a no-op unless the module declares
        # Cartesian IK actuators and sets a class-level ``ik_backend`` spec.
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
        
    def _hardware_actuators_by_port(self) -> dict[str, list[str]]:
        """Port → hardware actuator names (excludes virtual IK Cartesian axes)."""
        from solarian.core.ik.actuator import CartesianIKActuator

        out: dict[str, list[str]] = {}
        for port, names in self.actuators_by_port.items():
            hw = [
                n for n in names
                if not isinstance(self.actuators[n], CartesianIKActuator)
            ]
            if hw:
                out[port] = hw
        return out

    def calibrate(self) -> None:
        """Calibrate hard-stop joints from a range sweep only.

        Flow:
        1. Reset homing offset to 0 and record raw min/max while the operator
           sweeps each joint's full travel (no middle-pose prompt). Glitch
           samples that would expand the span across the encoder wrap are
           ignored so travel near 0/CPR does not require remounting.
        2. Keep ``homing_offset=0``; refuse non-contiguous / over-wide spans.
        3. Ask the operator to move each joint to its **low command** end to set
           ``drive_mode``.
        """
        from solarian.core.data import PosData
        from solarian.utils.general import enter_pressed
        from solarian.core.joint_calibration import (
            CalibrationSafetyError,
            DerivedHoming,
            EncoderRangeTracker,
            calibration_from_raw_range,
            drive_mode_from_low_end_pose,
            low_command_label,
        )

        if self.cfg.calibration_file is None:
            logger.error(f"No calibration required for module {self.name}")
            return

        hw_by_port = self._hardware_actuators_by_port()
        if not hw_by_port:
            logger.error(f"No hardware actuators to calibrate on module {self.name}")
            return

        # Continuous (no hard-stop) joints are zeroed in ``calibrate-ik``, not here.
        continuous = set(getattr(type(self), "ik_continuous_joints", []) or [])
        if continuous:
            hw_by_port = {
                port: [n for n in names if n not in continuous]
                for port, names in hw_by_port.items()
            }
            hw_by_port = {p: ns for p, ns in hw_by_port.items() if ns}
            print(
                f"Skipping continuous joints (no hard stop; zero them in "
                f"calibrate-ik): {sorted(continuous)}"
            )
            if not hw_by_port:
                logger.error(
                    f"No hard-stop actuators left to calibrate on module {self.name}"
                )
                return

        print(f"\nRunning calibration of {self}")
        print(
            "Homing offset stays 0 (raw encoder). Move each joint through its\n"
            "FULL mechanical travel while recording, then press ENTER.\n"
            "The sweep defines the limits (no span cap). Travel must stay on\n"
            "one side of the encoder wrap (POS must not jump 4095→0). If a\n"
            "joint crosses the wrap, remount that horn, then re-run."
        )

        self.start_calibration_mode()

        for port, actuator_names in hw_by_port.items():
            bus = self.module_bus.component_bus_by_port[port]
            assert isinstance(bus, ActuatorBus)
            bus.reset_calibration(actuator_names)

        # Glitch-filtered range recording (per joint, needs CPR from the bus).
        trackers: dict[str, EncoderRangeTracker] = {}
        for port, actuator_names in hw_by_port.items():
            bus = self.module_bus.component_bus_by_port[port]
            for name in actuator_names:
                cpr = bus.counts_per_revolution(name)
                if not cpr:
                    print(
                        f"\nCalibration aborted: {name} has no counts_per_revolution."
                    )
                    self.end_calibration_mode()
                    return
                trackers[name] = EncoderRangeTracker(cpr=int(cpr))

        print(
            "\nRecording positions. Move every joint through its entire range of "
            "motion, then press ENTER to stop..."
        )
        while not enter_pressed():
            status_min, status_pos, status_max = {}, {}, {}
            for port, actuator_names in hw_by_port.items():
                bus = self.module_bus.component_bus_by_port[port]
                positions = bus.read(PosData, actuator_names, normalize=False)
                for name in actuator_names:
                    pos = int(positions[name])
                    trackers[name].update(pos)
                    t = trackers[name]
                    status_pos[name] = pos
                    status_min[name] = t.raw_min
                    status_max[name] = t.raw_max
                    if t.rejected:
                        # Surface rejected samples so wrap-cross / over-wide
                        # travel is obvious while sweeping (not only at the end).
                        StatusBoard.set(
                            f"{name}_ignored",
                            t.rejected,
                        )
            StatusBoard.set("MIN", status_min)
            StatusBoard.set("POS", status_pos)
            StatusBoard.set("MAX", status_max)

        # Build calibration windows (homing_offset=0).
        derived: dict[str, DerivedHoming] = {}
        try:
            for name, tracker in trackers.items():
                if tracker.rejected:
                    wrap_note = (
                        f" ({tracker.wrap_rejected} crossed encoder wrap)"
                        if tracker.wrap_rejected
                        else ""
                    )
                    print(
                        f"  {name}: ignored {tracker.rejected} glitch/wrap sample(s)"
                        f"{wrap_note}"
                    )
                raw_lo, raw_hi = tracker.result()
                # Re-validate with the joint name in the error message.
                derived[name] = calibration_from_raw_range(
                    name, raw_lo, raw_hi, tracker.cpr
                )
                d = derived[name]
                print(
                    f"  {name}: raw/Present range {d.range_min} .. {d.range_max} "
                    f"(span {d.range_max - d.range_min}, homing_offset=0)"
                )
                if tracker.wrap_rejected:
                    print(
                        f"  warning: {name} saw wrap-crossing samples during the "
                        f"sweep — if MIN/MAX look incomplete, remount the horn "
                        f"so travel does not cross 0/{tracker.cpr - 1}."
                    )
        except CalibrationSafetyError as exc:
            print(f"\nCalibration aborted (safety): {exc}")
            logger.error("Calibration aborted (safety): %s", exc)
            self.end_calibration_mode()
            return

        # Drive mode: operator moves each joint to its low-command pose.
        print("\n=== Drive mode (command direction) ===")
        drive_modes: dict[str, int] = {}
        for port, actuator_names in hw_by_port.items():
            bus = self.module_bus.component_bus_by_port[port]
            for name in actuator_names:
                mode = self.actuators[name].cfg.mode
                label = low_command_label(mode)
                input(
                    f"Move '{name}' to the pose that should map to command {label}, "
                    "then press ENTER..."
                )
                reading = bus.read(PosData, [name], normalize=False)
                pose = float(reading[name])
                d = derived[name]
                dm = drive_mode_from_low_end_pose(pose, d.range_min, d.range_max)
                drive_modes[name] = dm
                end = "min" if dm == 0 else "max"
                print(
                    f"  {name}: pose={pose:.0f} near encoder {end} → drive_mode={dm}"
                )
                sim = getattr(self.actuators[name].cfg, "sim", None)
                sim_dm = getattr(sim, "drive_mode", None) if sim is not None else None
                if sim_dm is not None and int(sim_dm) != dm:
                    print(
                        f"  warning: sim.drive_mode={sim_dm} disagrees with operator "
                        f"result ({dm}); keeping operator result."
                    )

        # Preserve continuous-joint entries (they are not swept here; ``calibrate-ik``
        # Step 2 anchors their zero). Dropping them would leave wrist_roll etc.
        # without a calibration row after a hard-stop-only sweep.
        preserved = {
            name: entry
            for name, entry in (self.calibration or {}).items()
            if name in continuous
        }
        self.calibration = dict(preserved)
        for port, actuator_names in hw_by_port.items():
            port_calibration = {}
            for name in actuator_names:
                d = derived[name]
                port_calibration[name] = MotorCalibration(
                    id=self.actuators[name].cfg.id,
                    drive_mode=drive_modes[name],
                    homing_offset=d.homing_offset,
                    range_min=d.range_min,
                    range_max=d.range_max,
                )
            port_calibration = self.module_bus.component_bus_by_port[port].write_calibration(
                port_calibration
            )
            self.calibration.update(port_calibration)

        self._save_calibration(self.calibration)
        logger.info(f"Calibration saved to {self.cfg.calibration_file}")
        self.write_calibration()
        logger.info("Calibration written to actuators")
        self.end_calibration_mode()

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
        """Optionally wrap a port's joint bus to add Cartesian (IK) control.

        Called by :class:`~solarian.core.module_bus.ModuleBus` when connecting a
        port that has Cartesian IK actuators. The default builds an
        :class:`~solarian.core.ik.bus.IKActuatorBus` around ``inner_bus`` using
        the backend resolved from the module's :attr:`ik_backend`; modules with
        no Cartesian actuators never reach here (``inner_bus`` is used directly).
        """
        if not cartesian_components:
            return inner_bus
        from solarian.core.ik.bus import IKActuatorBus
        ik_bus = IKActuatorBus(
            inner_bus=inner_bus,
            joint_components=joint_components,
            cartesian_components=cartesian_components,
            backend=self._resolve_ik_backend(),
        )
        return ik_bus

    def _resolve_ik_backend(self):
        """Build the :class:`DifferentialIKBackend` from :attr:`ik_backend`.

        Accepts a declarative spec (anything exposing ``.build()``, e.g.
        :class:`~solarian.core.ik.MinkIK`), a ready
        :class:`~solarian.core.ik.DifferentialIKBackend`, or a zero-arg callable
        returning one. This is why arms using a built-in backend never need to
        write backend-construction code.
        """
        from solarian.core.ik.base import DifferentialIKBackend

        spec = type(self).ik_backend
        if spec is None:
            raise NotImplementedError(
                f"Module '{self.name}' declares Cartesian IK actuators but no "
                f"`ik_backend` is set. Declare one on the module class, e.g. "
                f"`ik_backend = MinkIK(mjcf=..., joints=..., frame=...)`."
            )
        if isinstance(spec, DifferentialIKBackend):
            return spec
        build = getattr(spec, "build", None)
        if callable(build):
            return build()
        if callable(spec):
            return spec()
        raise TypeError(
            f"Module '{self.name}'.ik_backend must be a DifferentialIKBackend, a "
            f"spec with .build() (e.g. MinkIK), or a zero-arg callable; got "
            f"{type(spec).__name__}."
        )

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
                sensor_data[sensor_name]['timestamp'] = time.time() - self.cfg.receive_latency
                data[field_name] = sensor_data[sensor_name]

        obs = self._module_obs_cls(data)
        obs = self.enrich_observation(obs)

        self.observation_buffer.put(obs)

    def reset(self):
        """Reset the module. Eg: set to home position"""
        pass

    # ========= pose / trajectory interpolation ============
    def _cartesian_pose_names(self) -> list[str]:
        """Ordered Cartesian actuator names (by ``cfg.id`` 0..5)."""
        from solarian.core.ik.actuator import CartesianIKActuator

        named: list[tuple[int, str]] = []
        for name, act in self.actuators.items():
            if isinstance(act, CartesianIKActuator):
                named.append((int(act.cfg.id), name))
        named.sort(key=lambda item: item[0])
        return [n for _, n in named]

    def _is_pose_interpolator(self) -> bool:
        return self.cfg.interpolator_type == InterpolatorType.POSE

    def _is_scalar_interpolator(self) -> bool:
        return self.cfg.interpolator_type == InterpolatorType.SCALAR

    def _scalar_actuator_name(self) -> str | None:
        """Name of the single actuator driven by a scalar interpolator.

        Override via class attr ``scalar_actuator = \"gripper\"``, or defaults to
        the sole non-Cartesian actuator on the module.
        """
        from solarian.core.ik.actuator import CartesianIKActuator

        explicit = getattr(type(self), "scalar_actuator", None)
        if explicit:
            return str(explicit)
        names = [
            n for n, act in self.actuators.items()
            if not isinstance(act, CartesianIKActuator)
        ]
        return names[0] if names else None

    def _scalar_to_actuator(self, value: float) -> float:
        """Map interpolator/action units to ``write_actuators`` units.

        Override when the action space differs from the servo command space
        (e.g. gripper aperture cm → normalized [-100, 100]).
        """
        return float(value)

    def _actuator_to_scalar(self, value: float) -> float:
        """Inverse of :meth:`_scalar_to_actuator` (for seeding from sensors)."""
        return float(value)

    def _read_current_pose(self) -> np.ndarray:
        """FK the current end-effector pose via Cartesian actuator reads."""
        names = self._cartesian_pose_names()
        if not names:
            return np.zeros(6, dtype=np.float64)
        readings = self.read_sensors(PosData, names)
        pose = np.zeros(len(names), dtype=np.float64)
        for i, name in enumerate(names):
            entry = readings[name]
            pose[i] = float(entry["value"] if isinstance(entry, dict) else entry.value)
        return pose

    def _ensure_pose_interpolator(self, t_now: float | None = None) -> None:
        """Seed a PoseTrajectoryInterpolator from the live FK pose if needed."""
        if not self._is_pose_interpolator():
            return
        if self.interp is not None:
            return
        t = float(t_now if t_now is not None else time.monotonic())
        try:
            pose = self._read_current_pose()
        except Exception as exc:  # noqa: BLE001 - degrade to zeros so teleop can start
            logger.warning(
                "Module '%s': could not seed pose interpolator from FK (%s); using zeros.",
                self.name,
                exc,
            )
            pose = np.zeros(6, dtype=np.float64)
        self.interp = self._create_interpolator(
            np.array([t]), np.asarray(pose, dtype=np.float64)[None, :]
        )
        self.last_waypoint_time = t

    def _clear_pose_interpolator(self) -> None:
        """Drop the pose trajectory so the next tick re-seeds from live FK."""
        self.interp = None
        self.last_waypoint_time = 0.0

    def _extract_pose_waypoint(
        self, action: ModuleAction
    ) -> tuple[np.ndarray, float] | None:
        """Pull a 6D pose + wall-clock timestamp from a Cartesian ModuleAction."""
        names = self._cartesian_pose_names()
        if not names:
            return None
        fields = action.values()
        missing = [n for n in names if n not in fields]
        if missing:
            logger.warning(
                "Module '%s': SCHEDULE_WAYPOINT missing pose fields %s; ignoring.",
                self.name,
                missing,
            )
            return None
        pose = np.array([float(fields[n].value) for n in names], dtype=np.float64)
        target_time_wall = float(fields[names[0]].timestamp)
        return pose, target_time_wall

    def _schedule_pose_waypoint(
        self,
        pose: np.ndarray,
        target_time_wall: float,
        t_now: float,
    ) -> None:
        """Insert a Cartesian waypoint (UMI-style wall→monotonic conversion)."""
        self._ensure_pose_interpolator(t_now)
        assert self.interp is not None
        target_time = time.monotonic() - time.time() + float(target_time_wall)
        self.interp = self.interp.schedule_waypoint(
            pose=np.asarray(pose, dtype=np.float64),
            time=target_time,
            curr_time=t_now,
            last_waypoint_time=self.last_waypoint_time,
            **self._get_max_speed_kwargs(),
        )
        self.last_waypoint_time = target_time

    def _write_pose(self, pose: np.ndarray) -> None:
        """Write a 6D pose to Cartesian actuators (IK bus solves joints)."""
        names = self._cartesian_pose_names()
        if not names:
            return
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        self.write_actuators(
            {name: PosData(value=float(pose[i])) for i, name in enumerate(names)}
        )

    def _read_current_scalar(self) -> float:
        """Read the live scalar command (in interpolator/action units)."""
        name = self._scalar_actuator_name()
        if not name:
            return 0.0
        readings = self.read_sensors(PosData, [name])
        entry = readings[name]
        raw = float(entry["value"] if isinstance(entry, dict) else entry.value)
        return self._actuator_to_scalar(raw)

    def _ensure_scalar_interpolator(self, t_now: float | None = None) -> None:
        """Seed a ScalarTrajectoryInterpolator from the live reading if needed."""
        if not self._is_scalar_interpolator():
            return
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
        self.interp = self._create_interpolator(np.array([t]), np.array([value], dtype=np.float64))
        self.last_waypoint_time = t

    def _extract_scalar_waypoint(
        self, action: ModuleAction
    ) -> tuple[float, float] | None:
        """Pull a scalar value + wall-clock timestamp from a ModuleAction."""
        name = self._scalar_actuator_name()
        if not name:
            return None
        fields = action.values()
        if name not in fields:
            # Fall back to the sole field when the action uses a different key.
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
    ) -> None:
        """Insert a scalar waypoint (UMI-style wall→monotonic conversion)."""
        self._ensure_scalar_interpolator(t_now)
        assert self.interp is not None
        target_time = time.monotonic() - time.time() + float(target_time_wall)
        self.interp = self.interp.schedule_waypoint(
            value=float(value),
            time=target_time,
            curr_time=t_now,
            last_waypoint_time=self.last_waypoint_time,
            **self._get_max_speed_kwargs(),
        )
        self.last_waypoint_time = target_time

    def _write_scalar(self, value: float) -> None:
        """Write a scalar sample to the module's primary actuator."""
        name = self._scalar_actuator_name()
        if not name:
            return
        self.write_actuators({name: PosData(value=self._scalar_to_actuator(value))})

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
        if self._is_pose_interpolator():
            try:
                return self._read_current_pose()
            except Exception:
                return np.zeros(6, dtype=np.float64)
        if self._is_scalar_interpolator():
            try:
                return np.array([self._read_current_scalar()], dtype=np.float64)
            except Exception:
                return np.array([0.0], dtype=np.float64)
        return np.array([0.0] * len(self.actuators))

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

    # ========= standard action handlers ============
    @action_handler(ActionType.SCHEDULE_WAYPOINT)
    def _handle_schedule_waypoint(self, action: ModuleAction, t_now: float) -> bool:
        """Schedule a timestamped waypoint into the module trajectory interpolator.

        * ``interpolator_type: pose`` — Cartesian ``x..rz`` (MinkIK arms).
        * ``interpolator_type: scalar`` — single actuator (grippers, etc.).

        Controllers / skills send sparse waypoints; ``control_step`` samples at
        the module rate and writes targets (pose → IK bus, scalar → actuator).
        """
        if self._is_pose_interpolator():
            extracted = self._extract_pose_waypoint(action)
            if extracted is None:
                return True
            pose, target_time_wall = extracted
            self._schedule_pose_waypoint(pose, target_time_wall, t_now)
            return True

        if self._is_scalar_interpolator():
            extracted = self._extract_scalar_waypoint(action)
            if extracted is None:
                return True
            value, target_time_wall = extracted
            self._schedule_scalar_waypoint(value, target_time_wall, t_now)
            return True

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

    def control_step(self):
        """Sample pose/scalar interpolators (when configured) and write targets.

        Subclasses may override; call ``super().control_step()`` to keep UMI-style
        upsampling, or replace entirely for latch-only modules.
        """
        if self._is_pose_interpolator():
            self._ensure_pose_interpolator()
            if self.interp is not None:
                self._write_pose(self.interp(time.monotonic()))
            return
        if self._is_scalar_interpolator():
            self._ensure_scalar_interpolator()
            if self.interp is not None:
                self._write_scalar(float(self.interp(time.monotonic())))
            return

    # ========= main loop in process ============
    def run(self):
        """Main control loop running in a separate process."""

        status_mp_q = getattr(self, '_status_mp_q', None)
        if status_mp_q is not None:
            StatusBoard.init_child(status_mp_q)

        logger.info(f"Module {self.name} process start method: {get_start_method()}")
        setup_process_logging(logger_name=None, log_queue=self._log_queue, install_fallback=False)

        logger.info(f"Starting module {self.name} process")

        # Reconnect hardware in the subprocess — device handles inherited
        # from the parent (via fork) are not reliably usable (e.g. AVFoundation cameras).
        # Keep per-port calibration so normalized reads work immediately.
        calibration = self.get_calibration() or {}
        self.disconnect()
        logger.info(f"Module {self.name} process disconnected")
        self.connect(calibration)

        logger.info(f"Module {self.name} process connected")

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
                        actions = self.action_queue.get_all()
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

    