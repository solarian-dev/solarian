"""
UR5 Actuator Bus

Bus implementation for communicating with Universal Robots UR5 arms.
Supports both joint-space and Cartesian-space control.
"""

import math
import platform
import sys
import time
from typing import Any, Dict
from solarian.core.data import Data, PosData
import logging
logger = logging.getLogger(__name__)

try:
    import rtde_control
    import rtde_receive
except ImportError as e:
    if sys.platform == "darwin" and platform.machine() == "arm64":
        raise ImportError(
            "solarian-ur5 requires ur-rtde, which has no pre-built wheel on macOS (Darwin) arm64. "
            "Install it from source: https://sdurobotics.gitlab.io/ur_rtde/installation/installation.html — "
            "clone the repo from https://gitlab.com/sdurobotics/ur_rtde, install Boost (e.g. brew install boost), "
            "then build and install with cmake/make."
        ) from e
    raise

from solarian.core.actuator_bus import ActuatorBus
from solarian.core.actuator import Actuator


FREQ = 120                           # default cadence (Hz) used until we measure the real one
KP = 2.0                             # proportional gain [1/s]; tune 1.0–4.0
MAX_JOINT_SPEED = 1.0                # rad/s cap (per joint)
ACC = 1.0                            # acceleration cap for speedj
SAFE_MARGIN = math.radians(2.0)      # keep some distance from joint limits (in rad)
DEFAULT_DT = 1.0 / FREQ              # initial servo period before EMA converges
# Bounds on the per-tick servo period passed to servoJ/servoL.  The lower
# bound matches the UR controller's max command rate; the upper bound keeps
# us in a region where the lookahead/gain filter still tracks reasonably.
MIN_DT = 1.0 / 250                   # 4 ms
MAX_DT = 1.0 / 30                    # ~33 ms
# How aggressively the measured cadence updates DT.  Small alpha → smoother,
# slower to adapt; large alpha → noisy.
DT_EMA_ALPHA = 0.2
LOOKAHEAD = 0.08                       # s (0.03–0.2 typical)
GAIN = 300                            # unitless (start 100–600 on CB3)

# UR5 (CB3/e-Series) soft limits (approx; adjust to your exact model if needed)
UR5_LIMITS = [
    (-2*math.pi,  2*math.pi),        # base
    (-2*math.pi,  2*math.pi),        # shoulder
    (-2*math.pi,  2*math.pi),        # elbow
    (-2*math.pi,  2*math.pi),        # wrist1
    (-2*math.pi,  2*math.pi),        # wrist2
    (-2*math.pi,  2*math.pi),        # wrist3
]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def clamp_to_limits(q):
    out = []
    for j, qj in enumerate(q):
        lo = UR5_LIMITS[j][0] + SAFE_MARGIN
        hi = UR5_LIMITS[j][1] - SAFE_MARGIN
        out.append(clamp(qj, lo, hi))
    return out


def clamp_joint_value(value: float, joint_id: int):
    lo = UR5_LIMITS[joint_id][0] + SAFE_MARGIN
    hi = UR5_LIMITS[joint_id][1] - SAFE_MARGIN
    return clamp(value, lo, hi)


class UR5Bus(ActuatorBus):
    """
    Actuator bus for Universal Robots UR5 arms.
    
    Supports both joint-space and Cartesian-space control.
    Automatically used when UR5JointActuator or UR5CartesianActuator are defined in a module.
    """
    
    def __init__(self, port: str, components: Dict[str, Actuator], calibration: Dict[str, Any] | None = None):
        super().__init__(port, components, calibration)

        from solarian_ur5.actuator import UR5JointActuator, UR5CartesianActuator

        self.joint_names2ids: Dict[str, int] = {}
        self.cartesian_names2ids: Dict[str, int] = {}

        for name, component in components.items():
            if component.cfg is None:
                raise ValueError(f"Component {name} has no configuration")
            if isinstance(component, UR5JointActuator):
                self.joint_names2ids[name] = component.cfg.id
            elif isinstance(component, UR5CartesianActuator):
                self.cartesian_names2ids[name] = component.cfg.id
            else:
                class_name = component.__class__.__name__
                if class_name == 'UR5JointActuator':
                    self.joint_names2ids[name] = component.cfg.id
                elif class_name == 'UR5CartesianActuator':
                    self.cartesian_names2ids[name] = component.cfg.id
                elif hasattr(component, 'cfg') and hasattr(component.cfg, 'id'):
                    self.joint_names2ids[name] = component.cfg.id

        self.connected = False

        # Streaming-mode state.  servoJ and servoL cannot be active at the
        # same time; when we switch between them we issue servoStop and
        # invalidate the cached target so the next write reseeds from the
        # robot's actual state.
        self._streaming_mode: str | None = None  # 'joint' | 'cartesian' | None
        self._last_cart_target: list[float] | None = None

        # Online measurement of the period between consecutive write() calls,
        # used as the dt argument to servoJ/servoL.  Seeded with DEFAULT_DT
        # and clamped to [MIN_DT, MAX_DT] so jitter or pauses can't push us
        # into a regime servoJ/servoL refuses or tracks badly.
        self._dt_ema: float = DEFAULT_DT
        self._last_write_time: float | None = None

    @property
    def is_connected(self) -> bool:
        """bool: `True` if the underlying serial port is open."""
        return self.connected

    def connect(self):

        print(f"Connecting rtde control interface to {self.port}") 
        self.rtde_c = rtde_control.RTDEControlInterface(self.port)
        print(f"Connecting rtde receive interface to {self.port}") 
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.port)
        self.connected = True
        self._streaming_mode = None
        self._last_cart_target = None
        self._dt_ema = DEFAULT_DT
        self._last_write_time = None

    def disconnect(self):
        self.connected = False
        try:
            self.rtde_c.servoStop()
        except Exception:
            pass
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
        self._streaming_mode = None
        self._last_cart_target = None
        self._last_write_time = None

    def _tick_dt(self) -> float:
        """Update the EMA of the inter-write period and return the current dt.

        servoJ/servoL are streaming primitives: they expect to be called once
        per ``dt`` seconds.  Rather than hard-coding a constant we measure the
        actual cadence at which the module is calling ``write`` and feed that
        back as ``dt``, so the value matches the real loop period regardless
        of the module's configured frequency.
        """
        now = time.monotonic()
        if self._last_write_time is not None:
            measured = now - self._last_write_time
            if measured > 0:
                self._dt_ema = (
                    DT_EMA_ALPHA * measured
                    + (1.0 - DT_EMA_ALPHA) * self._dt_ema
                )
        self._last_write_time = now
        # Clamp so a temporary pause (no writes for a while) doesn't poison
        # the next servo call with an out-of-range period.
        return max(MIN_DT, min(MAX_DT, self._dt_ema))

    def read(self, data_type, component_names: list[str], normalize: bool = True) -> Dict[str, Any]:
        values = {}
        # start with joint values
        names = [name for name in component_names if name in self.joint_names2ids.keys()]
        if names:
            joint_values = self.rtde_r.getActualQ()
            values.update({name: joint_values[self.joint_names2ids[name]] for name in names})

        # then with cartesian values
        names = [name for name in component_names if name in self.cartesian_names2ids.keys()]
        if names:
            cartesian_values = self.rtde_r.getActualTCPPose()
            values.update({name: cartesian_values[self.cartesian_names2ids[name]] for name in names})

        if normalize:
            values = self._normalize(values)

        return values

    def write(self, values: Dict[str, Data], normalize: bool = True):
        raw_values: dict[str, int] = {}
        for name, value in values.items():
            if not isinstance(value, PosData):
                raise ValueError(f"Value {value} must be a PosData, got {type(value)}")
            raw_values[name] = value.value
        if normalize:
            raw_values = self._unnormalize(raw_values)

        dt = self._tick_dt()

        # first joint values
        names = [name for name in raw_values.keys() if name in self.joint_names2ids.keys()]
        if names:
            # If we were just streaming cartesian, end that mode cleanly
            # before starting servoJ — UR's controller does not accept both
            # at the same time, and skipping servoStop here would either
            # block or produce a jolt on the first servoJ.
            if self._streaming_mode == 'cartesian':
                try:
                    self.rtde_c.servoStop()
                except Exception as e:
                    print(e)
                self._last_cart_target = None
            self._streaming_mode = 'joint'

            q_now = self.rtde_r.getActualQ()
            q_target = [clamp_joint_value(raw_values[name], j) if name in raw_values else q_now[j] for name, j in self.joint_names2ids.items()]
            try:
                self.rtde_c.servoJ(q_target,
                                MAX_JOINT_SPEED,
                                ACC,
                                dt,
                                LOOKAHEAD,
                                GAIN)
                # gives priority to joint commands, discards cartesian commands
                return
            except Exception as e:
                print(e)
        
        # then with cartesian values
        names = [name for name in raw_values.keys() if name in self.cartesian_names2ids.keys()]
        if names:
            # Symmetric handshake on the way back into cartesian streaming.
            if self._streaming_mode == 'joint':
                try:
                    self.rtde_c.servoStop()
                except Exception as e:
                    print(e)
                self._last_cart_target = None
            self._streaming_mode = 'cartesian'

            # Maintain the *commanded* cartesian target across ticks.  If we
            # rebased on getActualTCPPose() every call, the (intentional)
            # tracking lag of servoL — controlled by lookahead/gain — would
            # leak into our base pose and produce drift / oscillation.  Seed
            # once from actual, then update only the components that change.
            if self._last_cart_target is None:
                self._last_cart_target = list(self.rtde_r.getActualTCPPose())
            for name in names:
                self._last_cart_target[self.cartesian_names2ids[name]] = raw_values[name]

            try:
                # speed/accel are dummies on UR5 servoL — dt/lookahead/gain
                # are what actually shape the motion.
                self.rtde_c.servoL(self._last_cart_target,
                                   0.5, 0.5,
                                   dt,
                                   LOOKAHEAD,
                                   GAIN)
            except Exception as e:
                print(e)

    def is_calibrated(self):
        return self.calibration != {}

    def reset_calibration(self, component_names: list[str]):
        self.calibration = {}

    def read_calibration(self) -> dict[str, Any]:
        return self.calibration
        
    def write_calibration(self, calibration_dict: dict[str, Any]) -> dict[str, Any]:
        self.calibration = calibration_dict
        return calibration_dict

    def start_calibration_mode(self):
        print(f"Starting calibration mode for {self.port}")
        # Stop any in-flight servo stream first — teachMode and servoJ/servoL
        # cannot be active simultaneously.
        try:
            self.rtde_c.servoStop()
        except Exception:
            pass
        self._streaming_mode = None
        self._last_cart_target = None
        # Enable freedrive (gravity compensation)
        result = self.rtde_c.teachMode()
        print(f"Result of freedriveMode: {result}")
        print("freedrive status:", self.rtde_c.getFreedriveStatus())

    def end_calibration_mode(self):
        self.rtde_c.endTeachMode()


    def set_half_turn_homings(self, actuator_names: list[str]) -> Dict[str, int]:
        self.reset_calibration(actuator_names)
        half_turn_homings = {}
        actual_positions = self.read(PosData, actuator_names, normalize=False)
        for name, pos in actual_positions.items():
            if name not in self.calibration:
                self.calibration[name] = {}
            self.calibration[name]["homing_offset"] = pos
            half_turn_homings[name] = pos
        return half_turn_homings
