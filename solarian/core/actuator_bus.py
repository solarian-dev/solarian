import abc
from typing import Dict, Any

from solarian.core.calibration import MotorCalibration, MotorNormMode
from solarian.core.data import Data, PosData
from solarian.core.component import Component
from solarian.core.actuator import Actuator
from solarian.core.component_bus import ComponentBus
from solarian.utils.general import enter_pressed


class ActuatorBus(ComponentBus):
    """
    Abstract base class for actuator bus implementations.
    
    Actuator buses handle communication with specific hardware types
    (Feetech servos, Dynamixel servos, UR5 robots, etc.)
    
    To create a custom actuator bus:
    
    1. Subclass ActuatorBus and implement all abstract methods
    2. In your Actuator subclass, set bus_class to point to it:
    
        from my_package.bus import MyBus
        
        class MyActuator(Actuator):
            bus_class = MyBus
    """
    
    def __init__(self, port: str, components: Dict[str, Actuator], calibration: Dict[str, MotorCalibration] | None = None):
        super().__init__(port, components, calibration)
        self.components = components
        self.calibration = calibration

    @abc.abstractmethod
    def write(self, values: Dict[str, Data], normalize: bool = True):
        pass
    
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        """Check if the actuators are calibrated."""
        pass
    
    @abc.abstractmethod
    def reset_calibration(self, component_names: list[str]):
        """Reset calibration parameters of the actuators to factory defaults."""
        pass

    @abc.abstractmethod
    def read_calibration(self) -> dict[str, MotorCalibration]:
        """Read calibration parameters from the actuators."""
        pass

    

    @abc.abstractmethod
    def set_half_turn_homings(self, actuator_names: list[str]) -> Dict[str, int]:
        """Set half turn homings for the actuators. Read present position and set/store"""
        pass

    def record_ranges_of_motion(self, actuator_names: list[str]) -> tuple[dict[str, int], dict[str, int]]:
        """Interactively record the min/max encoder values of each motor.
        """
        # if motors is None:
        #     motors = list(self.motors)
        # elif isinstance(motors, (str, int)):
        #     motors = [motors]
        # elif not isinstance(motors, list):
        #     raise TypeError(motors)

        start_positions = self.read(PosData, actuator_names, normalize=False)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        from solarian.utils.status_board import StatusBoard
        StatusBoard.start_ui()
        try:
            user_pressed_enter = False
            while not user_pressed_enter:
                positions = self.read(PosData, actuator_names, normalize=False)
                mins = {motor: min(positions[motor], min_) for motor, min_ in mins.items()}
                maxes = {motor: max(positions[motor], max_) for motor, max_ in maxes.items()}

                StatusBoard.set("MIN", mins)
                StatusBoard.set("POS", positions)
                StatusBoard.set("MAX", maxes)

                if enter_pressed():
                    user_pressed_enter = True
        finally:
            StatusBoard.stop()

        same_min_max = [motor for motor in actuator_names if mins[motor] == maxes[motor]]
        if same_min_max:
            raise ValueError(f"Some motors have the same min and max values:\n{same_min_max}")

        return mins, maxes

    # -- optional angular hook (used by IK sim<->real alignment tooling) ----
    def counts_per_revolution(self, name: str) -> int | None:
        """Encoder counts per full revolution for actuator ``name``.

        Optional and servo-specific: a tick-based bus (e.g. Feetech) returns its
        resolution so the IK-alignment tools can convert raw encoder span to
        radians. The default returns ``None`` (unknown); callers then degrade to
        the qualitative live-mirror / Cartesian checks instead.
        """
        return None

    def center_offset_in_hardware(self, name: str) -> bool:
        """True when this bus applies ``center_offset`` on the servo (Present is already shifted).

        Default ``False``: :meth:`_normalize` / :meth:`_unnormalize` apply
        ``(raw - offset) mod CPR`` in software.
        """
        del name
        return False

    def max_center_offset_abs(self, name: str) -> int | None:
        """Max ``|center_offset|`` for actuator ``name``.

        Hardware buses return the register cap (e.g. Feetech ±2047). Software
        buses return ``CPR-1`` so a wrap can always be cleared. ``None`` means
        no known cap.
        """
        hw = self.max_homing_offset_abs(name)
        if hw is not None:
            return hw
        if not self.center_offset_in_hardware(name):
            cpr = self.counts_per_revolution(name)
            if cpr:
                return int(cpr) - 1
        return None

    def max_homing_offset_abs(self, name: str) -> int | None:
        """Hardware register cap, if any. Prefer :meth:`max_center_offset_abs`."""
        del name
        return None

    def _present_ticks(self, name: str, raw: float) -> float:
        """Map a raw encoder reading into Present (range) ticks."""
        if self.center_offset_in_hardware(name):
            return raw
        offset = int(self.calibration[name].homing_offset or 0) if self.calibration else 0
        if not offset:
            return raw
        cpr = self.counts_per_revolution(name)
        if cpr:
            return (float(raw) - offset) % cpr
        return float(raw) - offset

    def _raw_from_present(self, name: str, present: float) -> float:
        """Inverse of :meth:`_present_ticks`."""
        if self.center_offset_in_hardware(name):
            return present
        offset = int(self.calibration[name].homing_offset or 0) if self.calibration else 0
        if not offset:
            return present
        cpr = self.counts_per_revolution(name)
        if cpr:
            return (float(present) + offset) % cpr
        return float(present) + offset

    def raw_to_radians(self, name: str, raw: float) -> float | None:
        """Convert a raw encoder value to radians, or ``None`` if unknown.

        Built on :meth:`counts_per_revolution`; override for servos that report
        position directly in degrees/radians.
        """
        import math

        cpr = self.counts_per_revolution(name)
        if not cpr:
            return None
        return float(raw) * (2.0 * math.pi) / float(cpr)

    def _normalize(self, values: dict[str, float]) -> dict[str, float]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        normalized_values = {}
        for name, val in values.items():
            min_ = self.calibration[name].range_min
            max_ = self.calibration[name].range_max
            drive_mode = self.calibration[name].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{name}': min and max are equal.")

            cfg = self.components[name].cfg
            if cfg is None:
                raise ValueError(f"Component {name} has no configuration")

            present = self._present_ticks(name, val)
            bounded_val = min(max_, max(min_, present))
            if cfg.mode == MotorNormMode.RANGE_M100_100.value:
                norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
                normalized_values[name] = -norm if drive_mode else norm
            elif cfg.mode == MotorNormMode.RANGE_0_100.value:
                norm = ((bounded_val - min_) / (max_ - min_)) * 100
                normalized_values[name] = 100 - norm if drive_mode else norm
            # elif cfg.mode == MotorNormMode.DEGREES.value:
            #     mid = (min_ + max_) / 2
            #     max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
            #     normalized_values[id_] = (val - mid) * 360 / max_res
            elif cfg.mode == "raw":
                # TODO: implement raw mode enumeration
                normalized_values[name] = val
            else:
                raise NotImplementedError

        return normalized_values

    def _unnormalize(self, values: dict[str, float]) -> dict[str, float]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        unnormalized_values = {}
        for name, val in values.items():
            min_ = self.calibration[name].range_min
            max_ = self.calibration[name].range_max
            drive_mode = self.calibration[name].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{name}': min and max are equal.")

            cfg = self.components[name].cfg
            if cfg is None:
                raise ValueError(f"Component {name} has no configuration")

            if cfg.mode == MotorNormMode.RANGE_M100_100.value:
                val = -val if drive_mode else val
                bounded_val = min(100.0, max(-100.0, val))
                present = ((bounded_val + 100) / 200) * (max_ - min_) + min_
                unnormalized_values[name] = self._raw_from_present(name, present)
            elif cfg.mode == MotorNormMode.RANGE_0_100.value:
                val = 100 - val if drive_mode else val
                bounded_val = min(100.0, max(0.0, val))
                present = (bounded_val / 100) * (max_ - min_) + min_
                unnormalized_values[name] = self._raw_from_present(name, present)
            # elif cfg.mode is MotorNormMode.DEGREES:
            #     mid = (min_ + max_) / 2
            #     max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
            #     unnormalized_values[name] = int((val * max_res / 360) + mid)
            elif cfg.mode == "raw":
                unnormalized_values[name] = val
            else:
                raise NotImplementedError

        return unnormalized_values
        