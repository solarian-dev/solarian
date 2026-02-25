import abc
from typing import Dict, Any

from solarian.core.calibration import MotorCalibration, MotorNormMode
from solarian.core.data import Data, PosData
from solarian.core.component import Component
from solarian.core.component_bus import ComponentBus
from solarian.utils.general import enter_pressed
from solarian.utils.status_board import StatusBoard


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
    
    def __init__(self, port: str, components: Dict[str, Component], calibration: Dict[str, MotorCalibration]):
        super().__init__(port, components)
        self.calibration = calibration

    @classmethod
    def create(cls, port: str, components: Dict[str, Component], calibration=None) -> "ActuatorBus":
        """
        Factory method to create a bus instance.
        
        Override this in subclasses that need special construction logic
        (e.g., extracting joint_names2ids from components).
        
        Args:
            port: The communication port (e.g., '/dev/ttyUSB0' or IP address)
            components: Dict of component name to Component
            calibration: Calibration data for the components
            
        Returns:
            An instance of the ActuatorBus subclass
        """
        return cls(port=port, components=components, calibration=calibration)

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

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """Read calibration parameters from the actuators."""
        pass

    @abc.abstractmethod
    def write_calibration(self, calibration_dict: dict[str, MotorCalibration]) -> dict[str, MotorCalibration]:
        """Write calibration parameters to the actuators and cache them.

        Args:
            calibration_dict (dict[str, MotorCalibration]): parameters per actuator

        Returns:
            dict[str, MotorCalibration]: parameters per actuator, it might bound the ranges of motion
        """
        pass

    @abc.abstractmethod
    def start_calibration_mode(self):
        """Sets the actuators where the arm can be moved manually freely"""
        pass

    @abc.abstractmethod
    def end_calibration_mode(self):
        pass

    @abc.abstractmethod
    def set_half_turn_homings(self, component_names: list[str]) -> Dict[str, int]:
        """Set half turn homings for the actuators. Read present position and set/store"""
        pass

    def record_ranges_of_motion(self, component_names: list[str]) -> tuple[dict[str, int], dict[str, int]]:
        """Interactively record the min/max encoder values of each motor.
        """
        # if motors is None:
        #     motors = list(self.motors)
        # elif isinstance(motors, (str, int)):
        #     motors = [motors]
        # elif not isinstance(motors, list):
        #     raise TypeError(motors)

        start_positions = self.read(PosData, component_names, normalize=False)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        user_pressed_enter = False
        while not user_pressed_enter:
            positions = self.read(PosData, component_names, normalize=False)
            mins = {motor: min(positions[motor], min_) for motor, min_ in mins.items()}
            maxes = {motor: max(positions[motor], max_) for motor, max_ in maxes.items()}

            # print("\n-------------------------------------------")
            # print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
            # for motor in component_names:
            #     print(f"{motor:<15} | {mins[motor]:>6} | {positions[motor]:>6} | {maxes[motor]:>6}")
            StatusBoard.set("MIN", mins)
            StatusBoard.set("POS", positions)
            StatusBoard.set("MAX", maxes)

            if enter_pressed():
                user_pressed_enter = True

        same_min_max = [motor for motor in component_names if mins[motor] == maxes[motor]]
        if same_min_max:
            raise ValueError(f"Some motors have the same min and max values:\n{pformat(same_min_max)}")

        return mins, maxes

    def _normalize(self, values: dict[str, int]) -> dict[int, float]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        normalized_values = {}
        for name, val in values.items():
            min_ = self.calibration[name].range_min
            max_ = self.calibration[name].range_max
            drive_mode = self.calibration[name].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{name}': min and max are equal.")

            bounded_val = min(max_, max(min_, val))
            if self.components[name].cfg.mode == MotorNormMode.RANGE_M100_100.value:
                norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
                normalized_values[name] = -norm if drive_mode else norm
            elif self.components[name].cfg.mode == MotorNormMode.RANGE_0_100.value:
                norm = ((bounded_val - min_) / (max_ - min_)) * 100
                normalized_values[name] = 100 - norm if drive_mode else norm
            # elif self.components[name].cfg.mode == MotorNormMode.DEGREES.value:
            #     mid = (min_ + max_) / 2
            #     max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
            #     normalized_values[id_] = (val - mid) * 360 / max_res
            elif self.components[name].cfg.mode == "raw":
                # TODO: implement raw mode enumeration
                normalized_values[name] = val
            else:
                raise NotImplementedError

        return normalized_values

    def _unnormalize(self, values: dict[str, float]) -> dict[str, int]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        unnormalized_values = {}
        for name, val in values.items():
            min_ = self.calibration[name].range_min
            max_ = self.calibration[name].range_max
            drive_mode = self.calibration[name].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{name}': min and max are equal.")

            if self.components[name].cfg.mode == MotorNormMode.RANGE_M100_100.value:
                val = -val if drive_mode else val
                bounded_val = min(100.0, max(-100.0, val))
                unnormalized_values[name] = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)
            elif self.components[name].cfg.mode == MotorNormMode.RANGE_0_100.value:
                val = 100 - val if drive_mode else val
                bounded_val = min(100.0, max(0.0, val))
                unnormalized_values[name] = int((bounded_val / 100) * (max_ - min_) + min_)
            # elif self.components[name].cfg.mode is MotorNormMode.DEGREES:
            #     mid = (min_ + max_) / 2
            #     max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
            #     unnormalized_values[name] = int((val * max_res / 360) + mid)
            elif self.components[name].cfg.mode == "raw":
                unnormalized_values[name] = val
            else:
                raise NotImplementedError

        return unnormalized_values
        