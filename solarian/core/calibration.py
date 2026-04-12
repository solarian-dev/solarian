from dataclasses import dataclass
from enum import Enum



class MotorNormMode(str, Enum):
    RANGE_0_100 = "range_0_100"
    RANGE_M100_100 = "range_m100_100"
    DEGREES = "degrees"

@dataclass
class MotorCalibration:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int

    def __eq__(self, other):
        return self.id == other.id and self.homing_offset == other.homing_offset and self.range_min == other.range_min and self.range_max == other.range_max

@dataclass
class Motor:
    id: int
    model: str
    norm_mode: MotorNormMode