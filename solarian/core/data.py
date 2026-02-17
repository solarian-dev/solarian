from __future__ import annotations
from typing import Any, Optional, Tuple, Type
from enum import Enum


class DataComputer(int, Enum):
    SERVER = 0
    ROBOT = 1
    CONTROLLER = 2


class DataKind(int, Enum):
    OTHER = 0
    POS = 1
    SPEED = 2
    IMAGE = 3
    BOUNDING_BOX = 4
    DEPTH_IMAGE = 5
    TIMESTAMP = 6
    


class Data():
    '''
    Data class for storing data points in an observation or action.
    
    Args:
        dtype: The data type of the data.
        shape: The shape of the data. (optional, only used for array types)
        computer: The node that the data is from. (server, robot, controller)
        value: The value of the data.
    '''

    def __init__(self, dtype: Type, computer: DataComputer, default: Any = None, shape: Optional[Tuple[int, ...]] = None, kind: DataKind = DataKind.OTHER):
        self.dtype = dtype
        self.computer = computer
        self.default = default
        self.shape = shape
        self.kind = kind

    def __repr__(self) -> str:
        return f"Data(dtype={self.dtype!r}, computer={self.computer!r}, default={self.default!r}, shape={self.shape!r}, kind={self.kind!r})"

