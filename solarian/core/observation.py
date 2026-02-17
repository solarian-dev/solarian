import inspect
import numpy as np
import time

from .data import Data


class TargetTime():

    @classmethod
    def validate(cls, target_time: float):
        if type(target_time) != float:
            raise ValueError("target_time must be a float")
        if target_time < 0:
            raise ValueError("target_time must be non-negative")

    @classmethod
    def zeros(cls):
        return cls()

    def __init__(self, data: float = time.time()):
        self.validate(data)
        self.data = data
        


class BaseObservation():
    field_name = "field"

    @classmethod
    def zeros(cls):
        data = {}
        for name, field_type in cls.field_types().items():
            data[name] = field_type.zeros().data
        return cls(data)


    @classmethod
    def data_attrs(cls):
        return [
            name for name, value in cls.__dict__.items()
            if not name.startswith("__") and not name.startswith("_") and not inspect.isroutine(value)
        ]

    @classmethod
    def keys(cls):
        return cls.data_attrs() + ["target_time"]

    @classmethod
    def __class_getitem__(cls, item):
        return getattr(cls, item)

    
    def __getitem__(self, name: str):
        return self.__class__.field_types()[name](self.data[name])


    @classmethod
    def field_types(cls):
        res = {}
        for name, value in cls.__dict__.items():
            if not name.startswith("__") and not inspect.isroutine(value):
                res[name] = type(value)
        res["target_time"] = TargetTime
        return res

    @classmethod
    def validate(cls, data: dict):
        for name, value in data.items():
            if name not in cls.keys():
                raise ValueError(f"Invalid {cls.field_name} name: '{name}'")
            cls.field_types()[name].validate(value)


    def __init__(self, data: dict = {}):
        self.__class__.validate(data)
        self.data = data
        # for name, value in self.data.items():
        #     setattr(self, name, value)


    def is_empty(self):
        if len(self.data) == 0:
            return True
        
        for name, value in self.data.items():
            if value is not None:
                return False
        return True

class SensorObs(BaseObservation):
    field_name = "field"

    allowed_attr_type = Data

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for name, value in cls.__dict__.items():
            if name.startswith("_") or inspect.isroutine(value):
                continue

            if not isinstance(value, cls.allowed_attr_type):
                raise TypeError(
                    f"{cls.__name__}.{name} must be of type {cls.allowed_attr_type}, "
                    f"got {type(value)}"
                )

    @classmethod
    def validate(cls, data: dict):
        field_types = cls.field_types()
        for name, value in data.items():
            if name not in cls.keys():
                raise ValueError(f"Invalid {cls.field_name} name: '{name}'")
            
            if cls[name].shape is not None: 
                if not isinstance(value, np.ndarray):
                    raise ValueError(f"Invalid type for {cls.field_name} '{name}': expected {np.ndarray}, got {type(value)}")
                if value.shape != cls[name].shape:
                    raise ValueError(f"Invalid shape for {cls.field_name} '{name}': expected {cls[name].shape}, got {value.shape}")
            else:
                if type(value) != cls[name].dtype:
                    raise ValueError(f"Invalid type for {cls.field_name} '{name}': expected {cls[name].dtype}, got {type(value)}")
            

    def __init__(self, data: dict = {}):
        self.__class__.validate(data)
        self.data = data

    @classmethod
    def field_types(cls):
        res = {}
        for name, value in cls.__dict__.items():
            if not name.startswith("__") and not inspect.isroutine(value):
                res[name] = type(value)
        return res

    @classmethod
    def zeros(cls):
        data = {}
        for name, value in cls.__dict__.items():
            if not name.startswith("__") and not inspect.isroutine(value):
                if value.shape is not None:
                    data[name] = np.zeros(value.shape, dtype=value.dtype)
                elif value.dtype == str:
                    data[name] = ''
                else:
                    data[name] = value.dtype(0)
        return cls(data)

    def __getitem__(self, name: str):
        return self.data[name]


class ModuleObs(BaseObservation):
    field_name = "sensor"

    allowed_attr_type = SensorObs

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for name, value in cls.__dict__.items():
            if name.startswith("_") or inspect.isroutine(value):
                continue

            if not isinstance(value, cls.allowed_attr_type):
                raise TypeError(
                    f"{cls.__name__}.{name} must be of type {cls.allowed_attr_type}, "
                    f"got {type(value)}"
                )


class Observation(BaseObservation):
    field_name = "module"
    
    allowed_attr_type = ModuleObs

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for name, value in cls.__dict__.items():
            if name.startswith("_") or inspect.isroutine(value):
                continue

            if not isinstance(value, cls.allowed_attr_type):
                raise TypeError(
                    f"{cls.__name__}.{name} must be of type {cls.allowed_attr_type}, "
                    f"got {type(value)}"
                )

    