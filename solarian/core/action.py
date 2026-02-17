import time
import inspect
from typing import Dict, Any, Callable
from pydantic import BaseModel, Field
from solarian.core.configs import ValueName
import numpy as np
# from solarian.conf import settings
from solarian.utils import value_str
from enum import IntEnum
from typing import Any, Union, get_args, get_origin
from solarian.core.data import Data


class ActionType(IntEnum):
    """
    Standard action types supported by all modules.
    
    Modules can extend this by creating their own IntEnum that starts
    numbering from CUSTOM_START to avoid conflicts.
    """
    STOP = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2
    
    # Reserve range for future standard actions
    # Custom module actions should start from this value
    CUSTOM_START = 100


def action_handler(action_type: int):
    """
    Decorator to register a method as a handler for a specific action type.
    
    Usage:
        class MyModule(Module):
            @action_handler(BaseAction.SCHEDULE_WAYPOINT)
            def handle_schedule_waypoint(self, action: "Action") -> bool:
                # Handle the action
                return True  # Return False to stop the module
    
    Args:
        action_type: The action type value (from BaseAction or custom enum)
    """
    def decorator(method: Callable) -> Callable:
        method._handles_action = action_type
        return method
    return decorator


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


class BaseAction():
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


class ActuatorAction(BaseAction):
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
                # if isinstance(value, Data):
                #     if value.shape is not None:
                #         res[name] = np.ndarray
                #     else:
                #         res[name] = value.dtype
                #     # res[name] = value.dtype
                    
                # else:
                #     raise ValueError(f"Invalid {cls.field_name} type: '{type(value)}'")
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


class ModuleAction(BaseAction):
    field_name = "sensor"

    allowed_attr_type = ActuatorAction

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

    def __init__(self, data: dict = {}, target_time: float = time.time()):
        super().__init__(data)
        self.target_time = target_time


class Action(BaseAction):
    field_name = "module"
    
    allowed_attr_type = ModuleAction

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



# class Action(BaseModel):
#     """
#     Class for robot actions.
    
#     An Action can represent:
#     1. A command to a module (with action_type set)
#     2. Actuator values to be executed (with data set)
    
#     Attributes:
#         action_type: The type of action (from BaseAction or module-specific enum)
#         target_time: Target time for time-based actions
#         data: Action payload - can be target values, actuator data, or custom data
#     """
    
#     action_type: int = ActionType.STOP
#     target_time: float = 0.0
#     # data: Dict[str, Any] = {}
    
#     class Config:
#         arbitrary_types_allowed = True

#     @classmethod
#     def keys(cls):
#         return list(cls.model_fields.keys())

#     def __getitem__(self, item: str):
#         return getattr(self, item)

#     def get(self, key: str):
#         return getattr(self, key)

#     def __setitem__(self, key: str, value: Any):
#         setattr(self, key, value)

#     def set(self, key: str, value: Any):
#         setattr(self, key, value)

#     def __iter__(self):
#         return iter(self.model_fields.keys())

#     def items(self):
#         for key in self:
#             yield key, getattr(self, key)


#     @classmethod
#     def create_zeros(cls):
#         action = cls.create_null()
#         for key in action:
#             val = action[key]
#             if isinstance(val, ModuleAction):
#                 action[key] = val.create_zeros()
#             elif val is None:
#                 # Handle Optional[ModuleAction] fields — unwrap and instantiate
#                 field = cls.model_fields[key]
#                 ann = _unwrap_optional(field.annotation)
#                 if isinstance(ann, type) and issubclass(ann, ModuleAction):
#                     action[key] = ann.create_zeros()
#         return action

#     @classmethod
#     def create_null(cls):
#         return cls()
        
#     # ========= Factory methods for common actions ============
#     @classmethod
#     def stop(cls) -> "Action":
#         """Create a STOP action."""
#         return cls(action_type=ActionType.STOP)
    
#     @classmethod
#     def schedule_waypoint(cls, target_values: np.ndarray, target_time: float) -> "Action":
#         """Create a SCHEDULE_WAYPOINT action."""
#         return cls(
#             action_type=ActionType.SCHEDULE_WAYPOINT,
#             target_time=target_time,
#             data={"target_values": target_values}
#         )
    
#     @classmethod
#     def restart_put(cls, target_time: float) -> "Action":
#         """Create a RESTART_PUT action."""
#         return cls(
#             action_type=ActionType.RESTART_PUT,
#             target_time=target_time
#         )
    
#     # ========= Queue serialization ============
#     def to_queue_format(self) -> Dict[str, Any]:
#         """Convert to format suitable for SharedMemoryQueue."""
#         result = {
#             "cmd": self.action_type,
#             "target_time": self.target_time,
#         }
#         # Flatten data dict into the result
#         result.update(self.data)
#         return result

#     # ========= LeRobot format conversion ============
#     def to_lerobot_format(self):
#         '''
#         Convert action to lerobot format.
#         Only the data specified in settings.ACTION_CLASS is included.

#         Returns:
#             dict: Lerobot action format
#         '''
#         result = {
#             "action": [],
#         }

#         # # deep copy self.data
#         # tmp_data = self.data.copy()

#         # # filter out modules, actuators and data types that are not in settings.ACTION
#         # tmp_data = {k_module: v_module for k_module, v_module in tmp_data.items() if k_module in settings.ACTION}
#         # for k_module, v_module in tmp_data.items():
#         #     tmp_data[k_module] = {k_actuator: v_actuator for k_actuator, v_actuator in v_module.items() if k_actuator in settings.ACTION[k_module]}
#         #     for k_actuator, v_actuator in tmp_data[k_module].items():
#         #         tmp_data[k_module][k_actuator] = {k_data: v_data for k_data, v_data in v_actuator.items() if k_data in settings.ACTION[k_module][k_actuator]}
        
#         # for module_name, module_data in tmp_data.items():
#         #     for actuator_name, actuator_data in module_data.items():
#         #         for data_name, data_value in actuator_data.items():
#         #             if data_name in [ValueName.POS, ValueName.SPEED]:
#         #                 result["action"].append(data_value)
#         #             else:
#         #                 raise ValueError(f"Unsupported data type for actions: {data_name}")

#         # result["action"] = np.array(result["action"]).astype(np.float32)

#         return result

#     def __str__(self):
#         if self.action_type is not None:
#             return f"Action(type={self.action_type}, target_time={self.target_time}, data={self.data})"
        
#         s = "\n"
#         for module_name, module_action in self.data.items():
#             s += f"{module_name}:\n"
#             for actuator_name, actuator_action in module_action.items():
#                 for value_name, value in actuator_action.items():
#                     if value_name == ValueName.IMAGE:
#                         s += f"    {actuator_name}: {value_name} = <image>\n"
#                     else:
#                         s += f"    {actuator_name}: {value_name} = {value_str(value)}\n"
#         return s


#     def __class_getitem__(cls, module_name: str):
#         try:
#             field = cls.model_fields[module_name]
#         except KeyError:
#             raise KeyError(
#                 f"{cls.__name__} has no module '{module_name}'. "
#                 f"Available: {list(cls.model_fields.keys())}"
#             ) from None

#         ann = _unwrap_optional(field.annotation)

#         if not isinstance(ann, type) or not issubclass(ann, ModuleAction):
#             raise TypeError(
#                 f"{cls.__name__}.{module_name} is not a ModuleAction field; "
#                 f"it is annotated as {field.annotation!r}"
#             )

#         return ann
