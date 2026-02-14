from typing import Dict, Any
from pydantic import BaseModel
from pydantic import Field
from solarian.utils import value_str
import time
import numpy as np
from typing import Any, Union, get_args, get_origin


import time

class ModuleObservation(BaseModel):
    timestamp: float = Field(default_factory=time.time)

    @classmethod
    def keys(cls):
        return list(cls.model_fields.keys())

    def __getitem__(self, item: str):
        return getattr(self, item)

    def get(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def set(self, key: str, value: Any):
        setattr(self, key, value)

    def __iter__(self):
        return iter(self.model_fields.keys())

    def items(self):
        for key in self:
            yield key, getattr(self, key)

    def __class_getitem__(cls, sensor_name: str):
        try:
            return cls.model_fields[sensor_name]  # pydantic FieldInfo-like
        except KeyError:
            raise KeyError(
                f"{cls.__name__} has no field '{sensor_name}'. "
                f"Available: {list(cls.model_fields.keys())}"
            ) from None


def _unwrap_optional(t: Any) -> Any:
    # Optional[X] is Union[X, NoneType]
    if get_origin(t) is Union:
        args = [a for a in get_args(t) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return t


class Observation(BaseModel):
    """Base class for robot observations. Subclass this in your project settings
    to define project-specific observation fields."""

    @classmethod
    def keys(cls):
        return list(cls.model_fields.keys())

    def __getitem__(self, item: str):
        return getattr(self, item)

    def get(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def set(self, key: str, value: Any):
        setattr(self, key, value)

    def __iter__(self):
        return iter(self.model_fields.keys())

    def items(self):
        for key in self:
            yield key, getattr(self, key)

    def __class_getitem__(cls, module_name: str):
        try:
            field = cls.model_fields[module_name]
        except KeyError:
            raise KeyError(
                f"{cls.__name__} has no module '{module_name}'. "
                f"Available: {list(cls.model_fields.keys())}"
            ) from None

        ann = _unwrap_optional(field.annotation)

        if not isinstance(ann, type) or not issubclass(ann, ModuleObservation):
            raise TypeError(
                f"{cls.__name__}.{module_name} is not a ModuleObservation field; "
                f"it is annotated as {field.annotation!r}"
            )

        return ann

    


# class Observation(BaseModel):
#     """Class for storing robot state observations including camera images and actuator positions"""

#     data: Dict[ModuleName, Dict[SensorName, Dict[ValueName, Any]]]
    
#     class Config:
#         arbitrary_types_allowed = True  # Allow numpy arrays

#          # "action": nparray shape 6
#         # "observation.state": nparray shape 6
#         # "observation.images.front": nparray shape (480, 640, 3)
#         # "observation.images.wrist": nparray shape (480, 640, 3)
#         # "task": string

#     def flatten_keys(self):
#         pass

#     def to_lerobot_format(self):
#         '''
#         Convert observation to lerobot format.
#         Only the data specified in settings.OBSERVATION is included.

#         Returns:
#             dict: Lerobot observation format
#         '''


#         data = {
#             "observation.state": [],
#             "observation.environment_state": [],
#         }

#         # deep copy self.data
#         tmp_data = self.data.copy()

#         # filter out modules, actuators and data types that are not in settings.OBSERVATION
#         tmp_data = {k_module: v_module for k_module, v_module in tmp_data.items() if k_module in settings.OBSERVATION}
#         for k_module, v_module in tmp_data.items():
#             tmp_data[k_module] = {k_sensor: v_sensor for k_sensor, v_sensor in v_module.items() if k_sensor in settings.OBSERVATION[k_module]}
#             for k_sensor, v_sensor in tmp_data[k_module].items():
#                 tmp_data[k_module][k_sensor] = {k_data: v_data for k_data, v_data in v_sensor.items() if k_data in settings.OBSERVATION[k_module][k_sensor]}

#         for module_name, module_data in tmp_data.items():
#             for sensor_name, sensor_data in module_data.items():
#                 for data_name, data_value in sensor_data.items():
#                     if data_name in [ValueName.POS, ValueName.SPEED]:
#                         data[settings.OBSERVATION[module_name][sensor_name][data_name].lerobot_group].append(data_value)
#                     elif data_name == ValueName.BOUNDING_BOX:
#                         data[settings.OBSERVATION[module_name][sensor_name][data_name].lerobot_group].extend(data_value)
#                     elif data_name == ValueName.IMAGE:
#                         data["observation.images." + sensor_name] = np.array(data_value)
#                     elif data_name == ValueName.DEPTH_IMAGE:
#                         data["observation.images." + sensor_name + "_depth"] = np.array(data_value)
#                     else:
#                         raise ValueError(f"Unknown data type: {data_name}")

#         data["observation.state"] = np.array(data["observation.state"]).astype(np.float32)
#         data["observation.environment_state"] = np.array(data["observation.environment_state"]).astype(np.float32)
#         return data

#     def __str__(self):
#         s = "\n"
#         for module_name, module_data in self.data.items():
#             s += f"{module_name}:\n"
#             for sensor_name, sensor_data in module_data.items():
#                 for value_name, value in sensor_data.items():
#                     s += f"    {sensor_name}: {value_name} = {value_str(value)}\n"
#         return s


# def create_example_observation() -> Observation:
#         """
#         Create an example Observation to define the ring buffer schema.
        
#         Override in subclass if you need custom data fields in observations.
#         The example observation should have the same structure as observations
#         that will be produced by this module.
        
#         Returns:
#             Example Observation with all fields that will be used
#         """
#         # Build example data structure from settings.OBSERVATION
#         example_data = {
#         }
        
        
#         for module_name, module_spec in settings.OBSERVATION.items():
#             for sensor_name, sensor_spec in module_spec.items():
#                 example_data[module_name] = {sensor_name: {}}
#                 for data_name, data_spec in sensor_spec.items():
#                     if data_spec.computer == DataComputer.ROBOT:
#                         if data_name in [ValueName.POS, ValueName.SPEED, ValueName.TORQUE_ENABLE]:
#                             example_data[module_name][sensor_name][data_name] = 0.0
#                         elif data_name in [ValueName.IMAGE, ValueName.DEPTH_IMAGE]:
#                             example_data[module_name][sensor_name][data_name] = np.zeros(data_spec.shape, dtype=np.uint8)
#                         elif data_name == ValueName.BOUNDING_BOX:
#                             example_data[module_name][sensor_name][data_name] = np.zeros(data_spec.shape, dtype=np.float32)
#                         else:
#                             raise ValueError(f"Unknown data type: {data_name}")
        
#         return Observation(data=example_data)