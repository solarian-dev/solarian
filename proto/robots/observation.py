from typing import Dict, Any
from pydantic import BaseModel
from pydantic import Field
from proto.common.configs import ValueName
from proto.conf import settings
from proto.common.utils import value_str
import numpy as np

class Observation(BaseModel):
    """Class for storing robot state observations including camera images and actuator positions"""

    data: Dict[str, Any]
    
    class Config:
        arbitrary_types_allowed = True  # Allow numpy arrays

         # "action": nparray shape 6
        # "observation.state": nparray shape 6
        # "observation.images.front": nparray shape (480, 640, 3)
        # "observation.images.wrist": nparray shape (480, 640, 3)
        # "task": string

    def flatten_keys(self):
        pass

    def to_lerobot_format(self):
        '''
        Convert observation to lerobot format.
        Only the data specified in settings.OBSERVATION is included.

        Returns:
            dict: Lerobot observation format
        '''


        data = {
            "observation.state": [],
            "observation.environment_state": [],
        }

        # deep copy self.data
        tmp_data = self.data.copy()

        # filter out modules, actuators and data types that are not in settings.OBSERVATION
        tmp_data = {k_module: v_module for k_module, v_module in tmp_data.items() if k_module in settings.OBSERVATION}
        for k_module, v_module in tmp_data.items():
            tmp_data[k_module] = {k_sensor: v_sensor for k_sensor, v_sensor in v_module.items() if k_sensor in settings.OBSERVATION[k_module]}
            for k_sensor, v_sensor in tmp_data[k_module].items():
                tmp_data[k_module][k_sensor] = {k_data: v_data for k_data, v_data in v_sensor.items() if k_data in settings.OBSERVATION[k_module][k_sensor]}

        for module_name, module_data in tmp_data.items():
            for sensor_name, sensor_data in module_data.items():
                for data_name, data_value in sensor_data.items():
                    if data_name in [ValueName.POS, ValueName.SPEED]:
                        data[settings.OBSERVATION[module_name][sensor_name][data_name].lerobot_group].append(data_value)
                    elif data_name == ValueName.BOUNDING_BOX:
                        data[settings.OBSERVATION[module_name][sensor_name][data_name].lerobot_group].extend(data_value)
                    elif data_name == ValueName.IMAGE:
                        data["observation.images." + sensor_name] = np.array(data_value)
                    elif data_name == ValueName.DEPTH_IMAGE:
                        data["observation.images." + sensor_name + "_depth"] = np.array(data_value)
                    else:
                        raise ValueError(f"Unknown data type: {data_name}")

        data["observation.state"] = np.array(data["observation.state"]).astype(np.float32)
        data["observation.environment_state"] = np.array(data["observation.environment_state"]).astype(np.float32)
        return data

    def __str__(self):
        s = "\n"
        for module_name, module_data in self.data.items():
            s += f"{module_name}:\n"
            for sensor_name, sensor_data in module_data.items():
                for value_name, value in sensor_data.items():
                    s += f"    {sensor_name}: {value_name} = {value_str(value)}\n"
        return s