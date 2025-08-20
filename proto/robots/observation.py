from typing import Dict, Any
from pydantic import BaseModel
from pydantic import Field
from proto.common.configs import ValueName
from proto.conf import settings
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
        data = {
            "observation.state": [],
            "observation.environment_state": [],
        }
        
        for module_name, module_data in self.data.items():
            for sensor_name, sensor_data in module_data.items():
                for data_name, data_value in sensor_data.items():
                    if data_name in [ValueName.POS, ValueName.SPEED]:
                        data[settings.OBS_LEROBOT_MAP[sensor_name].group].append(data_value)
                    elif data_name == ValueName.BOUNDING_BOX:
                        data[settings.OBS_LEROBOT_MAP[sensor_name].group].extend(data_value)
                    elif data_name == ValueName.IMAGE:
                        data["observation.images." + sensor_name] = np.array(data_value)
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
                    if value_name == ValueName.IMAGE:
                        s += f"    {sensor_name}: {value_name} = <image> ({type(value)})\n"
                    else:
                        s += f"    {sensor_name}: {value_name} = {value}\n"
        return s