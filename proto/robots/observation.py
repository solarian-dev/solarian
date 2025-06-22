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

    def to_lerobot_format(self):
        data = {
            "observation.state": [],
        }
        
        for module_name, module_data in settings.OBSERVATION.items():
            for sensor_name, (data_name, data_shape) in module_data.items():
                if data_name in [ValueName.POS, ValueName.SPEED]:
                    data["observation.state"].append(self.data[module_name.value][sensor_name.value][data_name.value])
                elif data_name == ValueName.IMAGE:
                    data["observation.images." + sensor_name.value] = np.array(self.data[module_name.value][sensor_name.value][data_name.value])
                else:
                    raise ValueError(f"Unknown data type: {data_name}")

        data["observation.state"] = np.array(data["observation.state"]).astype(np.float32)
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