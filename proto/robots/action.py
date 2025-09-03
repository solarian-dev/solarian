from typing import Dict, Any
from pydantic import BaseModel
from proto.common.configs import ValueName
import numpy as np
from proto.conf import settings


class Action(BaseModel):
    """Class for storing robot actions"""
    
    data: Dict[str, Any]
    
    class Config:
        arbitrary_types_allowed = True  # Allow numpy arrays

        # "action": nparray shape 6
        # "observation.state": nparray shape 6
        # "observation.images.front": nparray shape (480, 640, 3)
        # "observation.images.wrist": nparray shape (480, 640, 3)
        # "task": string

    def to_lerobot_format(self):
        '''
        Convert action to lerobot format.
        Only the data specified in settings.ACTION is included.

        Returns:
            dict: Lerobot action format
        '''

        data = {
            "action": [],
        }

        # deep copy self.data
        tmp_data = self.data.copy()

        # filter out modules, actuators and data types that are not in settings.ACTION
        tmp_data = {k_m: v_m for k_m, v_m in tmp_data.items() if k_m in settings.ACTION}
        for k_m, v_m in tmp_data.items():
            tmp_data[k_m] = {k_s: v_s for k_s, v_s in v_m.items() if k_s in settings.ACTION[k_m]}
            for k_s, v_s in tmp_data[k_m].items():
                tmp_data[k_m][k_s] = {k_d: v_d for k_d, v_d in v_s.items() if k_d in settings.ACTION[k_m][k_s]}
        
        for module_name, module_data in tmp_data.items():
            for actuator_name, actuator_data in module_data.items():
                for data_name, data_value in actuator_data.items():
                    if data_name in [ValueName.POS, ValueName.SPEED]:
                        data["action"].append(data_value)
                    else:
                        raise ValueError(f"Unsupported data type for actions: {data_name}")

        data["action"] = np.array(data["action"]).astype(np.float32)

        return data

    def __str__(self):
        s = "\n"
        for module_name, module_action in self.data.items():
            s += f"{module_name}:\n"
            for actuator_name, actuator_action in module_action.items():
                for value_name, value in actuator_action.items():
                    if value_name == ValueName.IMAGE:
                        s += f"    {actuator_name}: {value_name} = <image>\n"
                    else:
                        s += f"    {actuator_name}: {value_name} = {value}\n"
        return s


