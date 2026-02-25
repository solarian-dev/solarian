from solarian.core.component import Component
from solarian.core.data import ImageData
import cv2
import numpy as np

class Sensor(Component):
    
    @classmethod
    def validate_obs(cls, data: dict):
        for data_name, data_value in data.items():
            if data_name not in cls.data_names():
                raise ValueError(f"Invalid data name: '{data_name}' for sensor '{cls.__name__}'")
            data_class = getattr(cls, data_name)
            if isinstance(data_class, ImageData) and type(data_value) == bytes:
                data_value = cv2.imdecode(np.frombuffer(data_value, dtype=data_class.dtype), cv2.IMREAD_COLOR)
            data_class.model_validate({"value": data_value})
            data[data_name] = data_value
        return data
    

