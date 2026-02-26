from solarian.core.component import Component
from solarian.core.configs import ActuatorConfig


class Actuator(Component):
    """
    Base class for actuators.
    
    Each actuator subclass must define `bus_class` pointing to an
    ActuatorBus subclass that handles its hardware communication.
    
    Example:
    
        from solarian_ur5.bus import UR5Bus
        
        class UR5JointActuator(Servo):
            bus_class = UR5Bus
            model: str = "ur5"
    """
    cfg: ActuatorConfig | None

    @classmethod
    def validate_act(cls, data: dict):
        for data_name, data_value in data.items():
            if data_name not in cls.data_names():
                raise ValueError(f"Invalid data name: '{data_name}' for actuator '{cls.__name__}'")
            data_class = getattr(cls, data_name)
            data_class.model_validate({"value": data_value})


    def __init__(self, cfg: ActuatorConfig | None = None):
        super().__init__(cfg)
    
    
