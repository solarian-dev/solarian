from typing import ClassVar, Optional, TYPE_CHECKING
from solarian.core.component import Component

if TYPE_CHECKING:
    from solarian.core.actuator_bus import ActuatorBus


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
    
    
