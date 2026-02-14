from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional, TYPE_CHECKING
from solarian.core.component import Component

if TYPE_CHECKING:
    from solarian.core.actuator_bus import ActuatorBus


class Actuator(Component):
    """
    Base class for actuators.
    
    Each actuator type (Feetech, Dynamixel, UR5, etc.) should subclass this
    and declare which bus type it uses via the `_bus_type` class variable.
    
    Actuator classes are registered via entry points and can be retrieved
    from the plugin registry:
    
        from solarian.plugins import actuators
        FeetechServo = actuators.get("feetech")
        
        class MyModule(Module):
            motor1: FeetechServo = FeetechServo(id=1, model="sts3215")
    
    The `_bus_type` string must match a registered entry point in
    `solarian.actuator_buses`.
    """
    
    # Subclasses MUST override this to specify which bus handles them
    _bus_type: ClassVar[Optional[str]] = None
    
    @classmethod
    def get_bus_type(cls) -> str:
        """
        Get the bus type identifier for this actuator class.
        
        Returns:
            The bus type string (e.g., 'feetech', 'dynamixel', 'ur5')
            
        Raises:
            NotImplementedError: If subclass didn't define _bus_type
        """
        if cls._bus_type is None:
            raise NotImplementedError(
                f"{cls.__name__} must define _bus_type class variable. "
                f"Example: _bus_type: ClassVar[str] = 'feetech'"
            )
        return cls._bus_type
    
    @classmethod
    def get_bus_class(cls) -> type["ActuatorBus"]:
        """
        Get the ActuatorBus class that handles this actuator type.
        
        Returns:
            The ActuatorBus subclass
            
        Raises:
            KeyError: If bus type not registered
        """
        from solarian.plugins import actuator_buses
        return actuator_buses.get_or_raise(cls.get_bus_type())
