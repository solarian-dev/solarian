from abc import ABC, abstractmethod
from typing import Any
from solarian.core.configs import ComponentConfig
from typing import ClassVar, Optional, TYPE_CHECKING
# from solarian.core.joint import Joint
# from solarian.core.body import Body
# from pydantic import BaseModel


class Component(ABC):
    """Base class for components"""

    # Subclasses MUST set this to their ComponentBus class (ActuatorBus, SensorBus, etc.)
    bus_class: ClassVar[type["ComponentBus"]]

    def __init__(self, cfg: ComponentConfig):
        self.cfg = cfg

    def needs_setup(self) -> bool:
        """Check if the component needs setup. Override in subclasses."""
        return False
    

