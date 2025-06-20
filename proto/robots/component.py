from abc import ABC, abstractmethod

from proto.common.configs import ComponentConfig


class Component(ABC):
    """Base class for components"""

    cfg: ComponentConfig
    name: str

    def __init__(self, cfg: ComponentConfig, name: str):
        self.cfg = cfg
        self.name = name
    
    @abstractmethod
    def connect(self):
        """Connect the component."""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect the component."""
        pass

    

    
