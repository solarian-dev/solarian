from abc import ABC
from solarian.core.configs import ComponentConfig


class Component(ABC):
    """Base class for components"""

    def __init__(self, cfg: ComponentConfig | None = None):
        self.cfg = cfg

    def needs_setup(self) -> bool:
        """Check if the component needs setup. Override in subclasses."""
        return False
    

