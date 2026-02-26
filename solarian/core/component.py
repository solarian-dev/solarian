from abc import ABC
from solarian.core.configs import ComponentConfig
from solarian.core.data import Data
from solarian.core.component_bus import ComponentBus


class Component(ABC):
    """Base class for components"""

    def __set_name__(self, owner, name):
        self.name = name
        # print("Component name:", name)
        # print("Component owner:", owner)

    def __get__(self, instance, owner):
        # print("Component called __get__")
        # print("Component instance:", instance)
        # print("Component owner:", owner)
        object.__setattr__(self, 'parent', instance)
        return self

    @classmethod
    def data_names(cls):
        return [name for name in cls.__dict__ if isinstance(getattr(cls, name), Data)]

    @classmethod
    def data_types(cls):
        return {name: type(getattr(cls, name)) for name in cls.data_names()}

    bus_class: type[ComponentBus]

    @property
    def data_points(self):
        cls = type(self)
        return {name: getattr(cls, name) for name in cls.data_names()}

    def __init__(self, cfg: ComponentConfig | None = None):
        self.cfg = cfg
        self.name = None
        self.parent = None

    def needs_setup(self) -> bool:
        """Check if the component needs setup. Override in subclasses."""
        return False

    @classmethod
    def __class_getitem__(cls, item):
        return getattr(cls, item)

    def __getitem__(self, item):
        return getattr(self, item)
    

