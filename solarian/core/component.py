from abc import ABC
from solarian.core.configs import ComponentConfig
from solarian.core.data import Data
from solarian.core.component_bus import ComponentBus


class Component(ABC):
    """Base class for components"""

    #: What this component *is*, for a consumer of a recorded session that
    #: knows nothing about the class. Declared on the class rather than
    #: inferred from the MRO so a plugin naming a new kind needs no change here.
    kind: str = "component"

    #: Unit per data-point name, where the component knows what its readings
    #: mean and the ``Data`` class cannot: a servo's ``pos`` is normalized
    #: travel, while its ``temperature`` is degrees whatever the servo.
    units: dict[str, str] = {}

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
    def unit_for(cls, component_name: str, data_name: str) -> str | None:
        """Unit of one data point, or ``None`` to defer to the ``Data`` class.

        ``component_name`` is passed because a single class can cover several
        units across its instances — a :class:`CartesianIKActuator` is metres
        as ``x`` and radians as ``rx``.
        """
        return cls.units.get(data_name)

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
    

