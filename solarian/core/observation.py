import numpy as np
from typing import Callable
from .data import Data, ImageData, DataComputer


# ---------------------------------------------------------------------------
# Schema primitives
# ---------------------------------------------------------------------------

class SensorField:
    """
    Links an observation field to a single data point on a robot sensor.

    The reference is a ``Data`` instance from the robot tree, e.g.
    ``MySolarianRobot.arm.elbow_flex.pos``.  It carries schema information
    (dtype, shape, computer) and is used at runtime to wrap raw values.

    Supports attribute access to create ``FieldRef`` for ``CalculatedField``
    ``depends_on`` declarations.
    """

    def __init__(self, source: Data):
        self.source = source
    
    @property
    def dtype(self):
        return self.source.dtype

    @property
    def shape(self):
        return self.source.shape

    @property
    def computer(self):
        return self.source.computer

    @property
    def value(self):
        return self.source.value

    @property
    def data_class(self):
        return self.source.__class__

    @property
    def data_name(self):
        assert self.source.path is not None, "Data.path not set; was the descriptor accessed on its owner?"
        return self.source.path[2]

    @property
    def sensor_name(self):
        assert self.source.path is not None, "Data.path not set; was the descriptor accessed on its owner?"
        return self.source.path[1]

    @property
    def module_name(self):
        assert self.source.path is not None, "Data.path not set; was the descriptor accessed on its owner?"
        return self.source.path[0]


class CalculatedField:
    """
    Computed observation field with declared dependencies and a processor.

    Has the same schema attributes as ``Data`` (``dtype``, ``shape``,
    ``computer``) so the observation schema is self-describing.
    """

    def __init__(self, *, inputs: list[SensorField], processor: Callable, output: Data):
        self.depends_on = inputs
        self.processor = processor
        self.output = output

    def compute(self, **inputs):
        return self.processor(**inputs)


# ---------------------------------------------------------------------------
# ModuleObs
# ---------------------------------------------------------------------------

class ModuleObs:
    """
    Groups sensor fields for a single module: schema (class-level) + data container (instances).

    Subclass with ``SensorField`` and ``CalculatedField`` instances::

        class MyArmObs(ModuleObs):
            shoulder_pan_pos = SensorField(MyRobot.arm.shoulder_pan.pos)
            shoulder_lift_pos = SensorField(MyRobot.arm.shoulder_lift.pos)
            elbow_flex_pos = SensorField(MyRobot.arm.elbow_flex.pos)

    When a concrete subclass is defined it auto-registers so that calls to
    ``ModuleObs.zeros()`` on the base class transparently delegate to it.
    """

    _active_subclass = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ModuleObs._active_subclass = cls

    @classmethod
    def _resolve_class(cls):
        if cls is ModuleObs and ModuleObs._active_subclass is not None:
            return ModuleObs._active_subclass
        return cls

    # -- Init --------------------------------------------------------------

    def __init__(self, data=None):
        data = data if data is not None else {}
        self._fields = {}

        actual_cls = type(self)._resolve_class()
        sensor_fields_spec = actual_cls.get_sensor_fields_spec()
        calculated_fields_spec = actual_cls.get_calculated_fields_spec()
        valid_keys = set(sensor_fields_spec) | set(calculated_fields_spec)

        for key in data:
            if key not in valid_keys:
                raise ValueError(
                    f"Invalid module obs key '{key}' for {actual_cls.__name__}. "
                    f"Valid keys: {sorted(valid_keys)}"
                )
            if not isinstance(data[key], dict):
                raise ValueError(
                    f"Invalid module obs value for '{key}' in {actual_cls.__name__}. "
                    f"Value must be a dictionary."
                )
            if not set(data[key].keys()).issubset(['value', 'timestamp']):
                raise ValueError(
                    f"Invalid module obs value for '{key}' in {actual_cls.__name__}. "
                    f"Value must be a dictionary with keys 'value' and 'timestamp'."
                )

            if key in sensor_fields_spec:
                self._fields[key] = sensor_fields_spec[key].source.duplicate(data[key])
            elif key in calculated_fields_spec:
                self._fields[key] = calculated_fields_spec[key].output.duplicate(data[key])

        self.is_compressed = False

    # -- Schema introspection ----------------------------------------------

    @classmethod
    def get_sensor_fields_spec(cls):
        return {
            name: attr for name, attr in cls.__dict__.items()
            if isinstance(attr, SensorField)
        }

    @classmethod
    def get_calculated_fields_spec(cls):
        return {
            name: attr for name, attr in cls.__dict__.items()
            if isinstance(attr, CalculatedField)
        }

    # -- Item / attribute access -------------------------------------------

    def __getattribute__(self, name):
        try:
            fields = object.__getattribute__(self, '_fields')
        except AttributeError:
            return object.__getattribute__(self, name)
        if name in fields:
            return fields[name]
        return object.__getattribute__(self, name)

    def __getitem__(self, name):
        return self._fields[name]

    @classmethod
    def __class_getitem__(cls, item):
        cls = cls._resolve_class()
        sensor_fields_spec = cls.get_sensor_fields_spec()
        calculated_fields_spec = cls.get_calculated_fields_spec()
        if item in sensor_fields_spec:
            return sensor_fields_spec[item]
        if item in calculated_fields_spec:
            return calculated_fields_spec[item]
        return getattr(cls, item)

    # -- Factory -----------------------------------------------------------

    @classmethod
    def zeros(cls):
        cls = cls._resolve_class()
        data = {}
        for name, sf_spec in cls.get_sensor_fields_spec().items():
            data[name] = sf_spec.source.duplicate().to_dict()
        for name, cf_spec in cls.get_calculated_fields_spec().items():
            data[name] = cf_spec.output.zeros().to_dict()
        return cls(data)

    # -- Dict interface ----------------------------------------------------

    def to_dict(self):
        return {name: field.to_dict() for name, field in self._fields.items()}

    # -- Compression -------------------------------------------------------

    def compress(self):
        for field in self._fields.values():
            field.compress()
        self.is_compressed = True

    def uncompress(self):
        for field in self._fields.values():
            field.uncompress()
        self.is_compressed = False


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation:
    """
    Top-level Observation: groups ``ModuleObs`` instances.

    Subclass with ``ModuleObs`` instances::

        class MyObservation(Observation):
            arm = MyArmObs()
            gripper = MyGripperObs()
            gripper_camera = MyCameraObs()

    When a concrete subclass is defined it auto-registers so that calls to
    ``Observation.zeros()`` or ``Observation.compress()`` on the base class
    transparently delegate to it.
    """

    _active_subclass = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Observation._active_subclass = cls

    @classmethod
    def _resolve_class(cls):
        if cls is Observation and Observation._active_subclass is not None:
            return Observation._active_subclass
        return cls

    # -- Init --------------------------------------------------------------

    def __init__(self, data=None):
        data = data if data is not None else {}
        self._module_obs = {}
        
        actual_cls = type(self)._resolve_class()
        module_obs_spec = actual_cls.get_module_obs_spec()
        valid_keys = set(module_obs_spec.keys())

        for key in data:
            if key not in valid_keys:
                raise ValueError(
                    f"Invalid observation key '{key}' for {actual_cls.__name__}. "
                    f"Valid keys: {sorted(valid_keys)}"
                )
            if not isinstance(data[key], dict):
                raise ValueError(
                    f"Invalid observation value for '{key}' in {actual_cls.__name__}. "
                    f"Value must be a dictionary."
                )

            self._module_obs[key] = module_obs_spec[key].__class__(data=data[key])
        self.is_compressed = False


    # -- Schema introspection ----------------------------------------------

    @classmethod
    def get_module_obs_spec(cls):
        cls = cls._resolve_class()
        return {
            name: attr for name, attr in cls.__dict__.items()
            if isinstance(attr, ModuleObs)
        }

    # -- Item / attribute access -------------------------------------------

    def __getattribute__(self, name):
        try:
            obs = object.__getattribute__(self, '_module_obs')
        except AttributeError:
            return object.__getattribute__(self, name)
        if name in obs:
            return obs[name]
        return object.__getattribute__(self, name)

    def __getitem__(self, name):
        return self._module_obs[name] 

    @classmethod
    def __class_getitem__(cls, item):
        cls = cls._resolve_class()
        module_obs_spec = cls.get_module_obs_spec()
        if item in module_obs_spec:
            return module_obs_spec[item]
        return getattr(cls, item)

    # -- Factory -----------------------------------------------------------

    @classmethod
    def zeros(cls):
        cls = cls._resolve_class()
        data = {}
        for name, module_obs_spec in cls.get_module_obs_spec().items():
            data[name] = module_obs_spec.zeros().to_dict()
        return cls(data)

    # -- Dict interface ----------------------------------------------------

    def to_dict(self):
        return {name: module_obs.to_dict() for name, module_obs in self._module_obs.items()}

    # -- Compression -------------------------------------------------------

    def compress(self):
        for module_obs in self._module_obs.values():
            module_obs.compress()
        self.is_compressed = True

    def uncompress(self):
        for module_obs in self._module_obs.values():
            module_obs.uncompress()
        self.is_compressed = False
        
