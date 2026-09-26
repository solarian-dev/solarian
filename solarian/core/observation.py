import numpy as np
from .data import Data, ImageData, DataComputer, threshold_level as _threshold_level


# ---------------------------------------------------------------------------
# Schema primitives
# ---------------------------------------------------------------------------

class SensorField:
    """
    Links an observation field to a single data point on a robot sensor.

    The reference is a ``Data`` instance from the robot tree, e.g.
    ``MySolarianRobot.arm.elbow_flex.pos``.  It carries schema information
    (dtype, shape) and is used at runtime to wrap raw values.

    Supports attribute access to create ``FieldRef`` for ``CalculatedField``
    ``depends_on`` declarations.
    """

    def __init__(self, source: Data, *, mid=None, high=None, unit=None):
        self.source = source
        assert source.path is not None, "Data.path not set; was the descriptor accessed on its owner?"
        self._path = source.path
        # None = inherit from the Data class (e.g. TemperatureData.mid/high).
        self._mid = mid
        self._high = high
        self._unit = unit

    @property
    def mid(self):
        if self._mid is not None:
            return self._mid
        return getattr(type(self.source), "mid", None)

    @property
    def high(self):
        if self._high is not None:
            return self._high
        return getattr(type(self.source), "high", None)

    @property
    def unit(self) -> str | None:
        """Physical unit, most specific declaration first.

        The field overrides the component, which overrides the ``Data`` class.
        A ``PosData`` is metres on a Cartesian axis and normalized travel on a
        servo, so the component is usually the one that knows.
        """
        if self._unit is not None:
            return self._unit
        # None when the Data was built standalone rather than declared on a
        # component, which is how the recording tests construct their fixtures.
        if self.source.component_class is not None:
            unit = self.source.component_class.unit_for(self.sensor_name, self.data_name)
            if unit is not None:
                return unit
        return type(self.source).unit

    def threshold_level(self, value) -> str | None:
        """``"high"``, ``"mid"``, or ``None`` for a numeric observation value."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return _threshold_level(v, self.mid, self.high)
    
    @property
    def dtype(self):
        return self.source.dtype

    @property
    def shape(self):
        return self.source.shape

    @property
    def value(self):
        return self.source.value

    @property
    def data_class(self):
        return self.source.__class__

    @property
    def data_name(self):
        return self._path[2]

    @property
    def sensor_name(self):
        return self._path[1]

    @property
    def module_name(self):
        return self._path[0]


class CalculatedField:
    """
    Schema declaration for a computed observation field.

    Declares *what* to compute (Processor subclass), *where* it runs
    (computer), and *what* it depends on (inputs).

    ``processor`` must be a ``Processor`` subclass (the class, not an
    instance).  The component that owns the computation instantiates it
    via ``create_processors()``, passing ``inputs``, ``output``, and
    ``conf``::

        bbox = CalculatedField(
            inputs=[MyRobot.camera.sensor.rgb_image],
            processor=BBoxProcessor,
            computer=DataComputer.SERVER,
            output=BoundingBoxData(),
            conf='conf/bbox.yaml',
        )
    """

    def __init__(self, *, inputs: list[Data], computer: DataComputer,
                 processor: type, output: Data, conf: str | None = None,
                 unit: str | None = None):
        self.inputs = inputs
        self.processor = processor
        self.output = output
        self.conf = conf
        self._computer = computer
        self._unit = unit

    @property
    def unit(self) -> str | None:
        """Physical unit of the computed value.

        There is no component to ask — the processor decides what it produces —
        so this is either declared here or comes from the output ``Data`` class.
        """
        if self._unit is not None:
            return self._unit
        return type(self.output).unit

    @property
    def dtype(self):
        return self.output.dtype

    @property
    def shape(self):
        return self.output.shape

    @property
    def computer(self):
        return self._computer

    @property
    def value(self):
        return self.output.value

    @property
    def data_class(self):
        return self.output.__class__
        


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

    def __setitem__(self, name, value):
        self._fields[name] = value

    def items(self):
        """Iterate ``(field_name, Data)`` pairs, mirroring
        :meth:`Observation.items`.  ``Data`` is whatever the
        ``SensorField`` / ``CalculatedField`` declared (e.g.
        :class:`ImageData`, :class:`PosData`)."""
        return self._fields.items()

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


    @classmethod
    def create_processors(cls, computer: DataComputer) -> dict:
        """Instantiate processors for all CalculatedFields matching *computer*.

        Each processor class declared in a ``CalculatedField`` is
        instantiated with the field's ``inputs``, ``output``, and
        ``conf``.

        Returns:
            ``{field_name: Processor instance}``
        """
        result = {}
        for field_name, cf in cls.get_calculated_fields_spec().items():
            if cf.computer != computer:
                continue
            result[field_name] = cf.processor(
                inputs=cf.inputs,
                output=cf.output,
                conf=cf.conf,
            )
        return result


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
        cls._validate_module_obs_keys()

    @classmethod
    def _validate_module_obs_keys(cls):
        """Verify that each ModuleObs attribute name matches the Robot module
        name referenced by its SensorFields.  Raises ``TypeError`` at
        class-definition time when there is a mismatch."""
        for attr_name, attr_value in cls.__dict__.items():
            if not isinstance(attr_value, ModuleObs):
                continue

            module_obs_cls = type(attr_value)
            for field_name, sf in module_obs_cls.get_sensor_fields_spec().items():
                if sf.source.path is None:
                    continue
                if sf.module_name != attr_name:
                    raise TypeError(
                        f"{cls.__name__}.{attr_name}: SensorField '{field_name}' "
                        f"references module '{sf.module_name}', but the Observation "
                        f"attribute is named '{attr_name}'. Observation attribute "
                        f"names must match the module names declared on the Robot "
                        f"subclass."
                    )

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
        if name not in self._module_obs:
            return None
        return self._module_obs[name]

    def __setitem__(self, name, value):
        self._module_obs[name] = value


    def items(self):
        return self._module_obs.items()


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

    def is_empty(self):
        return len(self._module_obs) == 0

    # -- Compression -------------------------------------------------------

    def compress(self):
        for module_obs in self._module_obs.values():
            module_obs.compress()
        self.is_compressed = True

    def uncompress(self):
        for module_obs in self._module_obs.values():
            module_obs.uncompress()
        self.is_compressed = False
        

    @classmethod
    def create_processors(cls, computer: DataComputer) -> dict:
        """Instantiate processors for all CalculatedFields matching *computer*.

        Walks every ``ModuleObs`` and delegates to
        ``ModuleObs.create_processors()``.

        Returns:
            ``{(module_name, field_name): Processor instance}``
        """
        cls = cls._resolve_class()
        result = {}
        for module_name, module_obs in cls.get_module_obs_spec().items():
            for field_name, proc in type(module_obs).create_processors(computer).items():
                result[(module_name, field_name)] = proc
        return result