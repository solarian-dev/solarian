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
        return self.source.path[2]

    @property
    def sensor_name(self):
        return self.source.path[1]

    @property
    def module_name(self):
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
# Observation
# ---------------------------------------------------------------------------

class Observation:
    """
    Top-level observation: schema (class-level) + data container (instances).

    Subclass with ``SensorField`` and ``CalculatedField`` instances::

        class MyObservation(Observation):
            shoulder_pan_pos = SensorField(MyRobot.arm.shoulder_pan.pos)
            shoulder_pan_speed = SensorField(MyRobot.arm.shoulder_pan.speed)
            shoulder_lift_pos = SensorField(MyRobot.arm.shoulder_lift.pos)
            shoulder_lift_speed = SensorField(MyRobot.arm.shoulder_lift.speed)
            elbow_flex_pos = SensorField(MyRobot.arm.elbow_flex.pos)
            elbow_flex_speed = SensorField(MyRobot.arm.elbow_flex.speed)
            wrist_flex_pos = SensorField(MyRobot.arm.wrist_flex.pos)
            wrist_flex_speed = SensorField(MyRobot.arm.wrist_flex.speed)
            wrist_yaw_pos = SensorField(MyRobot.arm.wrist_yaw.pos)
            wrist_yaw_speed = SensorField(MyRobot.arm.wrist_yaw.speed)
            wrist_roll_pos = SensorField(MyRobot.arm.wrist_roll.pos)
            wrist_roll_speed = SensorField(MyRobot.arm.wrist_roll.speed)
            gripper_pos = SensorField(MyRobot.gripper.gripper_actuator.pos)
            image = SensorField(MyRobot.camera.camera.image)

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
        self.data = data if data is not None else {}
        self._fields = {}

        actual_cls = type(self)._resolve_class()
        sensor_fields = actual_cls.get_sensor_fields()
        calculated_fields = actual_cls.get_calculated_fields()
        valid_keys = set(sensor_fields) | set(calculated_fields)

        for key in self.data:
            if key not in valid_keys:
                raise ValueError(
                    f"Invalid observation key '{key}' for {actual_cls.__name__}. "
                    f"Valid keys: {sorted(valid_keys)}"
                )
            if not isinstance(self.data[key], dict):
                raise ValueError(
                    f"Invalid observation value for '{key}' in {actual_cls.__name__}. "
                    f"Value must be a dictionary."
                )
            if not set(self.data[key].keys()).issubset(['value', 'timestamp']):
                raise ValueError(
                    f"Invalid observation value for '{key}' in {actual_cls.__name__}. "
                    f"Value must be a dictionary with keys 'value' and 'timestamp'."
                )

        for name, sensor_fields in sensor_fields.items():
            if name in self.data:    
                self._fields[name] = sensor_fields.source.duplicate(self.data[name])
        for name, calculated_fields in calculated_fields.items():
            if name in self.data:
                self._fields[name] = calculated_fields.output.duplicate(self.data[name])
    # -- Schema introspection ----------------------------------------------

    @classmethod
    def get_sensor_fields(cls):
        return {
            name: attr for name, attr in cls.__dict__.items()
            if isinstance(attr, SensorField)
        }

    @classmethod
    def get_calculated_fields(cls):
        return {
            name: attr for name, attr in cls.__dict__.items()
            if isinstance(attr, CalculatedField)
        }

    def get_data(self):
        return self._fields

    # -- Item / attribute access -------------------------------------------

    def __getattribute__(self, name):
        try:
            return self._fields[name]
        except:
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                raise AttributeError(f"Observation has no attribute {name}, available attributes: {list(self._fields.keys())}")

    def __getitem__(self, name):
        return self._fields[name]

    @classmethod
    def __class_getitem__(cls, item):
        cls = cls._resolve_class()
        sensor_fields = cls.get_sensor_fields()
        calculated_fields = cls.get_calculated_fields()
        if item in sensor_fields:
            return sensor_fields[item]
        if item in calculated_fields:
            return calculated_fields[item]
        return getattr(cls, item)

    # -- Iteration ---------------------------------------------------------


            
    # -- Factory -----------------------------------------------------------

    @classmethod
    def zeros(cls):
        cls = cls._resolve_class()
        data = {}
        for name, sensor_fields in cls.get_sensor_fields().items():
            data[name] = sensor_fields.source.duplicate().to_dict()
        for name, calculated_fields in cls.get_calculated_fields().items():
            data[name] = calculated_fields.output.zeros().to_dict()
        return cls(data)

    # -- Dict interface ----------------------------------------------------

    def is_empty(self):
        if not self.data:
            return True
        return all(v is None for v in self.data.values())

    def to_dict(self):
        return self.data

    # -- Compression -------------------------------------------------------

    def compress(self):
        for field in self._fields.values():
            field.compress()
        return self

    def uncompress(self):
        for field in self._fields.values():
            field.uncompress()
        return self
