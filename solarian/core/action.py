import time
import numpy as np
from enum import IntEnum
from typing import Callable

from .data import Data, ImageData


class ActionType:
    _registry: dict[int, str] = {}
    
    STOP = 0
    MODULE_ACTION = 1
    SCHEDULE_WAYPOINT = 2
    RESTART_PUT = 3
    # Controller-orchestrated skill lifecycle (not forwarded to modules).
    START_SKILL = 4
    STOP_SKILL = 5
    CUSTOM_START = 100

    def __init__(self, value: int = 0):
        self._value = int(value)

    def __int__(self):
        return self._value

    def __eq__(self, other):
        if isinstance(other, ActionType):
            return self._value == other._value
        if isinstance(other, int):
            return self._value == other
        return NotImplemented

    def __hash__(self):
        return hash(self._value)

    def __repr__(self):
        name = self._registry.get(self._value)
        if name:
            return f"ActionType({name}={self._value})"
        return f"ActionType({self._value})"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, val in vars(cls).items():
            if isinstance(val, int) and not name.startswith('_'):
                if val < ActionType.CUSTOM_START and name not in vars(ActionType):
                    raise ValueError(f"{name}={val} conflicts with reserved range")
                ActionType._registry[val] = f"{cls.__name__}.{name}"

    @classmethod
    def name_of(cls, value: int) -> str:
        return cls._registry.get(value, f"UNKNOWN({value})")



def action_handler(action_type: int):
    """
    Decorator to register a method as a handler for a specific action type.
    
    Usage:
        class MyModule(Module):
            @action_handler(BaseAction.SCHEDULE_WAYPOINT)
            def handle_schedule_waypoint(self, action: "Action") -> bool:
                # Handle the action
                return True  # Return False to stop the module
    
    Args:
        action_type: The action type value (from BaseAction or custom enum)
    """
    def decorator(method: Callable) -> Callable:
        setattr(method, '_handles_action', action_type)
        return method
    return decorator


# ---------------------------------------------------------------------------
# Schema primitives
# ---------------------------------------------------------------------------

class ActuatorField:
    """
    Links an Action field to a single data point on a Solarian robot actuator.

    The reference is a ``Data`` instance from the robot tree, e.g.
    ``MyRobot.arm.elbow_flex.pos``.  It carries schema information
    (dtype, shape) and is used at runtime to wrap raw values.

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
    def actuator_name(self):
        assert self.source.path is not None, "Data.path not set; was the descriptor accessed on its owner?"
        return self.source.path[1]

    @property
    def module_name(self):
        assert self.source.path is not None, "Data.path not set; was the descriptor accessed on its owner?"
        return self.source.path[0]

    
# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class ModuleAction:
    """
    Top-level ModuleAction: schema (class-level) + data container (instances).

    Subclass with ``ActuatorField`` instances::

        class MyModuleAction(ModuleAction):
            shoulder_pan = ActuatorField(MyRobot.arm.shoulder_pan.pos)
            shoulder_lift = ActuatorField(MyRobot.arm.shoulder_lift.pos)
            elbow_flex = ActuatorField(MyRobot.arm.elbow_flex.pos)
            wrist_flex = ActuatorField(MyRobot.arm.wrist_flex.pos)
            wrist_yaw = ActuatorField(MyRobot.arm.wrist_yaw.pos)
            wrist_roll = ActuatorField(MyRobot.arm.wrist_roll.pos)
            x = ActuatorField(MyRobot.arm.cartesian.x)
            y = ActuatorField(MyRobot.arm.cartesian.y)
            z = ActuatorField(MyRobot.arm.cartesian.z)
            rx = ActuatorField(MyRobot.arm.cartesian.rx)
            ry = ActuatorField(MyRobot.arm.cartesian.ry)
            rz = ActuatorField(MyRobot.arm.cartesian.rz)
            gripper_pos = ActuatorField(MyRobot.gripper.gripper_actuator.pos)

    When a concrete subclass is defined it auto-registers so that calls to
    ``ModuleAction.zeros()`` or ``ModuleAction.compress()`` on the base class
    transparently delegate to it.

    ``drop_timestamps`` is an optional flag for ``SCHEDULE_WAYPOINT``: ``True``
    ignores client timestamps (teleop drive), ``False`` honors them (policy
    chunks). ``None`` falls back to the module's ``drive_waypoints`` config.
    """

    _active_subclass = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ModuleAction._active_subclass = cls

    @classmethod
    def _resolve_class(cls):
        if cls is ModuleAction and ModuleAction._active_subclass is not None:
            return ModuleAction._active_subclass
        return cls

    # -- Init --------------------------------------------------------------

    #: Sentinel used on the wire / shared-memory queue when the sender did
    #: not set ``drop_timestamps`` (module falls back to ``cfg.drive_waypoints``).
    _DROP_TIMESTAMPS_UNSET = -1

    @staticmethod
    def _coerce_drop_timestamps(value) -> bool | None:
        """Normalize a ``drop_timestamps`` flag from bool / int / missing."""
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        try:
            iv = int(value)
        except (TypeError, ValueError):
            return None
        if iv < 0:
            return None
        return bool(iv)

    def __init__(
        self,
        action_type: "int | ActionType" = ActionType.STOP,
        data: dict = {},
        drop_timestamps: bool | None = None,
        params: dict | None = None,
    ):
        self.action_type = action_type
        self.drop_timestamps = self._coerce_drop_timestamps(drop_timestamps)
        self.params = dict(params or {})
        data = data if data is not None else {}
        self._fields = {}

        actual_cls = type(self)._resolve_class()
        actuator_fields_spec = actual_cls.get_actuator_fields_spec()
        valid_keys = set(actuator_fields_spec)

        for key in data:
            if key not in valid_keys:
                raise ValueError(
                    f"Invalid module action key '{key}' for {actual_cls.__name__}. "
                    f"Valid keys: {sorted(valid_keys)}"
                )
            if not isinstance(data[key], dict):
                raise ValueError(
                    f"Invalid module action value for '{key}' in {actual_cls.__name__}. "
                    f"Value must be a dictionary."
                )
            if not set(data[key].keys()).issubset(['value', 'timestamp']):
                raise ValueError(
                    f"Invalid module action value for '{key}' in {actual_cls.__name__}. "
                    f"Value must be a dictionary with keys 'value' and 'timestamp'."
                )

            self._fields[key] = actuator_fields_spec[key].source.duplicate(data[key])

        self.is_compressed = False
    # -- Schema introspection ----------------------------------------------

    @classmethod
    def get_actuator_fields_spec(cls):
        return {
            name: attr for name, attr in cls.__dict__.items()
            if isinstance(attr, ActuatorField)
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

    def items(self):
        return self._fields.items()

    def values(self):
        return self._fields

    @classmethod
    def __class_getitem__(cls, item):
        cls = cls._resolve_class()
        actuator_fields_spec = cls.get_actuator_fields_spec()
        if item in actuator_fields_spec:
            return actuator_fields_spec[item]
        return getattr(cls, item)
            
    # -- Factory -----------------------------------------------------------

    @classmethod
    def zeros(cls):
        cls = cls._resolve_class()
        data = {}
        for name, actuator_fields_spec in cls.get_actuator_fields_spec().items():
            data[name] = actuator_fields_spec.source.duplicate().to_dict()
        return cls(action_type=ActionType.STOP, data=data)

    # -- Dict interface ----------------------------------------------------

    def to_dict(self):
        data = {name: field.to_dict() for name, field in self._fields.items()}
        drop = (
            self._DROP_TIMESTAMPS_UNSET
            if self.drop_timestamps is None
            else (1 if self.drop_timestamps else 0)
        )
        return {
            'data': data,
            'action_type': int(self.action_type),
            'drop_timestamps': drop,
            'params': dict(self.params),
        }

    # -- Compression -------------------------------------------------------

    def compress(self):
        for field in self._fields.values():
            field.compress()
        self.is_compressed = True

    def uncompress(self):
        for field in self._fields.values():
            field.uncompress()
        self.is_compressed = False


class _LooseModuleAction:
    """Raw nested MODULE_ACTION payload when no project Action subclass is loaded."""

    def __init__(self, payload: dict):
        self._payload = dict(payload)

    def to_dict(self):
        return dict(self._payload)


class Action:
    """
    Top-level Action: schema (class-level) + data container (instances).
    """

    _active_subclass = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Action._active_subclass = cls

    @classmethod
    def _resolve_class(cls):
        if cls is Action and Action._active_subclass is not None:
            return Action._active_subclass
        return cls


    def __init__(self, action_type: "int | ActionType" = ActionType.STOP, data: dict = {}):
        self.action_type = action_type
        data = data if data is not None else {}
        self._module_actions = {}
        # Raw payload kept so non-MODULE_ACTION actions (e.g. skill-launch
        # actions) can carry arbitrary params through the wire.
        self.data = data

        actual_cls = type(self)._resolve_class()
        module_actions_spec = actual_cls.get_module_actions_spec()
        valid_keys = set(module_actions_spec.keys())

        if action_type == ActionType.MODULE_ACTION:
            if not valid_keys:
                for key, payload in data.items():
                    if not isinstance(payload, dict):
                        raise ValueError(
                            f"Invalid action value for '{key}' in {actual_cls.__name__}. "
                            f"Value must be a dictionary."
                        )
                    self._module_actions[key] = _LooseModuleAction(payload)
            else:
                for key in data:
                    if key not in valid_keys:
                        raise ValueError(
                            f"Invalid action key '{key}' for {actual_cls.__name__}. "
                            f"Valid keys: {sorted(valid_keys)}"
                        )

                    if not isinstance(data[key], dict):
                        raise ValueError(
                            f"Invalid action value for '{key}' in {actual_cls.__name__}. "
                            f"Value must be a dictionary."
                        )

                    keys = set(data[key].keys())
                    if not {'data', 'action_type'}.issubset(keys):
                        raise ValueError(
                            f"Invalid action value for '{key}' in {actual_cls.__name__}. "
                            f"Value must be a dictionary with keys 'data' and 'action_type'."
                        )
                    extra = keys - {'data', 'action_type', 'drop_timestamps', 'params'}
                    if extra:
                        raise ValueError(
                            f"Invalid action value for '{key}' in {actual_cls.__name__}. "
                            f"Unknown keys: {sorted(extra)}."
                        )

                    self._module_actions[key] = module_actions_spec[key].__class__(
                        data=data[key]["data"],
                        action_type=data[key]['action_type'],
                        drop_timestamps=data[key].get('drop_timestamps'),
                        params=data[key].get('params'),
                    )
        

    # -- Schema introspection ----------------------------------------------

    @classmethod
    def get_module_actions_spec(cls):
        return {
            name: attr for name, attr in cls.__dict__.items()
            if isinstance(attr, ModuleAction)
        }

    # -- Item / attribute access -------------------------------------------

    def __getattribute__(self, name):
        try:
            actions = object.__getattribute__(self, '_module_actions')
        except AttributeError:
            return object.__getattribute__(self, name)
        if name in actions:
            return actions[name]
        return object.__getattribute__(self, name)

    def __getitem__(self, name):
        try:
            return self._module_actions[name]
        except KeyError:
            actual_cls = type(self)._resolve_class()
            valid_keys = sorted(actual_cls.get_module_actions_spec().keys())
            raise KeyError(
                f"No action defined for module '{name}' on '{actual_cls.__name__}'. "
                f"Add a ModuleAction attribute named '{name}' to your Action "
                f"subclass. Defined module actions: {valid_keys}."
            ) from None

    def items(self):
        return self._module_actions.items()

    @classmethod
    def __class_getitem__(cls, item):
        cls = cls._resolve_class()
        module_actions_spec = cls.get_module_actions_spec()
        if item in module_actions_spec:
            return module_actions_spec[item]
        return getattr(cls, item)

    # -- Factory -----------------------------------------------------------

    @classmethod
    def zeros(cls):
        cls = cls._resolve_class()
        data = {}
        for name, module_action_spec in cls.get_module_actions_spec().items():
            data[name] = module_action_spec.zeros().to_dict()
        return cls(action_type=ActionType.MODULE_ACTION, data=data)

    # -- Dict interface ----------------------------------------------------

    def to_dict(self):
        if self.action_type == ActionType.MODULE_ACTION:
            data = {name: module_action.to_dict() for name, module_action in self._module_actions.items()}
        else:
            data = self.data
        return {"action_type": int(self.action_type), "data": data}

    # -- Compression -------------------------------------------------------

    def compress(self):
        for module_action in self._module_actions.values():
            module_action.compress()
        self.is_compressed = True

    def uncompress(self):
        for module_action in self._module_actions.values():
            module_action.uncompress()
        self.is_compressed = False

