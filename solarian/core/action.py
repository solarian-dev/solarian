import time
import numpy as np
from enum import IntEnum
from typing import Callable

from .data import Data, ImageData


class ActionType(IntEnum):
    """
    Standard action types supported by all modules.
    
    Modules can extend this by creating their own IntEnum that starts
    numbering from CUSTOM_START to avoid conflicts.
    """
    STOP = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2
    
    # Reserve range for future standard actions
    # Custom module actions should start from this value
    CUSTOM_START = 100


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

    def __init__(self, action_type: ActionType = ActionType.STOP, data: dict = {}):
        self.action_type = action_type
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
        return {'data': data, 'action_type': int(self.action_type)}

    # -- Compression -------------------------------------------------------

    def compress(self):
        for field in self._fields.values():
            field.compress()
        self.is_compressed = True

    def uncompress(self):
        for field in self._fields.values():
            field.uncompress()
        self.is_compressed = False


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


    def __init__(self, data: dict = {}):
        data = data if data is not None else {}
        self._module_actions = {}

        actual_cls = type(self)._resolve_class()
        module_actions_spec = actual_cls.get_module_actions_spec()
        valid_keys = set(module_actions_spec.keys())

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

            if set(data[key].keys()) != set(['data', 'action_type']):
                raise ValueError(
                    f"Invalid action value for '{key}' in {actual_cls.__name__}. "
                    f"Value must be a dictionary with keys 'data' and 'action_type'."
                )
            
            self._module_actions[key] = module_actions_spec[key].__class__(
                data=data[key]["data"],
                action_type=data[key]['action_type'],
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
        return self._module_actions[name]

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
        return cls(data)

    # -- Dict interface ----------------------------------------------------

    def to_dict(self):
        return {name: module_action.to_dict() for name, module_action in self._module_actions.items()}

    # -- Compression -------------------------------------------------------

    def compress(self):
        for module_action in self._module_actions.values():
            module_action.compress()
        self.is_compressed = True

    def uncompress(self):
        for module_action in self._module_actions.values():
            module_action.uncompress()
        self.is_compressed = False

