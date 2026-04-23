"""
Shared-memory FIFO queues for actions.

Two classes are exposed:

* :class:`ModuleActionQueue` — one queue per robot module.  Stores
  :class:`ModuleAction` instances with a fixed schema derived from an
  example action (flat ``dict[str, np.ndarray | number]``).  Used by the
  robot to deliver per-module actuator commands to modules with zero
  pickle overhead.

* :class:`ActionQueue` — one queue for the top-level :class:`Action`
  objects that flow from the think process to the controller's
  ``control_step``/``control_loop``.  Because these actions may carry
  arbitrary skill-launch ``params`` (non-``MODULE_ACTION`` types), the
  payload is serialized with ``msgpack`` into a fixed-size ``uint8``
  buffer stored in the underlying :class:`SharedMemoryQueue`.
"""

from typing import Dict, Any, List, Optional, Union
import numbers
from multiprocessing.managers import SharedMemoryManager

import msgpack
import numpy as np

from solarian.utils.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty, Full
from solarian.core.action import Action, ActionType, ModuleAction


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary into a flat dictionary with dot-notation keys.
    
    Example:
        {"a": 1, "b": {"c": 2, "d": 3}} -> {"a": 1, "b.c": 2, "b.d": 3}
    
    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys (used in recursion)
        sep: Separator between nested keys
        
    Returns:
        Flattened dictionary with only numbers and np.arrays as values
        
    Raises:
        TypeError: If a value is not a number, np.ndarray, or dict
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep))
        elif isinstance(v, (np.ndarray, numbers.Number)):
            items[new_key] = v
        elif v is None:
            # Store None as a special marker (NaN for float)
            items[new_key] = float('nan')
        else:
            raise TypeError(
                f"Unsupported type {type(v)} for key '{new_key}'. "
                f"Values must be numbers, np.ndarray, dict, or None."
            )
    return items


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Unflatten a flat dictionary with dot-notation keys back to nested structure.
    
    Example:
        {"a": 1, "b.c": 2, "b.d": 3} -> {"a": 1, "b": {"c": 2, "d": 3}}
    
    Args:
        d: Flattened dictionary
        sep: Separator used between nested keys
        
    Returns:
        Nested dictionary
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Convert NaN back to None
        if isinstance(value, float) and np.isnan(value):
            value = None
        
        current[parts[-1]] = value
    
    return result


class ModuleActionQueue:
    """
    A shared memory queue for :class:`ModuleAction` objects (one queue per module).

    Converts ``ModuleAction`` objects to flattened dictionaries for efficient
    storage in :class:`SharedMemoryQueue`. No pickle serialization overhead.

    Requirements for action data:
    - Values must be numbers, np.ndarray, or nested dicts with same constraints
    - The schema (keys and array shapes) must be consistent across all actions

    Args:
        shm_manager: SharedMemoryManager instance
        example_action: An example ``ModuleAction`` to define the schema
        buffer_size: Number of actions the queue can hold
    """
    
    PRESENT_PREFIX = '_present'

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        example_action: ModuleAction,
        buffer_size: int
    ):
        self.buffer_size = buffer_size
        self._module_action_class = type(example_action)

        actual_cls = self._module_action_class._resolve_class()
        self._actuator_field_names = list(actual_cls.get_actuator_fields_spec().keys())

        # Build the full-schema flat dict from the example (should come from
        # zeros() so every field is present).  Append one int8 presence flag
        # per actuator field so the consumer knows which fields were real.
        example_dict = flatten_dict(example_action.to_dict())
        for name in self._actuator_field_names:
            example_dict[f'{self.PRESENT_PREFIX}.{name}'] = np.int8(1)
        self._template_flat_dict = example_dict

        self._queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example_dict,
            buffer_size=buffer_size
        )
    
    @classmethod
    def create_from_example(
        cls,
        shm_manager: SharedMemoryManager,
        example_action,
        buffer_size: int = 1024
    ) -> "ModuleActionQueue":
        """
        Create a :class:`ModuleActionQueue` from an example ``ModuleAction``.
        """
        return cls(shm_manager, example_action, buffer_size)
    
    def _action_to_flat_dict(self, action: ModuleAction) -> Dict[str, Union[np.ndarray, numbers.Number]]:
        """Convert a ModuleAction to a flattened dictionary.

        Starts from the zero-filled template so every put produces the
        same key set.  Presence flags are set to 1 only for fields that
        were actually populated in the action.
        """
        if action is None:
            return {}

        action_dict = action.to_dict()
        present_fields = set(action_dict.get('data', {}).keys())
        flat = flatten_dict(action_dict)

        full = self._template_flat_dict.copy()
        full.update(flat)

        for name in self._actuator_field_names:
            full[f'{self.PRESENT_PREFIX}.{name}'] = np.int8(1 if name in present_fields else 0)

        return full

    @staticmethod
    def _to_native(v):
        """Convert 0-d numpy arrays back to Python scalars."""
        if isinstance(v, np.ndarray) and v.ndim == 0:
            return v.item()
        return v

    def _flat_dict_to_action(self, flat_dict: Dict[str, Any]) -> ModuleAction:
        """Convert a flattened dictionary back to a ModuleAction.

        Only fields whose presence flag is 1 are included in the
        reconstructed action.
        """
        present_fields = {
            name for name in self._actuator_field_names
            if int(flat_dict.get(f'{self.PRESENT_PREFIX}.{name}', 0)) == 1
        }

        # Strip presence flags and convert 0-d arrays to native scalars
        data_flat = {
            k: self._to_native(v) for k, v in flat_dict.items()
            if not k.startswith(self.PRESENT_PREFIX)
        }
        nested = unflatten_dict(data_flat)

        filtered_data = {
            k: v for k, v in nested.get('data', {}).items()
            if k in present_fields
        }

        action_type = ActionType(int(nested['action_type']))
        return self._module_action_class(action_type=action_type, data=filtered_data)
    
    def qsize(self) -> int:
        """Return the approximate number of actions in the queue."""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Return True if the queue is empty."""
        return self._queue.empty()
    
    def clear(self):
        """Clear all actions from the queue."""
        self._queue.clear()
    
    def put(self, action: ModuleAction) -> None:
        """
        Put an action into the queue.
        
        Args:
            action: An ModuleAction instance to enqueue
            
        Raises:
            Full: If the queue is full
            TypeError: If the action contains unsupported types
        """
        self._queue.put(self._action_to_flat_dict(action))
    
    def get(self) -> ModuleAction:
        """
        Get an action from the queue.
        
        Returns:
            The next ModuleAction in the queue
            
        Raises:
            Empty: If the queue is empty
        """
        return self._flat_dict_to_action(self._queue.get())
    
    def get_k(self, k: int) -> List[ModuleAction]:
        """
        Get up to k actions from the queue.
        
        Args:
            k: Maximum number of actions to retrieve
            
        Returns:
            List of ModuleAction instances
            
        Raises:
            Empty: If the queue is empty
        """
        flat_results = self._queue.get_k(k)
        
        # Convert each row back to an Action
        actions = []
        n_items = len(flat_results[next(iter(flat_results))])  # Get count from first key
        
        for i in range(n_items):
            flat_dict = {key: values[i] for key, values in flat_results.items()}
            actions.append(self._flat_dict_to_action(flat_dict))
        
        return actions
    
    def get_all(self) -> List:
        """
        Get all actions from the queue.
        
        Returns:
            List of all ModuleAction instances in the queue
            
        Raises:
            Empty: If the queue is empty
        """
        flat_results = self._queue.get_all()
        
        # Convert each row back to an ModuleAction
        actions = []
        n_items = len(flat_results[next(iter(flat_results))])
        
        for i in range(n_items):
            flat_dict = {key: values[i] for key, values in flat_results.items()}
            actions.append(self._flat_dict_to_action(flat_dict))
        
        return actions


class ActionQueue:
    """Shared-memory FIFO queue for top-level :class:`Action` objects.

    Unlike :class:`ModuleActionQueue`, the payload schema of a top-level
    ``Action`` is not fixed — skill-launch actions carry arbitrary ``data``
    (e.g. ``{"target_pos": [...]}``) that does not match ``Action.zeros()``.
    Rather than trying to unify all possible schemas, the action is serialized
    with ``msgpack`` into a fixed-size ``uint8`` buffer stored in the
    underlying :class:`SharedMemoryQueue`.

    Args:
        shm_manager: Live :class:`SharedMemoryManager` used to allocate the
            backing shared memory.
        buffer_size: Number of actions the queue can hold.
        max_payload_bytes: Maximum serialized size of a single action.
        action_class: Concrete :class:`Action` subclass used to reconstruct
            actions on ``get``.  Defaults to whichever ``Action`` subclass has
            been registered via ``Action._active_subclass`` (usually the
            project's ``Action`` class).
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        buffer_size: int = 256,
        max_payload_bytes: int = 8192,
        action_class: Optional[type] = None,
    ):
        self.buffer_size = buffer_size
        self.max_payload_bytes = int(max_payload_bytes)
        self._action_class = action_class or Action._resolve_class()

        example = {
            "payload_len": np.int64(0),
            "payload": np.zeros(self.max_payload_bytes, dtype=np.uint8),
        }
        self._queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=buffer_size,
        )

    @classmethod
    def create(
        cls,
        shm_manager: SharedMemoryManager,
        buffer_size: int = 256,
        max_payload_bytes: int = 8192,
        action_class: Optional[type] = None,
    ) -> "ActionQueue":
        return cls(shm_manager, buffer_size, max_payload_bytes, action_class)

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def clear(self):
        self._queue.clear()

    def _encode(self, action: Action) -> Dict[str, Any]:
        payload = msgpack.packb(action.to_dict(), use_bin_type=True)
        n = len(payload)
        if n > self.max_payload_bytes:
            raise ValueError(
                f"Serialized Action is {n}B, exceeds max_payload_bytes="
                f"{self.max_payload_bytes}."
            )
        buf = np.zeros(self.max_payload_bytes, dtype=np.uint8)
        buf[:n] = np.frombuffer(payload, dtype=np.uint8)
        return {"payload_len": np.int64(n), "payload": buf}

    def _decode(self, row: Dict[str, Any]) -> Action:
        n = int(row["payload_len"])
        payload = bytes(np.asarray(row["payload"], dtype=np.uint8)[:n].tobytes())
        d = msgpack.unpackb(payload, raw=False)
        action_type = int(d.get("action_type", 0))
        data = d.get("data", {}) or {}
        return self._action_class(action_type=action_type, data=data)

    def put(self, action: Action) -> None:
        """Enqueue an ``Action``.  Raises :class:`queue.Full` if full."""
        self._queue.put(self._encode(action))

    def get(self) -> Action:
        """Dequeue a single ``Action``.  Raises :class:`queue.Empty` if empty."""
        return self._decode(self._queue.get())

    def get_k(self, k: int) -> List[Action]:
        rows = self._queue.get_k(k)
        n_items = len(rows[next(iter(rows))])
        actions: List[Action] = []
        for i in range(n_items):
            row = {key: values[i] for key, values in rows.items()}
            actions.append(self._decode(row))
        return actions

    def get_all(self) -> List[Action]:
        rows = self._queue.get_all()
        n_items = len(rows[next(iter(rows))])
        actions: List[Action] = []
        for i in range(n_items):
            row = {key: values[i] for key, values in rows.items()}
            actions.append(self._decode(row))
        return actions
