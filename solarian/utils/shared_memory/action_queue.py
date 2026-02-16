"""
ActionQueue - A shared memory queue specialized for Action objects.

This provides a clean API for sending/receiving Action objects between
processes by flattening the Action dict structure for SharedMemoryQueue storage.

No pickle serialization - just dict flattening for efficiency.
"""

from typing import Dict, Any, List, Union
import numbers
from multiprocessing.managers import SharedMemoryManager
import numpy as np

from solarian.utils.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty, Full


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


class ActionQueue:
    """
    A shared memory queue for Action objects.
    
    Converts Action objects to flattened dictionaries for efficient storage
    in SharedMemoryQueue. No pickle serialization overhead.
    
    Requirements for Action data:
    - Values must be numbers, np.ndarray, or nested dicts with same constraints
    - The schema (keys and array shapes) must be consistent across all actions
    
    Args:
        shm_manager: SharedMemoryManager instance
        example_action: An example Action to define the schema
        buffer_size: Number of actions the queue can hold
    """
    
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        example_action,  # Action instance
        buffer_size: int
    ):
        self.buffer_size = buffer_size
        
        # Convert example action to flattened dict to define schema
        example_dict = self._action_to_flat_dict(example_action)
        
        # Create underlying SharedMemoryQueue with the flattened schema
        self._queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example_dict,
            buffer_size=buffer_size
        )
        
        # Store the Action class for reconstruction
        self._action_class = type(example_action)
    
    @classmethod
    def create_from_example(
        cls,
        shm_manager: SharedMemoryManager,
        example_action,
        buffer_size: int = 1024
    ) -> "ActionQueue":
        """
        Create an ActionQueue from an example Action.
        
        Args:
            shm_manager: SharedMemoryManager instance
            example_action: An example Action that defines the schema
            buffer_size: Number of actions the queue can hold
            
        Returns:
            ActionQueue instance
        """
        return cls(shm_manager, example_action, buffer_size)
    
    def _action_to_flat_dict(self, action) -> Dict[str, Union[np.ndarray, numbers.Number]]:
        """Convert an Action to a flattened dictionary."""
        # Use Pydantic's model_dump() to get dict representation
        if action is None:
            return {}
        action_dict = action.model_dump()
        return flatten_dict(action_dict)
    
    def _flat_dict_to_action(self, flat_dict: Dict[str, Any]):
        """Convert a flattened dictionary back to an Action."""
        nested_dict = unflatten_dict(flat_dict)
        return self._action_class.model_validate(nested_dict)
    
    def qsize(self) -> int:
        """Return the approximate number of actions in the queue."""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Return True if the queue is empty."""
        return self._queue.empty()
    
    def clear(self):
        """Clear all actions from the queue."""
        self._queue.clear()
    
    def put(self, action) -> None:
        """
        Put an action into the queue.
        
        Args:
            action: An Action instance to enqueue
            
        Raises:
            Full: If the queue is full
            TypeError: If the action contains unsupported types
        """
        flat_dict = self._action_to_flat_dict(action)
        self._queue.put(flat_dict)
    
    def get(self):
        """
        Get an action from the queue.
        
        Returns:
            The next Action in the queue
            
        Raises:
            Empty: If the queue is empty
        """
        flat_dict = self._queue.get()
        return self._flat_dict_to_action(flat_dict)
    
    def get_k(self, k: int) -> List:
        """
        Get up to k actions from the queue.
        
        Args:
            k: Maximum number of actions to retrieve
            
        Returns:
            List of Action instances
            
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
            List of all Action instances in the queue
            
        Raises:
            Empty: If the queue is empty
        """
        flat_results = self._queue.get_all()
        
        # Convert each row back to an Action
        actions = []
        n_items = len(flat_results[next(iter(flat_results))])
        
        for i in range(n_items):
            flat_dict = {key: values[i] for key, values in flat_results.items()}
            actions.append(self._flat_dict_to_action(flat_dict))
        
        return actions
