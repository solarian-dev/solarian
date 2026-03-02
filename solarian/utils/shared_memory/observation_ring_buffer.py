"""
ObservationRingBuffer - A shared memory ring buffer specialized for Observation objects.

This provides a clean API for storing/retrieving Observation objects between
processes by flattening the Observation dict structure for SharedMemoryRingBuffer storage.

No pickle serialization - just dict flattening for efficiency.
"""

from typing import Dict, Any, List, Union
from enum import Enum
import numbers
from multiprocessing.managers import SharedMemoryManager
import numpy as np

from solarian.utils.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from solarian.core.observation import ModuleObs


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
    
    Note:
        None values are skipped (not included in output). When reconstructing
        the object, Pydantic will use default values for missing keys.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if v is None:
            # Skip None values - Pydantic will use defaults when reconstructing
            continue
        elif isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep))
        elif isinstance(v, Enum):
            # Convert enum to its underlying value (int for IntEnum, etc.)
            items[new_key] = v.value
        elif isinstance(v, (np.ndarray, numbers.Number)):
            items[new_key] = v
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
        
        current[parts[-1]] = value
    
    return result


class ObservationRingBuffer:
    """
    A shared memory ring buffer for Observation objects.
    
    Converts Observation objects to flattened dictionaries for efficient storage
    in SharedMemoryRingBuffer. No pickle serialization overhead.
    
    Requirements for Observation data:
    - Values must be numbers, np.ndarray, or nested dicts with same constraints
    - The schema (keys and array shapes) must be consistent across all observations
    
    Args:
        shm_manager: SharedMemoryManager instance
        example_observation: An example Observation to define the schema
        get_max_k: Maximum number of observations that can be retrieved at once
        get_time_budget: Maximum time for get operations
        put_desired_frequency: Expected frequency of put operations
    """
    
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        example_observation: ModuleObs,  # Observation instance
        get_max_k: int = 32,
        get_time_budget: float = 0.2,
        put_desired_frequency: float = 60
    ):
        self.get_max_k = get_max_k
        
        # Convert example observation to flattened dict to define schema
        example_dict = flatten_dict(example_observation.to_dict())

        self.module_obs_class = example_observation.__class__
        
        # Create underlying SharedMemoryRingBuffer with the flattened schema
        self._buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example_dict,
            get_max_k=get_max_k,
            get_time_budget=get_time_budget,
            put_desired_frequency=put_desired_frequency
        )
        
        # Note: We don't store the Observation class reference because it causes
        # pickling issues with dynamically loaded settings modules.
        # Instead, get() returns plain dicts which can be converted by the caller.
    
    def __getstate__(self):
        """Custom pickle support - exclude any non-picklable attributes."""
        state = self.__dict__.copy()
        # Remove any class references that can't be pickled
        state.pop('module_obs_class', None)
        return state
    
    def __setstate__(self, state):
        """Custom unpickle support."""
        self.__dict__.update(state)
    
    @classmethod
    def create_from_example(
        cls,
        shm_manager: SharedMemoryManager,
        example_observation: ModuleObs,
        get_max_k: int = 32,
        get_time_budget: float = 0.2,
        put_desired_frequency: float = 60
    ) -> "ObservationRingBuffer":
        """
        Create an ObservationRingBuffer from an example ModuleObs.
        
        Args:
            shm_manager: SharedMemoryManager instance
            example_observation: An example ModuleObs that defines the schema
            get_max_k: Maximum number of observations that can be retrieved at once
            get_time_budget: Maximum time for get operations
            put_desired_frequency: Expected frequency of put operations
            
        Returns:
            ObservationRingBuffer instance
        """
        return cls(
            shm_manager, 
            example_observation, 
            get_max_k, 
            get_time_budget, 
            put_desired_frequency
        )
    
    # def _observation_to_flat_dict(self, observation: ModuleObs) -> Dict[str, Union[np.ndarray, numbers.Number]]:
    #     """Convert an Observation to a flattened dictionary."""
    #     # Use Pydantic's model_dump() to get dict representation
    #     if observation is None:
    #         return {}
    #     observation_dict = observation.to_dict()
    #     return flatten_dict(observation_dict)
    
    def _flat_dict_to_observation(self, flat_dict: Dict[str, Any]) -> ModuleObs:
        """Convert a flattened dictionary back to an Observation.
        
        Args:
            flat_dict: Flattened dictionary from the ring buffer
        """
        nested_dict = unflatten_dict(flat_dict)
        return self.module_obs_class(nested_dict)
    
    @property
    def count(self) -> int:
        """Return the total number of observations that have been put."""
        return self._buffer.count
    
    def clear(self):
        """Clear the ring buffer."""
        self._buffer.clear()
    
    def put(self, observation: ModuleObs, wait: bool = True):
        """
        Put an observation into the ring buffer.
        
        Args:
            observation: An Observation instance to store
            wait: If True, wait when buffer is too fast; if False, raise TimeoutError
            
        Raises:
            TimeoutError: If wait=False and puts are happening too fast
            TypeError: If the observation contains unsupported types
        """
        self._buffer.put(flatten_dict(observation.data), wait=wait)
    
    def get(self) -> ModuleObs:
        """
        Get the most recent observation from the ring buffer.
        
        Returns:
            The most recent Observation
            
        Raises:
            TimeoutError: If get operation exceeds time budget
        """
        flat_dict = self._buffer.get()
        return self._flat_dict_to_observation(flat_dict)
    
    def get_last_k(self, k: int) -> List[ModuleObs]:
        """
        Get the last k observations from the ring buffer.
        
        Args:
            k: Number of observations to retrieve (must be <= get_max_k)
            
        Returns:
            List of ModuleObs instances, oldest first
            
        Raises:
            AssertionError: If k > get_max_k or k > count
            TimeoutError: If get operation exceeds time budget
        """
        flat_results = self._buffer.get_last_k(k)
        
        # Convert each row back to an Observation
        observations = []
        
        for i in range(k):
            flat_dict = {key: values[i] for key, values in flat_results.items()}
            observations.append(self._flat_dict_to_observation(flat_dict))
        
        return observations
    
    def get_all(self) -> List:
        """
        Get all available observations from the ring buffer (up to get_max_k).
        
        Returns:
            List of ModuleObs instances, oldest first
        """
        flat_results = self._buffer.get_all()
        
        # Determine number of items from first key
        first_key = next(iter(flat_results))
        n_items = len(flat_results[first_key])
        
        # Convert each row back to an Observation
        observations = []
        
        for i in range(n_items):
            flat_dict = {key: values[i] for key, values in flat_results.items()}
            observations.append(self._flat_dict_to_observation(flat_dict))
        
        return observations

