"""
ObservationQueue - A shared memory queue specialized for ModuleObs objects.

This provides a clean API for sending/receiving ModuleObs objects between
processes by flattening the ModuleObs dict structure for SharedMemoryQueue storage.

No pickle serialization - just dict flattening for efficiency.
"""

from typing import Dict, Any, List, Union
import numbers
from multiprocessing.managers import SharedMemoryManager
import numpy as np

from solarian.utils.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty, Full
from solarian.utils.shared_memory.action_queue import flatten_dict, unflatten_dict
from solarian.core.observation import ModuleObs


class ObservationQueue:
    """
    A shared memory queue for ModuleObs objects.
    
    Converts ModuleObs objects to flattened dictionaries for efficient storage
    in SharedMemoryQueue. No pickle serialization overhead.
    
    Requirements for ModuleObs data:
    - Values must be numbers, np.ndarray, or nested dicts with same constraints
    - The schema (keys and array shapes) must be consistent across all observations
    
    Args:
        shm_manager: SharedMemoryManager instance
        example_obs: An example ModuleObs to define the schema
        buffer_size: Number of observations the queue can hold
    """
    
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        example_obs: ModuleObs,
        buffer_size: int
    ):
        self.buffer_size = buffer_size
        
        # Convert example observation to flattened dict to define schema
        example_dict = self._obs_to_flat_dict(example_obs)
        
        # Create underlying SharedMemoryQueue with the flattened schema
        self._queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example_dict,
            buffer_size=buffer_size
        )
        
        self._module_obs_class = type(example_obs)
    
    def _obs_to_flat_dict(self, obs: ModuleObs) -> Dict[str, Union[np.ndarray, numbers.Number]]:
        """Convert a ModuleObs to a flattened dictionary."""
        if obs is None:
            return {}
        obs_dict = obs.to_dict()
        return flatten_dict(obs_dict)
    
    @staticmethod
    def _to_native(v):
        """Convert 0-d numpy arrays back to Python scalars."""
        if isinstance(v, np.ndarray) and v.ndim == 0:
            return v.item()
        return v

    def _flat_dict_to_obs(self, flat_dict: Dict[str, Any]) -> ModuleObs:
        """Convert a flattened dictionary back to a ModuleObs."""
        native_flat = {k: self._to_native(v) for k, v in flat_dict.items()}
        nested_dict = unflatten_dict(native_flat)
        return self._module_obs_class(nested_dict)
    
    def qsize(self) -> int:
        """Return the approximate number of observations in the queue."""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Return True if the queue is empty."""
        return self._queue.empty()
    
    def clear(self):
        """Clear all observations from the queue."""
        self._queue.clear()
    
    def put(self, obs: ModuleObs) -> None:
        """
        Put an observation into the queue.
        
        Args:
            obs: A ModuleObs instance to enqueue
            
        Raises:
            Full: If the queue is full
            TypeError: If the observation contains unsupported types
        """
        self._queue.put(self._obs_to_flat_dict(obs))
    
    def get(self) -> ModuleObs:
        """
        Get an observation from the queue.
        
        Returns:
            The next ModuleObs in the queue
            
        Raises:
            Empty: If the queue is empty
        """
        return self._flat_dict_to_obs(self._queue.get())
    
    def get_k(self, k: int) -> List[ModuleObs]:
        """
        Get up to k observations from the queue.
        
        Args:
            k: Maximum number of observations to retrieve
            
        Returns:
            List of ModuleObs instances
            
        Raises:
            Empty: If the queue is empty
        """
        flat_results = self._queue.get_k(k)
        
        observations = []
        n_items = len(flat_results[next(iter(flat_results))])
        
        for i in range(n_items):
            flat_dict = {key: values[i] for key, values in flat_results.items()}
            observations.append(self._flat_dict_to_obs(flat_dict))
        
        return observations
    
    def get_all(self) -> List[ModuleObs]:
        """
        Get all observations from the queue.
        
        Returns:
            List of all ModuleObs instances in the queue
            
        Raises:
            Empty: If the queue is empty
        """
        flat_results = self._queue.get_all()
        
        observations = []
        n_items = len(flat_results[next(iter(flat_results))])
        
        for i in range(n_items):
            flat_dict = {key: values[i] for key, values in flat_results.items()}
            observations.append(self._flat_dict_to_obs(flat_dict))
        
        return observations
