"""
ObservationBuffer - the ring of recent observations every module publishes into.

Two backings, one API. A module that has to reach another process gets a
shared-memory ring: its observations are flattened to numbers and arrays and
sized from a typed ``ModuleObs``, which is what makes the memory layout fixed.
A module the host owns directly - a leader arm, a keyboard - gets a plain
in-process ring instead, holding whatever Python object it reported.

The second backing exists because the first cannot express everything. Sizing
shared memory needs a schema entry in the project's ``Observation``, which
controller-side devices do not have, and the flattened layout admits only
numbers and arrays, which a set of held keys is not. Rather than route those
modules around the buffer entirely, they get a buffer that can hold them, so
that consumers see one type and one API regardless of which side a module
sits on.
"""

from collections import deque
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


class _LocalRing:
    """In-process ring of whole Python objects.

    Mirrors the slice of :class:`SharedMemoryRingBuffer` that
    :class:`ObservationBuffer` uses, so the buffer above it needs no branch
    beyond skipping the flatten/reconstruct step.
    """

    def __init__(self, get_max_k: int):
        self._items: deque = deque(maxlen=get_max_k)
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def clear(self) -> None:
        self._items.clear()
        self._count = 0

    def put(self, value: Any, wait: bool = True) -> None:
        self._items.append(value)
        self._count += 1

    def get(self) -> Any:
        return self._items[-1]

    def get_last_k(self, k: int) -> List[Any]:
        assert k <= self._items.maxlen
        assert k <= self._count
        return list(self._items)[-k:] if k else []

    def get_all(self) -> List[Any]:
        return list(self._items)


class ObservationBuffer:
    """
    The ring of recent observations for one module.

    Backed by shared memory when built from a typed example, so a module in
    another process can publish into it; backed by a plain in-process ring
    when built with :meth:`local`, so a controller-side device can publish
    something shared memory could not hold.

    Shared-memory mode converts observations to flattened dictionaries rather
    than pickling them, which is why the data has to be numbers, np.ndarray,
    or nested dicts of those, with a schema consistent across every put.

    Args:
        shm_manager: SharedMemoryManager instance, or None for a local ring
        example_observation: An example Observation defining the schema, or
            None for a local ring
        get_max_k: Maximum number of observations that can be retrieved at once
        get_time_budget: Maximum time for get operations
        put_desired_frequency: Expected frequency of put operations
    """
    
    def __init__(
        self,
        shm_manager: SharedMemoryManager | None = None,
        example_observation: ModuleObs | None = None,  # Observation instance
        get_max_k: int = 32,
        get_time_budget: float = 0.2,
        put_desired_frequency: float = 60
    ):
        self.get_max_k = get_max_k

        if shm_manager is None or example_observation is None:
            # No schema to size shared memory from: keep the objects as they
            # were reported. ``module_obs_class`` is None, which is also the
            # flag the read/write paths use to skip flattening.
            self.module_obs_class = None
            self._buffer: Any = _LocalRing(get_max_k)
            return

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

    @property
    def is_local(self) -> bool:
        """Whether this buffer is in-process only and cannot cross a fork."""
        return isinstance(self._buffer, _LocalRing)
    
    def __getstate__(self):
        """Custom pickle support.

        ``module_obs_class`` may be a class loaded from a project-local
        module (e.g. ``ur5.observation.ArmObs``) whose direct class
        reference is not always picklable across the spawn boundary
        (notably when the project package was loaded via
        ``importlib.util.spec_from_file_location``).  We therefore
        serialise the class by its ``(__module__, __qualname__)``
        instead and re-import it in :meth:`__setstate__`.
        """
        state = self.__dict__.copy()
        cls = state.pop('module_obs_class', None)
        if cls is not None:
            state['_module_obs_class_ref'] = (cls.__module__, cls.__qualname__)
        return state

    def __setstate__(self, state):
        """Custom unpickle support — re-imports ``module_obs_class``."""
        ref = state.pop('_module_obs_class_ref', None)
        self.__dict__.update(state)
        # A local buffer has no class to re-import; restore the attribute the
        # read/write paths branch on rather than leaving it undefined.
        self.module_obs_class = None
        if ref is not None:
            import importlib
            try:
                mod = importlib.import_module(ref[0])
                obj = mod
                for part in ref[1].split('.'):
                    obj = getattr(obj, part)
                self.module_obs_class = obj
            except (ImportError, AttributeError):
                # Fall back to whatever ModuleObs subclass was registered
                # during the child's import phase.  This keeps ``get()``
                # usable even if the exact class is not reachable.
                from solarian.core.observation import ModuleObs
                self.module_obs_class = ModuleObs._resolve_class()
    
    @classmethod
    def create_from_example(
        cls,
        shm_manager: SharedMemoryManager,
        example_observation: ModuleObs,
        get_max_k: int = 32,
        get_time_budget: float = 0.2,
        put_desired_frequency: float = 60
    ) -> "ObservationBuffer":
        """
        Create a shared-memory ObservationBuffer from an example ModuleObs.
        
        Args:
            shm_manager: SharedMemoryManager instance
            example_observation: An example ModuleObs that defines the schema
            get_max_k: Maximum number of observations that can be retrieved at once
            get_time_budget: Maximum time for get operations
            put_desired_frequency: Expected frequency of put operations
            
        Returns:
            ObservationBuffer instance
        """
        return cls(
            shm_manager, 
            example_observation, 
            get_max_k, 
            get_time_budget, 
            put_desired_frequency
        )

    @classmethod
    def local(cls, get_max_k: int = 32) -> "ObservationBuffer":
        """
        Create an in-process ObservationBuffer holding whatever it is given.

        For modules the host owns directly, where there is no fork to cross
        and so no reason to demand a schema or a flattenable payload.
        """
        return cls(get_max_k=get_max_k)
    
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
    
    def put(self, observation, wait: bool = True):
        """
        Put an observation into the ring buffer.
        
        Args:
            observation: A ModuleObs to store, or any object on a local buffer
            wait: If True, wait when buffer is too fast; if False, raise TimeoutError
            
        Raises:
            TimeoutError: If wait=False and puts are happening too fast
            TypeError: If the observation contains unsupported types
        """
        if self.module_obs_class is None:
            self._buffer.put(observation, wait=wait)
            return
        self._buffer.put(flatten_dict(observation.to_dict()), wait=wait)
    
    def get(self):
        """
        Get the most recent observation from the ring buffer.
        
        Returns:
            The most recent observation
            
        Raises:
            TimeoutError: If get operation exceeds time budget
        """
        stored = self._buffer.get()
        if self.module_obs_class is None:
            return stored
        return self._flat_dict_to_observation(stored)

    def latest(self):
        """The most recent observation, or ``None`` if nothing has been put.

        ``get()`` on an empty buffer reads an uninitialised slot rather than
        failing, so callers that cannot assume a first sample has landed
        should ask for this instead.
        """
        if self.count == 0:
            return None
        return self.get()
    
    def get_last_k(self, k: int) -> List:
        """
        Get the last k observations from the ring buffer.
        
        Args:
            k: Number of observations to retrieve (must be <= get_max_k)
            
        Returns:
            List of observations, oldest first
            
        Raises:
            AssertionError: If k > get_max_k or k > count
            TimeoutError: If get operation exceeds time budget
        """
        stored = self._buffer.get_last_k(k)
        if self.module_obs_class is None:
            return stored
        return self._rows_to_observations(stored, k)

    def last_k(self, k: int) -> List:
        """The most recent up-to-*k* observations, oldest first.

        Unlike :meth:`get_last_k` this clamps rather than asserting, so a
        caller can ask for the window it wants without first working out how
        much history the buffer happens to hold.
        """
        return self.get_last_k(min(int(k), self.count, self.get_max_k))
    
    def get_all(self) -> List:
        """
        Get all available observations from the ring buffer (up to get_max_k).
        
        Returns:
            List of observations, oldest first
        """
        stored = self._buffer.get_all()
        if self.module_obs_class is None:
            return stored

        # Determine number of items from first key
        first_key = next(iter(stored))
        return self._rows_to_observations(stored, len(stored[first_key]))

    def _rows_to_observations(self, flat_results: Dict[str, Any], n: int) -> List[ModuleObs]:
        """Transpose the buffer's column-per-field layout back into objects."""
        return [
            self._flat_dict_to_observation(
                {key: values[i] for key, values in flat_results.items()}
            )
            for i in range(n)
        ]

