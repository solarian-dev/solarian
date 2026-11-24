from typing import Tuple
from dataclasses import dataclass
import weakref
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from atomics import atomicview, MemoryOrder, UINT

@dataclass
class ArraySpec:
    name: str
    shape: Tuple[int]
    dtype: np.dtype


class SharedAtomicCounter:
    def __init__(self, 
            shm_manager: SharedMemoryManager, 
            size :int=8 # 64bit int
            ):
        shm = shm_manager.SharedMemory(size=size)
        self.shm = shm
        self.size = size
        self._view = None
        self.store(0) # initialize

    def __getstate__(self):
        # The view points into this process's mapping of the block.
        return {**self.__dict__, "_view": None}

    @property
    def buf(self):
        return self.shm.buf[:self.size]

    @property
    def _atomic(self):
        # Held open rather than opened per operation: opening one costs about
        # 75 us on a Pi 5, seven times the load itself. A forked child inherits
        # it, still valid, at the same address. Released at exit, before the
        # block is closed, which refuses while a view exports it.
        if self._view is None:
            context = atomicview(buffer=self.buf, atype=UINT)
            self._view = context.__enter__()
            weakref.finalize(self, context.__exit__, None, None, None)
        return self._view

    def load(self) -> int:
        return self._atomic.load(order=MemoryOrder.ACQUIRE)
    
    def store(self, value: int):
        self._atomic.store(value, order=MemoryOrder.RELEASE)
    
    def add(self, value: int):
        self._atomic.add(value, order=MemoryOrder.ACQ_REL)
