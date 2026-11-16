"""
SkillStatusBuffer — Shared-memory ring buffer for skill lifecycle status updates.

Skills running as separate processes report their status (IDLE, RUNNING,
COMPLETED, FAILED, STOPPED) back to the think process through this buffer
instead of a :class:`multiprocessing.Queue`, to avoid pickling overhead.

Each record is a fixed-width tuple::

    (skill_id: int64, status: int64, timestamp: float64)

``skill_id`` is the run id the controller assigned when it started the skill,
unique per start.  ``status`` is the integer
value of a :class:`solarian.core.skill.SkillStatus` member.
"""

from typing import List, Optional
from multiprocessing.managers import SharedMemoryManager

import numpy as np

from solarian.utils.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


class SkillStatusRecord:
    """A single status record read from a :class:`SkillStatusBuffer`."""

    __slots__ = ("skill_id", "status", "timestamp")

    def __init__(self, skill_id: int, status: int, timestamp: float):
        self.skill_id = int(skill_id)
        self.status = int(status)
        self.timestamp = float(timestamp)

    def __repr__(self) -> str:
        return (
            f"SkillStatusRecord(skill_id={self.skill_id}, "
            f"status={self.status}, timestamp={self.timestamp:.3f})"
        )


class SkillStatusBuffer:
    """Shared-memory ring buffer holding ``(skill_id, status, timestamp)`` rows.

    Args:
        shm_manager: Live :class:`SharedMemoryManager` used to allocate the
            underlying shared memory.
        get_max_k: Maximum number of records readable in one call.
        get_time_budget: Maximum time a ``get`` may take before raising.
        put_desired_frequency: Expected max frequency of ``put`` calls.
    """

    _EXAMPLE = {
        "skill_id": np.int64(0),
        "status": np.int64(0),
        "timestamp": np.float64(0.0),
    }

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        get_max_k: int = 64,
        get_time_budget: float = 0.2,
        put_desired_frequency: float = 60,
    ):
        self._buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=self._EXAMPLE,
            get_max_k=get_max_k,
            get_time_budget=get_time_budget,
            put_desired_frequency=put_desired_frequency,
        )
        # Per-instance cursor for ``drain`` — tracks the total count of
        # records that have already been consumed through this handle.
        # Stored locally (not in shared memory) because each process that
        # wraps the same underlying buffer typically has its own consumer.
        self._read_count: int = 0

    @classmethod
    def create(
        cls,
        shm_manager: SharedMemoryManager,
        get_max_k: int = 64,
        get_time_budget: float = 0.2,
        put_desired_frequency: float = 60,
    ) -> "SkillStatusBuffer":
        return cls(shm_manager, get_max_k, get_time_budget, put_desired_frequency)

    @property
    def count(self) -> int:
        """Total number of records ever written."""
        return self._buffer.count

    def clear(self):
        self._buffer.clear()
        self._read_count = 0

    def put(
        self,
        skill_id: int,
        status: int,
        timestamp: float,
        wait: bool = True,
    ):
        """Append a status record to the buffer."""
        self._buffer.put(
            {
                "skill_id": np.int64(skill_id),
                "status": np.int64(status),
                "timestamp": np.float64(timestamp),
            },
            wait=wait,
        )

    def get(self) -> Optional[SkillStatusRecord]:
        """Return the most recent record, or ``None`` if the buffer is empty."""
        if self._buffer.count == 0:
            return None
        data = self._buffer.get()
        return SkillStatusRecord(
            skill_id=data["skill_id"],
            status=data["status"],
            timestamp=data["timestamp"],
        )

    def get_last_k(self, k: int) -> List[SkillStatusRecord]:
        """Return up to *k* most recent records, oldest first."""
        k = min(k, self._buffer.count)
        if k <= 0:
            return []
        data = self._buffer.get_last_k(k)
        return [
            SkillStatusRecord(
                skill_id=int(data["skill_id"][i]),
                status=int(data["status"][i]),
                timestamp=float(data["timestamp"][i]),
            )
            for i in range(k)
        ]

    def get_all(self) -> List[SkillStatusRecord]:
        """Return every record currently in the buffer, oldest first."""
        if self._buffer.count == 0:
            return []
        data = self._buffer.get_all()
        n = len(data["skill_id"])
        return [
            SkillStatusRecord(
                skill_id=int(data["skill_id"][i]),
                status=int(data["status"][i]),
                timestamp=float(data["timestamp"][i]),
            )
            for i in range(n)
        ]

    def drain(self) -> List[SkillStatusRecord]:
        """Return records appended since the last ``drain`` call, oldest first.

        Unlike :meth:`get_all`, this advances a per-instance read cursor so
        repeated calls never return the same record twice.  Records written
        while more than ``get_max_k`` were pending since the last drain are
        silently dropped (the ring buffer has already overwritten them).
        """
        cur = self._buffer.count
        pending = cur - self._read_count
        if pending <= 0:
            return []
        k = min(pending, self._buffer.get_max_k)
        data = self._buffer.get_last_k(k)
        self._read_count = cur
        return [
            SkillStatusRecord(
                skill_id=int(data["skill_id"][i]),
                status=int(data["status"][i]),
                timestamp=float(data["timestamp"][i]),
            )
            for i in range(k)
        ]
