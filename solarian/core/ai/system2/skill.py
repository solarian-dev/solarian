"""
Skill — Base class for skills that run as separate processes.

Skills are launched by the ThinkProcess when it decides an action should be taken.
Each skill runs at its own frequency, reads robot observations from shared-memory
observation buffers, and produces actions that get merged by the AIController.

Lifecycle::

    ThinkProcess.decide()  →  launches Skill
          ↓
    Skill.setup()
          ↓
    Skill.execute()  ← called in a loop at ``self.frequency`` Hz
          ↓
    Skill.teardown()

During ``execute()``, the skill should:
    1. Read observations via ``get_observation(module_name)``
    2. Compute actions (e.g. run a diffusion policy)
    3. Submit them via ``put_action(action_data)``
    4. Optionally report progress via ``put_skill_obs(status, data)``
"""

import time
import multiprocessing
from multiprocessing import Process, Queue
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class SkillStatus:
    """Skill lifecycle status constants."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class Skill(Process, ABC):
    """
    Base class for skills that execute as separate processes.

    Skills read observations from shared ``ObservationRingBuffer`` instances
    and produce actions that are placed into the controller's ``action_queue``.
    They also report their status and results through the ``skills_obs_queue``
    so the think process can incorporate them when making decisions.

    Subclasses **must** implement :meth:`run`.
    They may also override :meth:`setup` and :meth:`teardown`.

    Args:
        name: Unique identifier for this skill instance.
        observation_buffers: Dict mapping module names to
            ``ObservationRingBuffer`` instances (shared memory).
        action_queue: ``mp.Queue`` — skills put action dicts here for the
            controller to merge and send to the robot.
        skills_obs_queue: ``mp.Queue`` — skills put status / result dicts
            here for the think process to read.
        frequency: How often :meth:`execute` is called, in Hz.
        params: Skill-specific parameters forwarded from the think process
            decision.
    """

    def __init__(
        self,
        name: str,
        observation_buffers: Dict[str, Any],
        action_queue: Queue,
        skills_obs_queue: Queue,
        frequency: float = 10.0,
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=f"Skill-{name}", daemon=True)
        self.skill_name = name
        self.observation_buffers = observation_buffers
        self.action_queue = action_queue
        self.skills_obs_queue = skills_obs_queue
        self.frequency = frequency
        self.params = params or {}
        self._stop_event = multiprocessing.Event()

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def get_observation(self, module_name: str):
        """
        Get the latest observation for *module_name* from shared memory.

        Returns:
            The latest ``ModuleObs`` from the ring buffer, or ``None``
            if the buffer is empty or the module name is unknown.
        """
        buffer = self.observation_buffers.get(module_name)
        if buffer is None:
            return None
        try:
            return buffer.get()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Action / status helpers
    # ------------------------------------------------------------------

    def put_action(self, action_data: Dict):
        """
        Submit an action dict to the controller's action queue.

        The dict should follow the ``Action`` data format::

            {
                module_name: {
                    actuator_name: {data_name: value, ...},
                    ...
                },
                ...
            }
        """
        self.action_queue.put(action_data)

    def put_skill_obs(self, status: str, data: Optional[Dict] = None):
        """
        Report skill status / results to the think process.

        Args:
            status: One of :class:`SkillStatus` constants.
            data: Optional payload (e.g. error message, result summary).
        """
        obs = {
            "skill": self.skill_name,
            "status": status,
            "timestamp": time.time(),
        }
        if data is not None:
            obs["data"] = data
        self.skills_obs_queue.put(obs)

    # ------------------------------------------------------------------
    # Stop control
    # ------------------------------------------------------------------

    def request_stop(self):
        """Signal this skill to stop after the current ``execute()`` call."""
        self._stop_event.set()

    @property
    def stopped(self) -> bool:
        """Whether a stop has been requested."""
        return self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Process entry point
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self):
        """
        Main process entry point (called by ``Process.start()``).

        Runs :meth:`setup`, then loops :meth:`execute` at
        ``self.frequency`` Hz, and finally runs :meth:`teardown`.

        Subclasses that need full control over the loop may override this
        method entirely.
        """
        pass

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def setup(self):
        """Called once before the execution loop begins.

        Override to load models, allocate buffers, etc.
        """

    def teardown(self):
        """Called once after the execution loop ends.

        Override to release resources.
        """

    
