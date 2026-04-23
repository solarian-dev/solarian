"""
Skill — Base class for skills that run as separate processes.

Skills are launched by the ThinkProcess when it decides an action should be taken.
Each skill runs at its own frequency, reads robot observations from shared-memory
observation buffers, and produces actions pushed to the action queue.

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
    4. Optionally report progress via ``put_skill_status(status)``
"""

import logging
import time
import traceback
import multiprocessing
from multiprocessing import Process
from multiprocessing import Queue as MpQueue
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, Dict, Optional

from solarian.utils import get_log_queue, setup_process_logging
from solarian.utils.shared_memory.observation_ring_buffer import ObservationRingBuffer
from solarian.utils.shared_memory.action_queue import ActionQueue, ModuleActionQueue
from solarian.utils.shared_memory.skill_status_buffer import SkillStatusBuffer


logger = logging.getLogger(__name__)


class SkillStatus(IntEnum):
    """Skill lifecycle status codes (stored as int in shared memory)."""

    IDLE = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3
    STOPPED = 4


class SkillRunner(IntEnum):
    """Where a :class:`Skill` is executed.

    * ``CONTROLLER`` — the skill runs as a subprocess of the
      :class:`AIController`'s :class:`ThinkProcess` (typical for "digital"
      skills like a diffusion policy).  The skill pushes per-module actions
      into the robot's shared-memory action queues via :meth:`put_action`.
    * ``ROBOT`` — the skill runs on the robot process.  The think process
      does **not** spawn it locally; instead it enqueues a top-level
      :class:`Action` (with ``action_type`` configured on the skill) that
      the robot's action handler uses to start the skill on its side.
    * ``SERVER`` — same dispatch as ``ROBOT`` but semantically the skill
      lives on the relay server.  The think process still only enqueues an
      action; where the robot vs server picks it up is transport-layer
      concern.
    """

    CONTROLLER = 0
    ROBOT = 1
    SERVER = 2


class Skill(Process, ABC):
    """
    Base class for skills that execute as separate processes.

    Skills read observations from shared :class:`ObservationRingBuffer`
    instances and produce actions that are placed into per-module
    :class:`ModuleActionQueue` shared-memory queues.  They also report their
    lifecycle status through a shared-memory :class:`SkillStatusBuffer` so
    the think process can incorporate them when making decisions.

    Subclasses **must** implement :meth:`control_step`.
    They may also override :meth:`setup` and :meth:`teardown`.

    Args:
        name: Unique identifier for this skill instance.
        skill_id: Integer id assigned by the orchestrator (e.g. index in
            ``available_skills``).  Used as the key in
            :class:`SkillStatusBuffer` records.
        observation_buffers: Dict mapping module names to shared-memory
            :class:`ObservationRingBuffer` instances.
        action_queues: Dict mapping module names to shared-memory
            :class:`ModuleActionQueue` instances where this skill pushes
            per-module actions destined for the robot (robot-side skills).
        action_queue: Shared-memory :class:`ActionQueue` carrying top-level
            :class:`Action` objects.  Controller-side skills push actions
            here so the controller's ``control_step`` can forward them to
            the robot; it is typically ``None`` for robot-side skills.
        skills_status_buffer: Shared-memory :class:`SkillStatusBuffer` where
            the skill publishes status transitions.
        frequency: How often :meth:`control_step` is called, in Hz.
        params: Skill-specific parameters forwarded from the think process
            decision.
        runner: Where this skill is executed (see :class:`SkillRunner`).
            Declared by the subclass/controller author at attribute-decl
            time, and used by :class:`ThinkProcess` to decide whether to
            spawn the skill locally or send an ``Action`` to the robot /
            server instead.
        action_type: Integer action type used when ``runner`` is
            ``ROBOT``/``SERVER`` — the think process will build an
            ``Action(action_type=...)`` and push it to the controller's
            :class:`ActionQueue` so the remote side can launch the skill.
            Ignored when ``runner`` is ``CONTROLLER``.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        skill_id: Optional[int] = None,
        observation_buffers: Optional[Dict[str, ObservationRingBuffer]] = None,
        action_queues: Optional[Dict[str, ModuleActionQueue]] = None,
        action_queue: Optional[ActionQueue] = None,
        skills_status_buffer: Optional[SkillStatusBuffer] = None,
        frequency: float = 10.0,
        params: Optional[Dict[str, Any]] = None,
        runner: "SkillRunner | int" = SkillRunner.CONTROLLER,
        action_type: Optional[int] = None,
    ):
        # ``name`` defaults to the class name so subclasses can be declared
        # as class attributes on an AIController (e.g. ``pick = PickSkill()``)
        # without having to pass constructor arguments — the registry-building
        # code then picks up ``type(instance)`` by attribute name.
        resolved_name = name or type(self).__name__
        super().__init__(name=f"Skill-{resolved_name}", daemon=True)
        self.skill_name = resolved_name
        self.skill_id = int(skill_id) if skill_id is not None else 0
        self.observation_buffers = observation_buffers or {}
        self.action_queues = action_queues or {}
        self.action_queue = action_queue
        self.skills_status_buffer = skills_status_buffer
        self.frequency = frequency
        self.params = params or {}
        self.runner = SkillRunner(int(runner)) if not isinstance(runner, SkillRunner) else runner
        self.action_type = int(action_type) if action_type is not None else None
        self._stop_event = multiprocessing.Event()
        # Capture the parent's shared log queue so the child process can route
        # its log records back to the StatusBoard Logs panel.
        self._log_queue = get_log_queue()
        # Parent can poll this for (skill_name, traceback_string) after run()
        # if it wants to surface skill crashes in its own logs.
        self._error_queue: MpQueue = MpQueue()

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

    def put_action(self, module_name: str, action) -> None:
        """
        Submit a ``ModuleAction`` to the per-module shared-memory queue.

        Args:
            module_name: Target module name (must be a key in
                ``self.action_queues``).
            action: A ``ModuleAction`` instance for that module.
        """
        queue = self.action_queues.get(module_name)
        if queue is None:
            raise KeyError(f"Unknown action queue for module {module_name!r}")
        queue.put(action)

    def put_skill_status(self, status: SkillStatus) -> None:
        """
        Report a skill status transition to the think process.

        Args:
            status: One of :class:`SkillStatus` values.
        """
        self.skills_status_buffer.put(
            skill_id=self.skill_id,
            status=int(status),
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    # Stop control
    # ------------------------------------------------------------------

    def request_stop(self):
        """Signal this skill to stop after the current ``control_step`` call."""
        self._stop_event.set()

    @property
    def stopped(self) -> bool:
        """Whether a stop has been requested."""
        return self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Process entry point
    # ------------------------------------------------------------------

    def run(self):
        """
        Main process entry point (called by ``Process.start()``).

        Runs :meth:`setup`, then loops :meth:`control_step` at
        ``self.frequency`` Hz, and finally runs :meth:`teardown`.

        Subclasses that need full control over the loop may override this
        method entirely.
        """
        # Route log records from this child process back to the parent's
        # shared log queue so exceptions show up in the StatusBoard Logs
        # panel instead of being silently swallowed.
        setup_process_logging(
            logger_name=None,
            log_queue=self._log_queue,
            install_fallback=False,
        )

        self.put_skill_status(SkillStatus.RUNNING)
        final_status = SkillStatus.COMPLETED
        try:
            self.setup()
            while not self._stop_event.is_set():
                t0 = time.monotonic()
                try:
                    finished = self.control_step()
                    if finished:
                        break
                except Exception as exc:
                    logger.exception(
                        "Skill '%s' control_step crashed: %s",
                        self.skill_name,
                        exc,
                    )
                    self._push_error(exc)
                    final_status = SkillStatus.FAILED
                    break

                elapsed = time.monotonic() - t0
                sleep_time = (1.0 / self.frequency) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if self._stop_event.is_set():
                final_status = SkillStatus.STOPPED
            else:
                final_status = SkillStatus.COMPLETED
        except Exception as exc:
            logger.exception(
                "Skill '%s' setup/loop crashed: %s",
                self.skill_name,
                exc,
            )
            self._push_error(exc)
            final_status = SkillStatus.FAILED
        finally:
            try:
                self.teardown()
            except Exception as exc:
                logger.exception(
                    "Skill '%s' teardown crashed: %s",
                    self.skill_name,
                    exc,
                )
                self._push_error(exc)
            finally:
                try:
                    self.put_skill_status(final_status)
                except Exception as exc:
                    logger.exception(
                        "Skill '%s' failed to publish final status: %s",
                        self.skill_name,
                        exc,
                    )

    def _push_error(self, exc: BaseException) -> None:
        """Send (skill_name, traceback) back to the parent process.

        Kept best-effort: failing to push (e.g. queue full during shutdown)
        must never mask the original exception.
        """
        try:
            tb_str = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            self._error_queue.put_nowait((self.skill_name, tb_str))
        except Exception:
            pass

    def get_error(self) -> tuple[str, str] | None:
        """Return ``(skill_name, traceback_string)`` if this skill crashed, else None.

        Safe to call from the parent process only.
        """
        try:
            return self._error_queue.get_nowait()
        except Exception:
            return None

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

    @abstractmethod
    def control_step(self) -> bool:
        """
        Control step implementation.

        Subclasses should implement this method to define the control step logic.

        Returns:
            bool: ``True`` if the skill was completed.
        """
