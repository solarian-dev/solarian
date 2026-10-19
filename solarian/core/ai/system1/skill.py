"""
Skill — Base class for control logic: consumes observations, produces actions.

Skills are launched by the Controller or the ThinkProcess when it decides an
action should be taken. Each skill runs at its own frequency, reads robot
observations from shared-memory observation buffers, and produces actions
pushed to the action queue.

How a skill is scheduled is not part of what it is: ``execution_place`` says which
runtime owns it and ``execution_mode`` says how it is scheduled there, and a
:class:`~solarian.core.runner.Runner` carries that out.

Lifecycle::

    Controller / ThinkProcess.decide()  →  launches Skill
          ↓
    Skill.begin()  → setup()
          ↓
    Skill.control_step(dt)  ← at ``self.frequency`` Hz
          ↓
    Skill.teardown()

``control_step`` reads ``self.observations`` and ``self.events`` — both wired
up by the host and refreshed each step, whichever runtime the skill lives in —
and returns the action(s) it wants performed; the host routes them. It signals
its own completion with ``finish()`` rather than through the return value, so a
one-shot skill can command a pose and complete on the same step.
"""

import logging
import time
from multiprocessing import Queue as MpQueue
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Dict, Optional

from solarian.core.execution import ExecutionMode, ExecutionPlace, WorkerExit
from solarian.core.runner import Runner, make_runner
from solarian.utils import get_log_queue, setup_process_logging
from solarian.utils.shared_memory.observation_buffer import ObservationBuffer
from solarian.utils.shared_memory.action_queue import ModuleActionQueue
from solarian.utils.shared_memory.skill_status_buffer import SkillStatusBuffer

if TYPE_CHECKING:
    # Imported for annotations only: solarian.core.action reaches back here.
    from solarian.core.action import Action


logger = logging.getLogger(__name__)

# Param keys whose values are module names this skill *writes*. Occupancy and
# action-queue wiring read these; read-only roles (leader, camera) stay out.
_WRITER_KEYS = ("arm", "gripper")


class SkillStatus(IntEnum):
    """Skill lifecycle status codes (stored as int in shared memory)."""

    IDLE = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3
    STOPPED = 4


_EXIT_STATUS = {
    WorkerExit.COMPLETED: SkillStatus.COMPLETED,
    WorkerExit.STOPPED: SkillStatus.STOPPED,
    WorkerExit.FAILED: SkillStatus.FAILED,
}


class Skill(ABC):
    """
    Base class for skills: control logic that consumes observations and
    produces actions.

    Skills read observations from per-module :class:`ObservationBuffer`
    instances and produce actions that are placed into per-module
    :class:`ModuleActionQueue` shared-memory queues.  They also report their
    lifecycle status through a shared-memory :class:`SkillStatusBuffer` so
    the think process can incorporate them when making decisions.

    Where a skill lives and how it is scheduled are two separate axes:
    :attr:`execution_place` (controller / robot / server) and
    :attr:`execution_mode` (process / thread / inline). A keyboard teleop
    skill is ``CONTROLLER`` + ``INLINE``; a diffusion policy on the robot is
    ``ROBOT`` + ``PROCESS``. Remote placements are not launched locally at
    all — the controller enqueues an ``Action`` the remote side picks up.

    Subclasses **must** implement :meth:`control_step`.
    They may also override :meth:`setup` and :meth:`teardown`.

    A skill that is driven by a policy names the service that serves it via
    the :attr:`policy_service` class attribute, and its conf's ``server:``
    block says which checkpoint and port (see :mod:`solarian.services.launch`).
    Config is resolved by deep-merging the plugin's bundled defaults
    (``conf.yaml`` next to the skill module) with a **project-level** override
    at ``conf/skills/<config_name>.yaml`` (project wins), so each project
    points the skill at its own checkpoint without editing the installed
    plugin. ``solarian run-service <conf>`` runs that server standalone, and a
    skill may launch it itself when ``server.managed`` is true.

    Args:
        name: Unique identifier for this skill instance.
        skill_id: Integer id assigned by the orchestrator (e.g. index in
            ``available_skills``).  Used as the key in
            :class:`SkillStatusBuffer` records.
        observations: Dict mapping module names to their
            :class:`ObservationBuffer`. Populated by whichever host starts
            the skill, and read by :meth:`control_step` as
            :attr:`observations`.
        action_queues: Dict mapping module names to the
            :class:`ModuleActionQueue` this skill pushes that module's actions
            into. The only channel out of a skill, whichever host started it:
            robot-side the module reads it directly, controller-side the
            controller drains it and merges before sending to the robot.
        skills_status_buffer: Shared-memory :class:`SkillStatusBuffer` where
            the skill publishes status transitions.
        event_inbox: Queue the host fills with each tick's drained events.
            :meth:`step` empties it into :attr:`events`. Assigned by the host
            when the skill starts, so a skill constructed by hand simply sees
            no events.
        frequency: How often :meth:`control_step` is called, in Hz.
        params: Skill-specific parameters. Module names this skill writes
            live here as ``params["arm"]`` / ``params["gripper"]`` (plain
            strings); everything else the skill needs does too.
        execution_place: Which runtime owns this skill (see :class:`ExecutionPlace`).
            Declared by the subclass/controller author at attribute-decl
            time, and used to decide whether to launch the skill locally or
            send an ``Action`` to the robot / server instead.
        execution_mode: How the skill is scheduled where it lives (see
            :class:`ExecutionMode`).
    """

    #: Which runtime owns this skill. Overridable per instance.
    execution_place: ExecutionPlace = ExecutionPlace.CONTROLLER

    #: How this skill is scheduled where it lives. Overridable per instance.
    execution_mode: ExecutionMode = ExecutionMode.PROCESS

    #: Skills never outlive the runtime that launched them.
    daemon: bool = True

    #: ``solarian.services`` entry point serving this skill's policy (a
    #: :class:`~solarian.services.policy.PolicyServer`): the default for its
    #: conf's ``server.service``. ``None`` for skills that don't use a policy.
    policy_service: Optional[str] = None

    #: Filename of the bundled default config shipped alongside the skill's
    #: module.
    config_filename: str = "conf.yaml"

    #: Basename (without extension) of the *project-level* config file that
    #: overrides the bundled defaults, looked up at ``conf/skills/<config_name>.yaml``
    #: in the active project.  ``None`` means "no project override"; skills that
    #: want project-owned config (e.g. the policy checkpoint path) should set
    #: this to their stable skill name — typically the ``solarian.skills``
    #: entry-point name (e.g. ``"umi"``).
    config_name: Optional[str] = None

    def __init__(
        self,
        name: Optional[str] = None,
        skill_id: Optional[int] = None,
        observations: Optional[Dict[str, ObservationBuffer]] = None,
        action_queues: Optional[Dict[str, ModuleActionQueue]] = None,
        skills_status_buffer: Optional[SkillStatusBuffer] = None,
        frequency: float = 10.0,
        params: Optional[Dict[str, Any]] = None,
        execution_place: "ExecutionPlace | str | None" = None,
        execution_mode: "ExecutionMode | str | None" = None,
        status_mp_q: Optional[MpQueue] = None,
        event_inbox: Optional[MpQueue] = None,
    ):
        # Defaults to :meth:`registry_name` so a running skill is filed under
        # the same name the registry and a ``START_SKILL`` action use, rather
        # than under a second one only its host knows.
        resolved_name = name or type(self).registry_name()
        self.name = f"Skill-{resolved_name}"
        self.skill_name = resolved_name
        if skill_id is None:
            class_id = type(self).__dict__.get("skill_id")
            skill_id = class_id if isinstance(class_id, int) else 0
        self.skill_id = int(skill_id)
        #: Every module's :class:`ObservationBuffer`, keyed by module name.
        #: The host that starts this skill points it at its own buffers.
        self.observations = observations or {}
        #: Where this skill's per-module actions go, keyed by module name.
        #: The host that starts it decides what backs each queue.
        self.action_queues = action_queues or {}
        #: Where the host posts each tick's drained events for this skill.
        self.event_inbox = event_inbox
        #: What happened since the previous step, oldest first. Refreshed by
        #: :meth:`step`, so it is this tick's batch for the whole step.
        self.events: list = []
        self.skills_status_buffer = skills_status_buffer
        self.frequency = frequency
        self.params = dict(params or {})
        if execution_place is not None:
            self.execution_place = ExecutionPlace(execution_place)
        if execution_mode is not None:
            self.execution_mode = ExecutionMode(execution_mode)
        self._finished = False
        self._t_last_step: float | None = None
        self._runner: Runner | None = None
        # Capture the parent's shared log queue so the child process can route
        # its log records back to the StatusBoard Logs panel.
        self._log_queue = get_log_queue()
        # Parent StatusBoard mp queue so child ``StatusBoard.set`` reaches the
        # robot/controller UI (spawned skills do not inherit the singleton).
        self._status_mp_q = status_mp_q

    @classmethod
    def registry_name(cls) -> str:
        """This skill's identity everywhere a name has to travel.

        The registry the VLM reads, the ``START_SKILL`` action the controller
        sends the robot, and the robot's lookup of what to construct all key
        off this one string, so a skill cannot be known by one name on one
        side of the link and another name on the other. ``config_name`` is
        preferred because it is what a plugin publishes as its entry-point
        name; the class name is the fallback.
        """
        return cls.config_name or cls.__name__

    def commanded_module_names(self) -> set[str]:
        """Module names this skill writes (for occupancy / preemption)."""
        return {
            self.params[key]
            for key in _WRITER_KEYS
            if isinstance(self.params.get(key), str)
        }

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
    # Config
    # ------------------------------------------------------------------

    @classmethod
    def default_config_path(cls):
        """Return the path to the skill's bundled default config, or ``None``."""
        import inspect
        from pathlib import Path

        try:
            module_file = inspect.getfile(cls)
        except TypeError:
            return None
        path = Path(module_file).parent / cls.config_filename
        return path if path.exists() else None

    @classmethod
    def load_default_config(cls) -> Dict[str, Any]:
        """Load the skill's bundled default config (``{}`` if none is shipped)."""
        import yaml

        path = cls.default_config_path()
        if path is None:
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge ``override`` onto ``base`` (values in ``override`` win).

        Nested dicts are merged key-by-key so a project file can set e.g. just
        ``server.checkpoint`` and inherit the other ``server.*`` defaults from
        the bundled config.
        """
        out = dict(base or {})
        for key, value in (override or {}).items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = Skill._deep_merge(out[key], value)
            else:
                out[key] = value
        return out

    @classmethod
    def project_config_path(cls):
        """Path to the project-level ``conf/skills/<config_name>.yaml`` if it exists."""
        from solarian.core.conf_paths import skill_config_path

        if not cls.config_name:
            return None
        path = skill_config_path(cls.config_name)
        return path if path.exists() else None

    @classmethod
    def load_project_config(cls) -> Dict[str, Any]:
        """Load the project-level config override (``{}`` if none present)."""
        import yaml

        path = cls.project_config_path()
        if path is None:
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    @classmethod
    def load_config(cls) -> Dict[str, Any]:
        """Resolved skill config: bundled defaults deep-merged with the
        project-level override (``conf/skills/<config_name>.yaml``), project wins.

        ``${deployment.*}`` references are resolved last, so both files can name
        host-specific values such as a policy server URL."""
        from solarian.utils.configs import resolve_with_deployment

        merged = cls._deep_merge(cls.load_default_config(), cls.load_project_config())
        return resolve_with_deployment(merged)

    # ------------------------------------------------------------------
    # Stop control
    # ------------------------------------------------------------------

    def request_stop(self):
        """Signal this skill to stop after the current ``control_step`` call."""
        self.runner.stop(wait=False)

    @property
    def stopped(self) -> bool:
        """Whether a stop has been requested."""
        return not self.runner.is_alive()

    def finish(self) -> None:
        """Declare this skill done; the host stops it after the current step.

        Symmetric with :meth:`request_stop`, which is how *someone else* ends
        the skill. Keeping completion in state rather than in the return value
        lets :meth:`control_step` return an action on the very step it
        finishes — a one-shot move both commands the pose and completes.
        """
        self._finished = True

    @property
    def finished(self) -> bool:
        """Whether the skill has declared itself complete."""
        return self._finished

    # ------------------------------------------------------------------
    # Lifecycle (delegated to the runner)
    # ------------------------------------------------------------------

    @property
    def runner(self) -> Runner:
        """The scheduler for this skill, created on first use."""
        if self._runner is None:
            self._runner = make_runner(self)
        return self._runner

    def start(self, wait: bool = False) -> None:
        """Hand this skill to its runner."""
        self.runner.start(wait=wait)

    def stop(self, wait: bool = True) -> None:
        self.runner.stop(wait=wait)

    def terminate(self) -> None:
        self.runner.terminate()

    def is_alive(self) -> bool:
        return self.runner.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self.runner.join(timeout)

    @property
    def pid(self) -> int | None:
        """OS pid when this skill runs in its own process, else ``None``."""
        return self.runner.pid

    def get_error(self) -> tuple[str, str] | None:
        """Return ``(skill_name, traceback_string)`` if this skill crashed, else None.

        Safe to call from the host process only.
        """
        return self.runner.get_error()

    # ------------------------------------------------------------------
    # Worker protocol (driven by the runner)
    # ------------------------------------------------------------------

    def worker_setup(self, reconnect: bool) -> None:
        """Announce RUNNING and run the subclass :meth:`setup`."""
        if reconnect:
            # Route log records from this child process back to the parent's
            # shared log queue so exceptions show up in the StatusBoard Logs
            # panel instead of being silently swallowed.
            setup_process_logging(
                logger_name=None,
                log_queue=self._log_queue,
                install_fallback=False,
            )
            if self._status_mp_q is not None:
                from solarian.utils.status_board import StatusBoard

                StatusBoard.init_child(self._status_mp_q)

        self.put_skill_status(SkillStatus.RUNNING)
        self.begin()

    def worker_tick(self) -> bool:
        """One control step. Returns False once the skill reports completion."""
        self.dispatch(self.step())
        return not self._finished

    def _drain_inbox(self) -> list:
        """Take everything the host has posted since the previous step.

        Without an inbox — a skill constructed by hand, with no host attached
        — there is nothing to take and the step simply sees no events.
        """
        if self.event_inbox is None:
            return []
        events = []
        while True:
            try:
                events.append(self.event_inbox.get_nowait())
            except Exception:  # noqa: BLE001 - an empty inbox is the normal case
                return events

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def begin(self) -> None:
        """Clear per-run step state, then run the subclass :meth:`setup`.

        Called by whichever host is about to start stepping this skill, so a
        skill that finished once and is started again does not begin already
        finished, nor with a stale ``dt``.
        """
        self._finished = False
        self._t_last_step = None
        self.setup()

    def step(self):
        """Refresh this tick's inputs, then call :meth:`control_step`.

        The single entry point for both hosts — the runner loop and the
        host's inline tick — so ``dt`` is measured and :attr:`events` is
        collected the same way whether a skill runs in its own process or on
        someone else's loop. Taking the whole batch up front is what makes
        it stable for the length of the step.
        """
        self.events = self._drain_inbox()
        now = time.monotonic()
        dt = (
            now - self._t_last_step
            if self._t_last_step is not None
            else 1.0 / max(float(self.frequency), 1e-9)
        )
        self._t_last_step = now
        return self.control_step(dt)

    @staticmethod
    def as_actions(actions) -> list:
        """Normalize a ``control_step`` return value to a list.

        A step may want to do nothing, one thing, or — when it is scheduling a
        chunk of future waypoints — many things, so all three are accepted.
        """
        if actions is None:
            return []
        if isinstance(actions, (list, tuple)):
            return [a for a in actions if a is not None]
        return [actions]

    def dispatch(self, actions) -> None:
        """Push what :meth:`control_step` returned onto the module queues.

        The one way out of a skill, on either host. Robot-side the module at
        the other end applies it; controller-side the controller drains it,
        merges it with what the other skills commanded, and sends one message
        to the robot.
        """
        from solarian.core.action import ActionType

        for action in self.as_actions(actions):
            if int(action.action_type) != int(ActionType.MODULE_ACTION):
                logger.warning(
                    "Skill '%s' returned a non-module action (%s); skills "
                    "command modules, and starting or stopping another skill "
                    "is the host's call",
                    self.skill_name,
                    action.action_type,
                )
                continue
            for module_name, module_action in action.items():
                queue = self.action_queues.get(module_name)
                if queue is None:
                    logger.warning(
                        "Skill '%s' has no action queue for module '%s'",
                        self.skill_name,
                        module_name,
                    )
                    continue
                queue.put(module_action)

    def worker_teardown(self, exit_reason: WorkerExit) -> None:
        """Release resources and publish the final status."""
        final_status = _EXIT_STATUS[exit_reason]
        try:
            self.teardown()
        except Exception as exc:
            logger.exception(
                "Skill '%s' teardown crashed: %s", self.skill_name, exc
            )
            final_status = SkillStatus.FAILED
        try:
            self.put_skill_status(final_status)
        except Exception as exc:
            logger.exception(
                "Skill '%s' failed to publish final status: %s",
                self.skill_name,
                exc,
            )

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
    def control_step(self, dt: float) -> "Action | list[Action] | None":
        """One step of control logic: state and happenings in, intent out.

        Reads :attr:`observations` for what the world looks like and
        :attr:`events` for what just happened. Both are set up by whichever
        host started the skill and refreshed by :meth:`step`, so they mean the
        same thing whether this skill runs in its own process or inline on the
        host's loop, and a test can populate them directly.

        Args:
            dt: Seconds since the previous call, for rate-dependent maths.
                The one input that is genuinely per-call, hence an argument:
                on the first step it is the nominal ``1 / frequency``.

        Returns:
            An :class:`~solarian.core.action.Action` to perform, or ``None``
            for "nothing this tick". Completion is *not* signalled here: call
            :meth:`finish` for that, so the return type stays single-purpose.
        """
