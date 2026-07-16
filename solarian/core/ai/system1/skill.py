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

    A skill that is driven by a policy declares the policy server it uses via
    the :attr:`policy_server_module` class attribute and the checkpoint
    location in config (``policy_server.checkpoint``).  Config is resolved by
    deep-merging the plugin's bundled defaults (``conf.yaml`` next to the skill
    module) with a **project-level** override at ``conf/<config_name>.yaml``
    (project wins), so each project points the skill at its own checkpoint
    without editing the installed plugin.  The skill always defines where the
    checkpoint is; when ``policy_server.managed`` is true it launches that
    server as a child process to load it, and the ``solarian run-policy
    <skill>`` command can launch it standalone.

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

    #: Importable module that runs this skill's policy server (``python -m
    #: <module>``).  The module's ``main()`` must be built on
    #: :meth:`~solarian.core.ai.system1.policy_server.PolicyServer.main`, so it
    #: accepts ``--input <checkpoint> --host <host> --port <port>``.  ``None``
    #: for skills that don't use a policy server.
    policy_server_module: Optional[str] = None

    #: Filename of the bundled default config shipped alongside the skill's
    #: module (used to locate ``policy_server.checkpoint`` and friends).
    config_filename: str = "conf.yaml"

    #: Basename (without extension) of the *project-level* config file that
    #: overrides the bundled defaults, looked up at ``conf/<config_name>.yaml``
    #: in the active project.  ``None`` means "no project override"; skills that
    #: want project-owned config (e.g. the policy checkpoint path) should set
    #: this to their stable skill name — typically the ``solarian.skills``
    #: entry-point name (e.g. ``"cup_arrange"``).
    config_name: Optional[str] = None

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
    # Associated policy server
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
        ``policy_server.checkpoint`` and inherit the other ``policy_server.*``
        defaults from the bundled config.
        """
        out = dict(base or {})
        for key, value in (override or {}).items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = Skill._deep_merge(out[key], value)
            else:
                out[key] = value
        return out

    @classmethod
    def _project_conf_dir(cls):
        """Locate the active project's ``conf/`` directory.

        Uses ``SOLARIAN_CONF_PATH`` (set by the CLI for every command), falling
        back to ``SOLARIAN_PROJECT_PATH/conf`` or ``<cwd>/conf``.  These env
        vars are inherited by skill child processes spawned from the
        robot/controller, so this works both from ``run-policy`` and at skill
        runtime.
        """
        import os
        from pathlib import Path

        conf_path = os.environ.get("SOLARIAN_CONF_PATH")
        if conf_path:
            return Path(conf_path)
        project = os.environ.get("SOLARIAN_PROJECT_PATH")
        base = Path(project) if project else Path.cwd()
        return base / "conf"

    @classmethod
    def project_config_path(cls):
        """Path to the project-level ``conf/<config_name>.yaml`` if it exists."""
        from pathlib import Path

        if not cls.config_name:
            return None
        conf_dir = cls._project_conf_dir()
        if conf_dir is None:
            return None
        path = Path(conf_dir) / f"{cls.config_name}.yaml"
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
        project-level override (``conf/<config_name>.yaml``), project wins."""
        return cls._deep_merge(cls.load_default_config(), cls.load_project_config())

    @classmethod
    def get_policy_server_spec(
        cls,
        checkpoint: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Describe how to launch this skill's policy server.

        Returns ``None`` for skills that don't use a policy server (no
        :attr:`policy_server_module`).  Otherwise returns a dict with
        ``module``, ``checkpoint``, ``host``, ``port`` and ``managed``.

        The checkpoint is resolved from the merged config (project-level
        ``conf/<config_name>.yaml`` overriding the bundled ``policy_server.checkpoint``);
        explicit arguments here take precedence.  A missing checkpoint is an error.
        """
        if not cls.policy_server_module:
            return None

        server_cfg = (cls.load_config().get("policy_server") or {})
        checkpoint = checkpoint or server_cfg.get("checkpoint")
        if not checkpoint:
            project_file = f"conf/{cls.config_name}.yaml" if cls.config_name else "conf/<skill>.yaml"
            raise ValueError(
                f"{cls.__name__} declares a policy server "
                f"({cls.policy_server_module!r}) but no checkpoint is defined. "
                f"Set 'policy_server.checkpoint' in the project's {project_file} "
                f"(or the skill's bundled {cls.config_filename}), or pass an explicit checkpoint."
            )
        host = host or server_cfg.get("host", "127.0.0.1")
        port = int(port if port is not None else server_cfg.get("port", 8001))
        return {
            "module": cls.policy_server_module,
            "checkpoint": str(checkpoint),
            "host": str(host),
            "port": port,
            "managed": bool(server_cfg.get("managed", False)),
        }

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
