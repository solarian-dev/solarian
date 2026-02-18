"""
AIController — AI-driven controller with VLM perception and skill-based execution.

Architecture overview::

    ┌────────────────────────  Main Process  ─────────────────────────┐
    │                                                                  │
    │  Controller base class                                           │
    │    ├─ _state_receiver thread → observation_buffers (shared mem)  │
    │    ├─ control_loop thread    → control_step() → send_action()   │
    │    └─ display_loop (main thread)                                 │
    │                                                                  │
    │  AIController                                                    │
    │    ├─ action_queue    (mp.Queue)  ← skills push actions here    │
    │    ├─ skills_obs_queue(mp.Queue)  ← skills push status here     │
    │    └─ user_msg_queue  (mp.Queue)  ← user pushes commands here   │
    │                                                                  │
    │  control_step():                                                 │
    │    drains action_queue → deep-merges → returns single Action     │
    │                                                                  │
    │  on_start():                                                     │
    │    spawns ThinkProcess                                           │
    └──────────────────────────────────────────────────────────────────┘

    ┌─────────────── ThinkProcess (separate process, 5 Hz) ───────────┐
    │  1. Read latest image from observation_buffers (shared mem)      │
    │  2. perceive(image, command) → structured scene via VLM          │
    │  3. Drain skills_obs_queue → skill status history                │
    │  4. Drain user_msg_queue → latest command                        │
    │  5. decide(scene, user_msg, skills_obs) → skills to start/stop  │
    │  6. Launch / stop Skill processes accordingly                    │
    └──────────────────────────────────────────────────────────────────┘

    ┌────────────── Skill Processes (per-skill frequency) ────────────┐
    │  - Read observations from shared observation_buffers             │
    │  - Execute skill logic (diffusion policies, motion planners …)  │
    │  - put_action() → controller's action_queue                     │
    │  - put_skill_obs() → think process's skills_obs_queue           │
    └──────────────────────────────────────────────────────────────────┘
"""

import time
import multiprocessing
from multiprocessing import Queue
from typing import Dict, Any, Optional

from solarian.core.controller import Controller
from solarian.core.action import Action
from solarian.core.configs import ControllerConfig
from solarian.utils import get_class
from solarian.conf import settings
from solarian.control.ai.system1.think import ThinkProcess
from solarian.control.ai.system2.skill import Skill, SkillStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* into *base* (mutates *base*)."""
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# ---------------------------------------------------------------------------
# AIController
# ---------------------------------------------------------------------------

class AIController(Controller):
    """
    AI-driven controller that combines VLM perception with skill-based execution.

    * The **control loop** (inherited from ``Controller``) calls
      :meth:`control_step` at the configured frequency.  ``control_step``
      drains the ``action_queue``, deep-merges all pending skill actions into
      a single ``Action`` (as defined by ``settings.ACTION_CLASS``), and
      returns it so the base class can send it to the robot.

    * :meth:`on_start` spawns a :class:`ThinkProcess` in a separate OS
      process.  The think process reads camera images from the shared
      observation buffers, runs VLM-based perception (``perceive``) and
      decision-making (``decide``), and manages skill lifecycles.

    * **Skills** (subclasses of :class:`~solarian.control.ai.system2.skill.Skill`)
      run as individual processes at their own frequencies.  They read
      observations from the shared buffers and push actions into the
      ``action_queue``.

    Usage::

        from solarian.control.ai.ai_controller import AIController
        from my_project.skills import PickSkill, PlaceSkill

        ctrl = AIController(cfg, available_skills={
            "pick": PickSkill,
            "place": PlaceSkill,
        })
        ctrl.connect()
        ctrl.initialize()
        ctrl.send_user_message("Pick up the red cup and place it on the saucer")
        ctrl.start()

    Args:
        cfg: Controller configuration (includes VLM model, connection, etc.).
        available_skills: Registry mapping skill names to ``Skill`` subclasses
            that the think process is allowed to launch.
    """

    def __init__(
        self,
        cfg: ControllerConfig,
        available_skills: Optional[Dict[str, type]] = None,
    ):
        super().__init__(cfg)

        # Inter-process communication queues
        self.action_queue: Queue = multiprocessing.Queue()
        self.skills_obs_queue: Queue = multiprocessing.Queue()
        self.user_msg_queue: Queue = multiprocessing.Queue()

        # Shared event to coordinate shutdown
        self._running_event = multiprocessing.Event()

        # VLM configuration extracted from the controller config
        self.vlm_config: Dict[str, Any] = {
            "model": cfg.system2_model,
            "base_url": cfg.system2_base_url,
            "api_key": cfg.system2_api_key,
            "temperature": 0.0,
            "timeout_s": 10.0,
        }

        # Skill registry — think process can only launch skills listed here
        self.available_skills: Dict[str, type] = available_skills or {}

        self._think_process: Optional[ThinkProcess] = None

    # ------------------------------------------------------------------
    # Controller interface
    # ------------------------------------------------------------------

    def control_step(self) -> Action:
        """
        Drain the action queue and merge all pending skill actions into one.

        Actions targeting different modules are combined.  For the same
        module / actuator / data key the most recently queued value wins.

        Returns:
            A single ``Action`` instance (type determined by
            ``settings.ACTION_CLASS``) ready to send to the robot.
        """
        action_class = get_class(settings.ACTION_CLASS)
        merged = action_class.zeros()

        while not self.action_queue.empty():
            try:
                # try to get the action from the action queue, without blocking
                action_data = self.action_queue.get_nowait()
            except Exception:
                break

            if isinstance(action_data, dict):
                _deep_merge(merged.data, action_data)
            elif isinstance(action_data, Action):
                _deep_merge(merged.data, action_data.data)

        return merged

    def on_start(self):
        """
        Launch the :class:`ThinkProcess` in a separate OS process.

        Called automatically by ``Controller.start()`` after the state
        receiver and control loop threads are running.
        """
        print("[AIController] Launching think process")
        self._running_event.set()

        self._think_process = ThinkProcess(
            observation_buffers=self.observation_buffers,
            action_queue=self.action_queue,
            skills_obs_queue=self.skills_obs_queue,
            user_msg_queue=self.user_msg_queue,
            vlm_config=self.vlm_config,
            available_skills=self.available_skills,
            running_event=self._running_event,
        )
        self._think_process.start()
        print(
            f"[AIController] Think process started "
            f"(PID: {self._think_process.pid})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_user_message(self, message: str):
        """
        Send a natural-language command to the think process.

        The think process will use this as context for perception and
        skill decisions until a new message replaces it.
        """
        self.user_msg_queue.put(message)

    def stop(self):
        """Stop the controller and all child processes gracefully."""
        self._running_event.clear()
        if self._think_process is not None and self._think_process.is_alive():
            self._think_process.join(timeout=5.0)
            if self._think_process.is_alive():
                self._think_process.terminate()
                print("[AIController] Think process terminated forcefully")
        super().stop()
