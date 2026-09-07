"""
AIController — AI-driven controller with VLM perception and skill-based execution.

Architecture overview::

    ┌────────────────────────  Main Process  ─────────────────────────┐
    │                                                                  │
    │  Controller base class                                           │
    │    ├─ _state_receiver thread → observation_buffers (shared mem)  │
    │    ├─ control_loop thread    → control_step() → _send_action()  │
    │    └─ display_loop (main thread)                                 │
    │                                                                  │
    │  AIController                                                    │
    │    ├─ skill_command_queue  ← think process starts/stops skills  │
    │    ├─ skill_status_buffer(mp.Queue)  ← skills push status here     │
    │    └─ user_msg_queue  (mp.Queue)  ← user pushes commands here   │
    │                                                                  │
    │  control_step():                                                 │
    │    drains every skill's action_queues → merges → one Action      │
    │                                                                  │
    │  on_start():                                                     │
    │    spawns ThinkProcess                                           │
    └──────────────────────────────────────────────────────────────────┘

    ┌─────────────── ThinkProcess (separate process, ~2 Hz) ──────────┐
    │  1. Drain skill_status_buffer → skill status history             │
    │  2. Drain user_msg_queue → latest command                        │
    │  3. Read latest image + proprioception from observation_buffers  │
    │  4. decide(image, user_msg, skills_status)                       │
    │         → skills to start/stop via a VLM call                    │
    │  5. Launch / stop Skill processes accordingly                    │
    └──────────────────────────────────────────────────────────────────┘

    ┌────────────── Skill Processes (per-skill frequency) ────────────┐
    │  - Read observations from shared observation_buffers             │
    │  - Execute skill logic (diffusion policies, motion planners …)  │
    │  - control_step() → per-module action_queues → controller merge │
    │  - put_skill_status() → think process's skill_status_buffer      │
    └──────────────────────────────────────────────────────────────────┘
"""

import logging
import sys
import threading
import multiprocessing
from multiprocessing import Queue
from typing import Callable, Dict, Any, Optional

from solarian.core.controller import Controller
from solarian.core.action import Action
from solarian.core.configs import ControllerConfig
from solarian.utils import get_class, StatusBoard
from solarian.conf import settings
from solarian.core.ai.system2.think import ThinkProcess

try:
    from pynput import keyboard as _pynput_keyboard
except Exception:  # pragma: no cover - pynput is optional
    _pynput_keyboard = None


logger = logging.getLogger(__name__)


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
      ``control_step`` at the configured frequency.  This class does not
      override it: the inherited one steps the running skills, drains their
      per-module ``action_queues``, and merges what they commanded into a
      single ``Action`` (as defined by ``settings.ACTION_CLASS``) for the
      base class to send to the robot.

    * :meth:`on_start` spawns a :class:`ThinkProcess` in a separate OS
      process.  The think process reads camera images from the shared
      observation buffers along with proprioceptive sensor data, runs
      VLM-based decision-making (``decide``), and manages skill lifecycles.

    * **Skills** (subclasses of :class:`~solarian.control.ai.system2.skill.Skill`)
      run as individual processes at their own frequencies.  They read
      observations from the shared buffers and push per-module actions into
      their own ``action_queues``.

    Usage::

        from solarian.control.ai.ai_controller import AIController
        from my_project.skills import PickSkill, PlaceSkill

        class MyAIController(AIController):
            skills = (PickSkill, PlaceSkill)

        ctrl = MyAIController(cfg)
        ctrl.connect()
        ctrl.initialize()
        ctrl.send_user_message("Pick up the red cup and place it on the saucer")
        ctrl.start()

    The registry passed to the think process is the controller's own: the
    ``skills`` classes plus any installed skill plugins, keyed by
    :meth:`Skill.registry_name`.  The VLM therefore names a skill class and
    supplies its arguments as params, rather than picking a pre-configured
    instance.

    Prompt templates
    ----------------

    Subclasses drive the VLM with two Jinja2-template class attributes,
    which are read by ``__init__`` and handed to the
    :class:`ThinkProcess`:

    * ``system_prompt`` — the VLM ``system`` message.
    * ``user_prompt`` — the VLM ``user`` message.

    Both templates support:

    * ``{{ user_msg }}``, ``{{ available_skills }}``,
      ``{{ active_skills }}``, ``{{ prev_decide }}``,
      ``{{ recent_skills_status }}`` — context strings built by the
      think process.
    * ``{{ module_name.field_name }}`` — latest value of the
      corresponding observation field (e.g. ``{{ arm.x }}``,
      ``{{ arm.shoulder_pan }}``).
    * ``{{ module_name.field_name[k] }}`` — value at step ``k`` back in
      the observation ring buffer: ``[0]`` = latest, ``[-1]`` =
      previous, ``[-2]`` = two steps before, and so on.  These are
      consecutive entries of the ring buffer (one control-loop tick
      apart), NOT snapshots of prior ``decide()`` calls — use
      ``prev_obs.*`` for that.
    * ``{{ prev_obs.module_name.field_name }}`` — the latest value of
      the field at the time of the *previous* ``decide()`` call.
      Useful to tell whether the last action had the intended effect
      (the previous ring-buffer entry is only one control-tick old and
      will usually be indistinguishable from the latest value).
      ``{{ has_prev_obs }}`` is ``False`` on the first call; guard
      ``prev_obs`` accesses with
      ``{% if has_prev_obs %}...{% endif %}``.
    * ``{{ delta_obs.module_name.field_name }}`` — same shape as
      ``prev_obs``, but holds ``now - prev`` for every numeric field
      (silently dropped for non-numeric fields).  Handy for giving the
      VLM the *change* since the previous decide() call directly,
      without having to subtract in the template.  ``{{ has_delta_obs }}``
      mirrors ``{{ has_prev_obs }}`` and should be used as the same
      guard.
    * ``{{ stagnation_warning }}`` — a short English warning emitted
      once the same start/stop decision has been returned multiple
      times in a row, so the prompt can nudge the VLM to try a
      different action.  Renders as ``"(no stagnation detected)"``
      otherwise.
    * ``{{ expr | filter_name }}`` — pipe the resolved value through a
      function registered in ``prompt_filters`` (see below), or through
      the built-in ``fmt(spec)`` filter which applies a Python format
      spec.  Filters are applied left to right::

          {{ gripper.gripper[0] | cm | fmt('.2f') }}     # → "4.00"
          {{ arm.x | cm | fmt('.2f') }}                  # latest, 2 dp

      When a filter receives a field without an index (``arm.x`` vs
      ``arm.x[0]``), the latest value is auto-unwrapped before the
      filter runs, so filters stay ignorant of the view machinery.
    * Literal JSON braces inside the template don't need escaping —
      only Jinja's ``{{ ... }}`` / ``{% ... %}`` / ``{# ... #}``
      delimiters are interpreted.

    Filters are registered by overriding the ``prompt_filters`` class
    attribute::

        class UR5AIController(AIController):
            prompt_filters = {
                "cm": lambda v: v * 100.0,  # meters → centimeters
            }

    Args:
        cfg: Controller configuration (includes VLM model, connection, etc.).
    """

    # Default prompts.  Subclasses should override ``system_prompt`` and
    # ``user_prompt`` to tailor wording and reference their own
    # observation fields (see the class docstring for supported syntax).
    system_prompt: str = (
        "You are a robot skill orchestrator.\n"
        "You will be provided with the current USER COMMAND, the list of "
        "AVAILABLE SKILLS and their descriptions, the CURRENTLY ACTIVE "
        "skills, and a summary of the PREVIOUS decide() call.\n\n"
        "Decide which skills to start or stop in order to make progress "
        "on the user command.\n\n"
        "Return ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "skills_to_start": [{"skill_name": "...", "params": {...}}],\n'
        '  "skills_to_stop": ["skill_name", ...],\n'
        '  "reasoning": "..."\n'
        "}\n\n"
        "Rules:\n"
        "- No explanation outside the JSON. No markdown.\n"
        "- Only reference skills from the available skills list.\n"
        "- Consider currently active skills before deciding.\n"
        "- If no action is needed, return empty lists.\n"
        "- 'reasoning' MUST be a SINGLE short sentence (<=120 chars), no\n"
        "  newlines, no bullet points, no restating the schema.\n"
    )

    user_prompt: str = (
        "User command: {{ user_msg }}\n\n"
        "Available skills:\n{{ available_skills }}\n\n"
        "Currently active skills:\n{{ active_skills }}\n\n"
        "Previous decide() call:\n{{ prev_decide }}\n\n"
        "Decide which skills to start or stop.\n"
    )

    # Filter registry for the ``| filter`` pipe syntax in prompts.
    # Subclasses override this to register their own transforms.
    prompt_filters: Dict[str, Callable[[Any], Any]] = {}

    def __init__(
        self,
        cfg: ControllerConfig,
    ):
        super().__init__(cfg)

        # Inter-process communication queues.
        # ``observation_buffers`` and ``skill_status_buffer`` come from the
        # base ``Controller`` (shared memory), as does ``skill_command_queue``,
        # which is how the ThinkProcess asks for a skill to start or stop.
        self.user_msg_queue: Queue = multiprocessing.Queue()

        # Shared event to coordinate shutdown
        self._running_event = multiprocessing.Event()

        # VLM configuration extracted from the controller config.
        # ``temperature`` is non-zero on purpose: with temperature=0 the
        # VLM renders near-identical prompts to near-identical outputs,
        # which is great for reproducibility but causes the think loop
        # to get stuck repeating a decision that isn't working.  A
        # little sampling noise lets it escape those fixed points
        # without meaningfully hurting reliability.
        self.vlm_config: Dict[str, Any] = {
            "model": cfg.system2_model,
            "base_url": cfg.system2_base_url,
            "api_key": cfg.system2_api_key,
            "temperature": 0.3,
            "timeout_s": 10.0,
            # When True, the think process mirrors each frame it sends
            # to the VLM (already cropped / colour-corrected / resized
            # / annotated) into an OpenCV window for live debugging.
            # Off by default because it requires a display; set via
            # ``self.vlm_config["debug_show_annotated_image"] = True``
            # in a subclass constructor or at runtime before ``start()``.
            "debug_show_annotated_image": False,
            # Image preparation pipeline knobs.  Defaults assume the
            # observation image is RGB (as published by ModuleObs) and
            # is captured by a fisheye / circular lens with black
            # border around the illuminated disc — both true for the
            # UR5 wrist camera.  Override to False if your camera
            # delivers BGR natively or fills the whole frame.
            "image_is_rgb": True,
            "image_crop_black_borders": True,
            "image_draw_coordinate_grid": True,
            # Overlay the cyan target_bbox rectangle + green-L / red-R
            # finger-tip markers from the previous decide()'s
            # ``observation`` onto the current image.  Only meaningful
            # when the VLM's schema actually emits target_bbox +
            # left_finger_tip + right_finger_tip (e.g. the old bbox
            # schema); with a simpler ``next_move``-only schema,
            # subclasses should set this to False so the overlay
            # function does not draw stale / unrelated markers.
            "image_annotate_prev_obs": True,
            # Declarative ``{move_name: skill_spec | list | None}`` map
            # that translates a VLM-emitted ``next_move`` string into
            # concrete ``skills_to_start`` entries inside
            # ``ThinkProcess.decide()``.  Keys are case-insensitive;
            # ``None`` means "no skill this call" (e.g. DONE).  When
            # this map is empty the decision passes through unchanged
            # and the VLM is expected to emit ``skills_to_start``
            # directly, as before.
            "action_map": {},
        }

        # The menu the VLM picks from is the controller's own skill registry:
        # this subclass's ``skills`` list plus whatever plugins publish under
        # the ``solarian.skills`` entry-point group, keyed by
        # :meth:`Skill.registry_name`. One registry rather than two, so the
        # name the VLM names a skill by is the name the controller resolves.
        self.available_skills: Dict[str, Dict[str, Any]] = self._skill_registry

        self._stdin_thread: Optional[threading.Thread] = None

        # Interactive prompt capture driven by the keyboard module's listener.
        # Pressing Enter toggles ``_prompt_active``; while active every key is
        # consumed by the prompt, so no KEY_DOWN / KEY_UP event reaches the
        # skills.  The current buffer is mirrored on the shared ``StatusBoard``
        # so the operator can see what they are typing.
        self._prompt_active: bool = False
        self._prompt_buffer: str = ""
        self._prompt_lock = threading.Lock()
        self.keyboard.interceptor = self._intercept_key

        self._think_process = ThinkProcess(
            observation_buffers=self.observation_buffers,
            skill_command_queue=self.skill_command_queue,
            skill_status_buffer=self.skill_status_buffer,
            user_msg_queue=self.user_msg_queue,
            vlm_config=self.vlm_config,
            available_skills=self.available_skills,
            system_prompt=type(self).system_prompt,
            user_prompt=type(self).user_prompt,
            prompt_filters=dict(type(self).prompt_filters or {}),
            running_event=self._running_event,
            frequency=self.cfg.tps,
        )

    # ------------------------------------------------------------------
    # Controller interface
    # ------------------------------------------------------------------

    def on_start(self):
        """
        Launch the :class:`ThinkProcess` in a separate OS process.

        Called automatically by ``Controller.start()`` after the state
        receiver and control loop threads are running.  Observation
        buffers are created eagerly in ``Controller.__init__`` so they
        are already populated when the think process is spawned.
        """
        logger.info(
            f"Launching think process with observation buffers: "
            f"{list(self.observation_buffers.keys())}"
        )
        self._running_event.set()
        self._think_process.start()
        logger.info(
            f"Think process started (PID: {self._think_process.pid})"
        )

        # Prefer the pynput-driven prompt capture when a global keyboard
        # listener is running (non-headless + pynput available).  Fall back
        # to a line-buffered stdin reader only in headless mode or when
        # pynput failed to import, since the stdin path requires the user
        # to keep the controller terminal focused and steals keystrokes
        # from anything the base ``Controller`` might do with them.
        if self.cfg.headless or _pynput_keyboard is None:
            self._start_stdin_reader()
        else:
            self._publish_prompt_status()

    # ------------------------------------------------------------------
    # Pynput-driven interactive prompt
    # ------------------------------------------------------------------

    PROMPT_STATUS_KEY = "user prompt"

    def _intercept_key(self, key, is_press: bool) -> bool:
        """Claim keys for the prompt buffer; report whether one was consumed.

        Installed on the keyboard module, so consuming a key here means it
        never becomes a KEY_DOWN / KEY_UP event and the robot does not react
        to the characters being typed.  Enter is claimed unconditionally: it
        is what opens the prompt in the first place.
        """
        if not is_press:
            return self._prompt_active

        if _pynput_keyboard is not None and key == _pynput_keyboard.Key.enter:
            if self._prompt_active:
                self._submit_prompt()
            else:
                self._start_prompt()
            return True

        if self._prompt_active:
            self._handle_prompt_key(key)
            return True

        return False

    def _start_prompt(self) -> None:
        with self._prompt_lock:
            self._prompt_active = True
            self._prompt_buffer = ""
        logger.info("User-prompt capture started (type, then press Enter to submit, Esc to cancel)")
        self._publish_prompt_status()

    def _submit_prompt(self) -> None:
        with self._prompt_lock:
            message = self._prompt_buffer.strip()
            self._prompt_active = False
            self._prompt_buffer = ""
        if message:
            logger.info(f"User command received: {message!r}")
            self.send_user_message(message)
        else:
            logger.info("Empty user prompt, ignored")
        self._publish_prompt_status()

    def _cancel_prompt(self) -> None:
        with self._prompt_lock:
            self._prompt_active = False
            self._prompt_buffer = ""
        logger.info("User-prompt capture cancelled")
        self._publish_prompt_status()

    def _handle_prompt_key(self, key) -> None:
        """Apply a single key event to the prompt buffer (prompt is active)."""
        if _pynput_keyboard is None:
            return

        if key == _pynput_keyboard.Key.esc:
            self._cancel_prompt()
            return

        if key == _pynput_keyboard.Key.backspace:
            with self._prompt_lock:
                self._prompt_buffer = self._prompt_buffer[:-1]
            self._publish_prompt_status()
            return

        if key == _pynput_keyboard.Key.space:
            with self._prompt_lock:
                self._prompt_buffer += " "
            self._publish_prompt_status()
            return

        char = getattr(key, "char", None)
        if char and char.isprintable():
            with self._prompt_lock:
                self._prompt_buffer += char
            self._publish_prompt_status()

    def _publish_prompt_status(self) -> None:
        if self._prompt_active:
            text = f"> {self._prompt_buffer}\u258f"  # thin caret at end
        else:
            text = "(press Enter to type a command)"
        try:
            StatusBoard.set(self.PROMPT_STATUS_KEY, text)
        except Exception:
            logger.debug("StatusBoard.set failed for user-prompt row", exc_info=True)

    # ------------------------------------------------------------------
    # Stdin reader (fallback for headless / missing pynput)
    # ------------------------------------------------------------------

    def _start_stdin_reader(self):
        """Start a daemon thread that reads newline-terminated commands
        from stdin and pushes them onto ``user_msg_queue``."""
        if self._stdin_thread is not None and self._stdin_thread.is_alive():
            return
        if not sys.stdin or not sys.stdin.readable():
            logger.warning("stdin is not readable; user-command input disabled")
            return

        t = threading.Thread(
            target=self._stdin_reader_loop,
            name="AIController-stdin-reader",
            daemon=True,
        )
        t.start()
        self._stdin_thread = t
        logger.info(
            "User-command input ready — type a command and press Enter "
            "to send it to the think process."
        )

    def _stdin_reader_loop(self):
        """Blocking read-line loop; exits when the controller stops."""
        while self._running_event.is_set():
            try:
                line = sys.stdin.readline()
            except Exception as exc:
                logger.warning(f"stdin read failed, stopping reader: {exc}")
                return
            if line == "":
                # EOF (e.g. stdin closed / piped input ended)
                logger.info("stdin closed; user-command input stopped")
                return
            message = line.strip()
            if not message:
                continue
            logger.info(f"User command received: {message!r}")
            self.send_user_message(message)

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
                logger.warning("Think process terminated forcefully")
        super().stop()
