"""
ThinkProcess — Runs the decide loop in a separate OS process.

The think process is the "slow brain" that operates at ~2–5 Hz.  It reads
camera images and proprioceptive sensor data from the shared-memory
observation buffers, feeds them along with the user command and recent
skill status records into a VLM, and decides which skills to launch or
stop (``decide``).

Architecture::

    observation_buffers (shared mem)
            │  read (image + proprioception)
            ▼
    ThinkProcess (~2 Hz)
        decide(image, user_msg, skills_obs) → skills to start / stop
            │  (system_prompt / user_prompt templates reference
            │   observation fields directly, e.g. ``{arm.x[-1]}``)
            │  launch / stop
            ▼
    Skill processes
"""

import logging
import time
import base64
import json
import multiprocessing
from datetime import datetime
from multiprocessing import Process, Queue
from typing import Dict, List, Any, Optional

import numpy as np

from solarian.core.action import Action
from solarian.core.ai.system1.skill import Skill, SkillStatus, SkillRunner
from solarian.core.data import ImageData
from solarian.utils import setup_process_logging, get_log_queue
from solarian.utils.shared_memory.observation_ring_buffer import ObservationRingBuffer
from solarian.utils.shared_memory.action_queue import ActionQueue
from solarian.utils.shared_memory.skill_status_buffer import SkillStatusBuffer


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template namespace helpers
# ---------------------------------------------------------------------------
#
# ``system_prompt`` and ``user_prompt`` are arbitrary ``str.format``-style
# templates supplied by the AIController subclass.  They can reference:
#
#   - Context vars built by ThinkProcess:
#       {user_msg}, {available_skills}, {active_skills}, {prev_decide}
#
#   - Observation fields, written as ``module_name.field_name`` — optionally
#     with a step index selecting which entry in the shared-memory ring
#     buffer to read:
#
#       {arm.x}        → latest value of ``arm.x``
#       {arm.x[0]}     → latest value of ``arm.x`` (same as above)
#       {arm.x[-1]}    → one step back
#       {arm.x[-2]}    → two steps back
#       {arm.x[0]:.2f} → latest, formatted with a format spec
#
# Internally, the module name resolves to a ``_ModuleView`` and the field
# name to a ``_FieldView``, which implement attribute / item / format
# access so Python's ``str.format`` parser (which already understands
# ``{name.attr[idx]:spec}``) can walk the template without a custom parser.


class _FieldView:
    """A ``str.format`` proxy for a single observation field.

    Wraps a chronologically-ordered (oldest-first) list of raw field values.
    Indexing reinterprets the sequence as "steps back from now":

        view[0]   → latest value (last in buffer)
        view[-1]  → one step before latest
        view[-k]  → k steps before latest

    Formatting the view directly (``f"{view}"``, ``{arm.x:.2f}``) renders
    the latest value — i.e. it behaves like ``view[0]``.
    """

    __slots__ = ("_values", "_full_name")

    def __init__(self, values: List[Any], full_name: str = ""):
        # ``values`` is oldest-first, matching ``ObservationRingBuffer.get_last_k``.
        self._values = values
        self._full_name = full_name

    def _resolve(self, idx) -> Any:
        # ``str.format`` passes ``[-1]`` in a template as the string
        # ``"-1"`` (only pure-digit indices are auto-converted to int),
        # so accept both int and int-looking strings here.
        if isinstance(idx, bool):
            raise TypeError(
                f"Observation index for '{self._full_name}' must be an int, "
                f"got bool"
            )
        if isinstance(idx, str):
            try:
                idx = int(idx)
            except ValueError:
                raise TypeError(
                    f"Observation index for '{self._full_name}' must be an int, "
                    f"got {idx!r}"
                )
        if not isinstance(idx, int):
            raise TypeError(
                f"Observation index for '{self._full_name}' must be an int, "
                f"got {type(idx).__name__}"
            )
        if idx > 0:
            raise IndexError(
                f"Observation index for '{self._full_name}' must be <= 0 "
                f"(0 = latest, -1 = previous, ...); got {idx}"
            )
        pos = len(self._values) - 1 + idx  # idx=0 → last; idx=-1 → second to last
        if pos < 0:
            raise IndexError(
                f"Observation index {idx} out of range for '{self._full_name}' "
                f"(only {len(self._values)} value(s) available)"
            )
        return self._values[pos]

    def __getitem__(self, idx):
        return self._resolve(idx)

    def __format__(self, spec: str) -> str:
        return format(self._resolve(0), spec)

    def __str__(self) -> str:
        return format(self._resolve(0), "")

    def __repr__(self) -> str:
        return f"_FieldView({self._full_name!r}, latest={self._resolve(0)!r})"


class _ModuleView:
    """A ``str.format`` proxy for a single observation module.

    Attribute / item access returns a :class:`_FieldView` for the requested
    field, so ``{arm.x}`` resolves to ``arm_view.x`` which is a
    ``_FieldView`` over ``arm.x`` across recent observations.
    """

    __slots__ = ("_fields", "_name")

    def __init__(self, fields: Dict[str, _FieldView], name: str = ""):
        self._fields = fields
        self._name = name

    def __getattr__(self, name):
        # __getattr__ is only called when normal attribute lookup fails,
        # so slots like _fields / _name are handled by the default path.
        fields = object.__getattribute__(self, "_fields")
        if name in fields:
            return fields[name]
        raise AttributeError(
            f"Module '{self._name}' has no observation field '{name}'. "
            f"Available: {sorted(fields)}"
        )

    def __getitem__(self, name):
        return self._fields[name]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_data_url(
    image: np.ndarray,
    max_dim: Optional[int] = None,
    jpeg_quality: int = 80,
) -> str:
    """Encode a numpy image as a JPEG base64 data URL for VLM input.

    Args:
        image: HxWxC numpy image.
        max_dim: If set and the image's longest side exceeds this value,
            the image is downscaled (aspect-preserving) to cap the longest
            side at ``max_dim`` pixels.  Reduces upload bandwidth and
            VLM latency with minimal impact on scene-level understanding.
        jpeg_quality: JPEG quality 0..100 (default 80).
    """
    import cv2

    if max_dim is not None and max_dim > 0:
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest > max_dim:
            scale = max_dim / float(longest)
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    success, encoded = cv2.imencode(".jpg", image, encode_params)
    if not success:
        raise ValueError("Failed to JPEG-encode image")
    b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _parse_json(text: str) -> Dict[str, Any]:
    """Parse JSON from VLM output, stripping common markdown wrappers.

    Falls back to extracting the outermost ``{...}`` block when the VLM
    appends prose before/after the JSON payload.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


# ---------------------------------------------------------------------------
# ThinkProcess
# ---------------------------------------------------------------------------

class ThinkProcess(Process):
    """
    Runs the decide loop at :data:`frequency` Hz in a separate OS process.

    It reads the latest camera image and proprioceptive scalars from
    shared-memory observation buffers, incorporates the user command and
    skill status reports, and asks a VLM which skills to launch or stop
    (``decide``).

    Args:
        observation_buffers: Shared-memory ``ObservationRingBuffer`` dict
            (module_name → buffer).
        action_queue: Shared-memory :class:`ActionQueue` that both locally
            running skills and this process push top-level :class:`Action`
            objects into.  The controller's ``control_step`` drains it.
        skill_status_buffer: Skills put status records here for the think
            process.
        user_msg_queue: User commands arrive here.
        vlm_config: Dict with keys ``model``, ``base_url``, ``api_key``,
            ``temperature``, ``timeout_s``.  Optional keys:
            ``image_max_dim`` (int, longest-side pixel cap before
            JPEG-encoding; ``None`` disables resizing — default 512),
            ``image_jpeg_quality`` (int 0..100, default 80).
        available_skills: Registry mapping skill name → metadata dict
            ``{"cls": SkillClass, "runner": SkillRunner,
              "action_type": int | None, "docstring": str}``.
        system_prompt: ``str.format``-style template for the VLM ``system``
            message.  See the module docstring for the supported variables
            (observation fields + context vars).
        user_prompt: ``str.format``-style template for the VLM ``user``
            message.  Same variables as ``system_prompt``, plus
            ``{user_msg}``, ``{available_skills}``, ``{active_skills}``,
            ``{prev_decide}``.
        running_event: Cleared to signal the process to shut down.
    """

    def __init__(
        self,
        observation_buffers: Dict[str, ObservationRingBuffer],
        action_queue: ActionQueue,
        skill_status_buffer: SkillStatusBuffer,
        user_msg_queue: Queue,
        vlm_config: Dict[str, Any],
        available_skills: Dict[str, Dict[str, Any]],
        system_prompt: str,
        user_prompt: str,
        running_event: Any,
        frequency: float = 2.0,
    ):
        super().__init__(name="ThinkProcess", daemon=True)
        self.observation_buffers = observation_buffers
        self.action_queue = action_queue
        self.skill_status_buffer = skill_status_buffer
        self.user_msg_queue = user_msg_queue
        self.vlm_config = vlm_config
        self.available_skills = available_skills
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.frequency = frequency
        self.running_event = running_event

        # Populated at runtime inside the child process.  The VLM client is
        # NOT created here because it holds un-picklable ``_thread.RLock``
        # objects (via httpx) and this process is launched with the
        # ``spawn`` start method, which pickles the ``Process`` instance.
        self.active_skills: Dict[str, Skill] = {}
        self.skills_obs_history: List[Dict] = []
        self.last_user_msg: Optional[str] = None
        self.client: Any = None

        # Snapshot of the previous decide() call — proprio + decision —
        # so the next prompt can show the VLM what it just saw and did.
        # Keeping only the last one (no image) to cap token usage.
        self._last_decide: Optional[Dict[str, Any]] = None

        # skill_id → human-readable skill name, derived from the registry
        # so that observation logs show "arm_skill / running" instead of
        # raw integer ids.  Built lazily inside the child process.
        self._skill_id_to_name: Dict[int, str] = {}

        # Capture the parent's log queue so the child process can forward
        # logging records back to the StatusBoard running in the main process.
        self._log_queue = get_log_queue()

    # ------------------------------------------------------------------
    # VLM interaction
    # ------------------------------------------------------------------

    def _create_vlm_client(self):
        from openai import OpenAI

        # Silence HTTP-layer DEBUG logs from openai / httpx / httpcore —
        # they would otherwise dump the full base64-encoded image payload
        # into the StatusBoard log panel on every VLM call.
        for noisy in ("openai", "httpx", "httpcore", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        self.client = OpenAI(
            base_url=self.vlm_config["base_url"],
            api_key=self.vlm_config["api_key"],
        )

    def _chat(
        self,
        system: str,
        user_text: str,
        image_data_url: Optional[str] = None,
        max_tokens: int = 512,
    ) -> str:
        """Send a single-turn chat completion to the VLM."""
        user_content: list = [{"type": "text", "text": user_text}]
        if image_data_url:
            user_content.append(
                {"type": "image_url", "image_url": {"url": image_data_url}}
            )

        resp = self.client.chat.completions.create(
            model=self.vlm_config["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=self.vlm_config.get("temperature", 0.0),
            max_tokens=max_tokens,
            timeout=self.vlm_config.get("timeout_s", 10.0),
        )
        return (resp.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(
        self,
        image: Optional[np.ndarray],
        user_msg: str,
        skills_obs: List[Dict],
    ) -> Dict:
        """
        Decide which skills to start or stop.

        Formats the subclass-supplied ``system_prompt`` / ``user_prompt``
        templates with the latest observation snapshot plus context
        variables (``user_msg``, ``available_skills``, ``active_skills``,
        ``prev_decide``), optionally attaches the scene image, sends the
        result to the VLM, and parses its JSON response.

        Args:
            image: Latest camera image (H, W, C) to attach to the VLM call.
                Passing ``None`` sends a text-only turn.
            user_msg: Latest user command.
            skills_obs: Recent skill status records (see ``_think_step``).

        Returns:
            Dict with ``skills_to_start``, ``skills_to_stop``, ``reasoning``.
        """
        available_text = "\n".join(
            f"- {name}: {spec.get('docstring') or 'No description'}"
            for name, spec in self.available_skills.items()
        )
        active_text = (
            "\n".join(f"- {name}: running" for name in self.active_skills)
            or "None"
        )
        obs_text = self._format_skills_obs(skills_obs[-10:]) if skills_obs else "(none)"
        prev_text = self._format_previous_call()

        observation_ns = self._build_observation_namespace()

        fmt_vars: Dict[str, Any] = {
            "user_msg": user_msg,
            "available_skills": available_text,
            "active_skills": active_text,
            "prev_decide": prev_text,
            "recent_skill_obs": obs_text,
            **observation_ns,
        }

        try:
            system_text = self.system_prompt.format(**fmt_vars)
            user_text = self.user_prompt.format(**fmt_vars)
        except (KeyError, AttributeError, IndexError, TypeError) as exc:
            logger.warning(
                "Prompt formatting failed (%s: %s); skipping decide() step. "
                "This is usually transient at startup while observation "
                "buffers are still filling.",
                type(exc).__name__, exc,
            )
            return {
                "skills_to_start": [],
                "skills_to_stop": [],
                "reasoning": "prompt_format_failure",
            }

        logger.info("decide() system prompt:\n%s", system_text)
        logger.info("decide() user prompt:\n%s", user_text)

        image_data_url: Optional[str] = None
        if image is not None:
            image_data_url = _image_to_data_url(
                image,
                max_dim=self.vlm_config.get("image_max_dim", 512),
                jpeg_quality=self.vlm_config.get("image_jpeg_quality", 80),
            )

        raw = self._chat(system_text, user_text, image_data_url, max_tokens=400)
        decision: Optional[Dict] = None
        retried = False
        try:
            decision = _parse_json(raw)
        except Exception:
            retried = True
            repair = (
                user_text
                + "\nINVALID. Return ONLY valid JSON matching the schema.\n"
            )
            raw2 = self._chat(
                system_text, repair, image_data_url, max_tokens=400
            )
            try:
                decision = _parse_json(raw2)
            except Exception as exc:
                logger.error(f"Decision failed: {exc}")
                decision = {
                    "skills_to_start": [],
                    "skills_to_stop": [],
                    "reasoning": "decision_parse_failure",
                }

        logger.info(
            "decide() result%s: start=%s stop=%s reasoning=%s",
            " (retry)" if retried else "",
            decision.get("skills_to_start"),
            decision.get("skills_to_stop"),
            decision.get("reasoning"),
        )

        # Snapshot this call so the next prompt can show the VLM what it
        # just saw and did.  We store the rendered user prompt (which
        # already embeds the proprioception the subclass chose to show)
        # rather than a separate proprio dict, so the "prev_decide" block
        # stays in lockstep with whatever the template exposes.
        self._last_decide = {
            "timestamp": time.time(),
            "user_text": user_text,
            "decision": decision,
        }

        return decision

    # ------------------------------------------------------------------
    # Prompt templating helpers
    # ------------------------------------------------------------------

    def _build_observation_namespace(self) -> Dict[str, _ModuleView]:
        """Build ``{module_name: _ModuleView}`` from the shared-memory buffers.

        For each observation module we read up to ``get_max_k`` recent
        entries so the template can walk backwards in time via
        ``{module.field[-k]}``.  Image fields are skipped because they
        are not format-printable (the scene image is attached to the
        chat call separately).
        """
        namespace: Dict[str, _ModuleView] = {}
        for module_name, buffer in self.observation_buffers.items():
            count = buffer.count
            if count == 0:
                continue
            k = min(count, getattr(buffer, "get_max_k", count))
            try:
                observations = buffer.get_last_k(k)
            except Exception as exc:
                logger.debug(
                    "buffer.get_last_k(%d) failed for module '%s': %s",
                    k, module_name, exc,
                )
                continue

            # Re-shape module_obs list into per-field value lists, skipping
            # image / non-numeric fields that the prompt can't format.
            fields: Dict[str, List[Any]] = {}
            for module_obs in observations:
                for field_name, data in module_obs.items():
                    if isinstance(data, ImageData):
                        continue
                    fields.setdefault(field_name, []).append(data.payload)

            module_fields: Dict[str, _FieldView] = {
                field_name: _FieldView(values, f"{module_name}.{field_name}")
                for field_name, values in fields.items()
            }
            namespace[module_name] = _ModuleView(module_fields, module_name)
        return namespace

    def _format_previous_call(self) -> str:
        """Render the previous decide() call (user prompt + decision).

        Returns ``"(none)"`` on the first call.  Only textual context is
        included — the image from the previous call is deliberately
        excluded to keep token usage low.
        """
        prev = self._last_decide
        if not prev:
            return "(none)"

        age = max(0.0, time.time() - float(prev.get("timestamp", 0.0)))
        prev_user_text = (prev.get("user_text") or "").rstrip()

        decision = prev.get("decision", {}) or {}
        started = decision.get("skills_to_start") or []
        stopped = decision.get("skills_to_stop") or []
        reasoning = decision.get("reasoning") or ""

        if started:
            started_lines = "\n".join(
                f"    - {spec.get('skill_name')} params={spec.get('params', {})}"
                for spec in started
            )
        else:
            started_lines = "    (none)"

        if stopped:
            stopped_lines = "\n".join(f"    - {name}" for name in stopped)
        else:
            stopped_lines = "    (none)"

        return (
            f"  {age:.1f}s ago\n"
            f"  prompt then:\n{prev_user_text}\n"
            f"  actions taken then:\n"
            f"    started:\n{started_lines}\n"
            f"    stopped:\n{stopped_lines}\n"
            f"    reasoning: {reasoning}"
        )

    # ------------------------------------------------------------------
    # Skill obs formatting
    # ------------------------------------------------------------------

    def _build_skill_id_to_name(self) -> None:
        """Build the ``skill_id → skill_name`` map from the registry.

        Each entry in ``available_skills`` may carry an explicit
        ``skill_id``; entries without one are assigned a sequential id
        starting at 1 (0 stays reserved for "unknown").  This is called
        once inside the child process at startup so the prompt formatter
        can resolve ids back to readable names.
        """
        mapping: Dict[int, str] = {}
        next_id = 1
        for name, spec in self.available_skills.items():
            sid = spec.get("skill_id")
            if sid is None or int(sid) <= 0:
                while next_id in mapping:
                    next_id += 1
                sid = next_id
                next_id += 1
            mapping[int(sid)] = name
        self._skill_id_to_name = mapping

    @staticmethod
    def _status_name(status: int) -> str:
        try:
            return SkillStatus(int(status)).name.lower()
        except ValueError:
            return f"status_{int(status)}"

    @staticmethod
    def _fmt_timestamp(ts: float) -> str:
        ts = float(ts)
        if ts <= 0.0:
            return "--:--:--"
        try:
            return datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        except (OverflowError, OSError, ValueError):
            return "--:--:--"

    def _format_skills_obs(self, records: List[Dict]) -> str:
        """Render recent skill status records for the decide() prompt.

        Produces one human-readable line per record, e.g.::

            - 19:42:53 arm_skill: running
            - 19:42:53 gripper_skill: completed
        """
        lines: List[str] = []
        for rec in records:
            sid = int(rec.get("skill_id", 0))
            name = self._skill_id_to_name.get(sid) or f"skill_{sid}"
            status = self._status_name(rec.get("status", 0))
            ts = self._fmt_timestamp(rec.get("timestamp", 0.0))
            lines.append(f"- {ts} {name}: {status}")
        return "\n".join(lines) if lines else "(none)"

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    # TODO: could be more than one image and the images to be used should be configurable
    def _get_latest_image(self) -> Optional[np.ndarray]:
        """Return the payload of the first :class:`ImageData` field found
        in any observation buffer, or ``None`` if nothing has been
        published yet.

        The ring buffer's ``get()`` is non-destructive, so reading here
        does not affect skill processes that also read the same buffer.
        """
        for module_name, buffer in self.observation_buffers.items():
            if buffer.count == 0:
                continue
            try:
                module_obs = buffer.get()
            except Exception as exc:
                logger.debug(
                    "buffer.get() failed for module '%s': %s",
                    module_name, exc,
                )
                continue
            for field_name, data in module_obs.items():
                if isinstance(data, ImageData) and data.payload is not None:
                    return data.payload
        return None

    def _drain_queue(self, queue: Queue) -> List:
        """Non-blocking drain of all items currently in *queue*."""
        items: List = []
        while True:
            try:
                items.append(queue.get_nowait())
            except Exception:
                break
        return items

    # ------------------------------------------------------------------
    # Skill management
    # ------------------------------------------------------------------

    def _cleanup_finished_skills(self):
        finished = [n for n, s in self.active_skills.items() if not s.is_alive()]
        for name in finished:
            del self.active_skills[name]

    def _launch_skill(self, skill_name: str, params: Dict):
        spec = self.available_skills.get(skill_name)
        if spec is None:
            logger.warning(f"Unknown skill: {skill_name}")
            return

        runner: SkillRunner = spec.get("runner", SkillRunner.CONTROLLER)

        # Remote skills (ROBOT / SERVER) are not spawned here.  Instead we
        # push a top-level Action onto the shared-memory action queue so
        # ``control_step``/``control_loop`` can forward it over the wire,
        # and the remote side's action handler is what actually starts the
        # corresponding Skill.
        if runner != SkillRunner.CONTROLLER:
            action_type = spec.get("action_type")
            if action_type is None:
                logger.error(
                    f"Skill '{skill_name}' declared runner={runner.name} "
                    f"but has no action_type — cannot dispatch."
                )
                return
            try:
                action = Action(action_type=int(action_type), data=dict(params or {}))
                self.action_queue.put(action)
                logger.info(
                    f"Enqueued action (action_type={int(action_type)}) "
                    f"to launch remote skill '{skill_name}' on {runner.name}"
                )
            except Exception as exc:
                logger.error(f"Failed to enqueue action for '{skill_name}': {exc}")
            return

        # Controller-side skill — spawn it as a subprocess and track its
        # lifecycle locally.
        if skill_name in self.active_skills:
            logger.info(f"Skill '{skill_name}' already active, skipping")
            return

        cls = spec["cls"]
        skill = cls(
            name=skill_name,
            skill_id=spec.get("skill_id"),
            observation_buffers=self.observation_buffers,
            action_queue=self.action_queue,
            skills_status_buffer=self.skill_status_buffer,
            params=params,
            runner=runner,
            action_type=spec.get("action_type"),
        )
        skill.start()
        self.active_skills[skill_name] = skill
        logger.info(f"Launched skill '{skill_name}' (PID {skill.pid})")

    def _stop_skill(self, skill_name: str):
        skill = self.active_skills.get(skill_name)
        if skill is not None and skill.is_alive():
            skill.request_stop()
            logger.info(f"Requested stop for skill '{skill_name}'")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _think_step(self):
        """Execute one decide cycle."""
        # 1. Drain incoming skill-status records (shared memory) and queues
        for rec in self.skill_status_buffer.drain():
            self.skills_obs_history.append(
                {
                    "skill_id": rec.skill_id,
                    "status": rec.status,
                    "timestamp": rec.timestamp,
                }
            )

        user_msgs = self._drain_queue(self.user_msg_queue)
        if user_msgs:
            self.last_user_msg = user_msgs[-1]
            logger.info(f"Received new user command: {self.last_user_msg!r}")

        self._cleanup_finished_skills()

        # Nothing to do until the user sends a command
        if not self.last_user_msg:
            return

        # 2. Get latest image from observation buffers (optional — templates
        # that don't rely on image input can run without one).
        image = self._get_latest_image()
        if image is None:
            logger.debug("No image available yet, skipping think step")
            return

        # 3. Decide — which skills to start / stop
        decision = self.decide(
            image, self.last_user_msg, self.skills_obs_history
        )

        # 4. Act on the decision
        for spec in decision.get("skills_to_start", []):
            self._launch_skill(spec["skill_name"], spec.get("params", {}))

        for name in decision.get("skills_to_stop", []):
            self._stop_skill(name)

    def run(self):
        """Process entry point — loops ``_think_step``."""
        # Route this child process's log records back to the parent's
        # log queue so they show up in the StatusBoard Logs panel.
        setup_process_logging(
            logger_name=None,
            log_queue=self._log_queue,
            install_fallback=False,
        )

        logger.info(
            f"Think process started (PID {multiprocessing.current_process().pid}), "
            f"frequency={self.frequency} Hz"
        )
        logger.info(
            "Observation buffers visible in child: %s",
            list(self.observation_buffers.keys()) or "(none)",
        )
        logger.info(
            "Available skills: %s",
            list(self.available_skills.keys()) or "(none)",
        )

        # VLM client must be constructed inside the child process — the
        # underlying httpx client holds thread locks that can't be pickled
        # across the spawn boundary.
        self._create_vlm_client()

        # Resolve skill_id → name once so observation logs in the prompt
        # show readable names instead of raw ids.
        self._build_skill_id_to_name()
        logger.info("Skill id → name map: %s", self._skill_id_to_name)

        while self.running_event.is_set():
            t0 = time.monotonic()
            try:
                self._think_step()
            except Exception as exc:
                logger.exception(f"Error in think step: {exc}")

            elapsed = time.monotonic() - t0
            sleep_time = (1.0 / self.frequency) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Graceful shutdown — stop all active skills
        for name, skill in list(self.active_skills.items()):
            if skill.is_alive():
                skill.request_stop()
        for name, skill in list(self.active_skills.items()):
            skill.join(timeout=3.0)
            if skill.is_alive():
                skill.terminate()

        logger.info("Think process stopped")
