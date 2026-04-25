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
        decide(image, user_msg, skills_status) → skills to start / stop
            │  (system_prompt / user_prompt templates reference
            │   observation fields directly, e.g. ``{arm.x[-1]}``,
            │   and may pipe values through named filters, e.g.
            │   ``{arm.x[-1] | cm:.2f}``)
            │  launch / stop
            ▼
    Skill processes
"""

import copy
import logging
import time
import base64
import json
import multiprocessing
from multiprocessing import Process, Queue
from typing import Callable, Dict, List, Any, Optional

import jinja2
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
# ``system_prompt`` and ``user_prompt`` are Jinja2 templates supplied by
# the AIController subclass.  They can reference:
#
#   - Context vars built by ThinkProcess:
#       {{ user_msg }}, {{ available_skills }}, {{ active_skills }},
#       {{ prev_decide }}, {{ recent_skills_status }}
#
#   - Observation fields, written as ``module_name.field_name`` — optionally
#     with a step index selecting which entry in the shared-memory ring
#     buffer to read:
#
#       {{ arm.x }}        → latest value of ``arm.x``
#       {{ arm.x[0] }}     → latest value (same as above)
#       {{ arm.x[-1] }}    → one step back
#       {{ arm.x[-2] }}    → two steps back
#
#   - Named filters applied to the value via ``|`` pipes.  Filters are
#     registered on the subclass through ``prompt_filters``.  When a
#     pipe receives a ``_FieldView`` (e.g. ``arm.x`` without an index)
#     the view is auto-unwrapped to its latest value before the filter
#     runs:
#
#       {{ gripper.gripper[0] | cm | fmt('.2f') }}   → "4.00"
#       {{ arm.x | cm | round(2) }}                  → latest in cm, 2 dp
#
#   - Built-in filter ``fmt(spec)`` applies a Python format spec, e.g.
#     ``fmt('.2f')``; use it for fixed-width / trailing-zero formatting
#     (Jinja's ``round`` returns a float, so ``4.2`` prints as ``4.2``
#     instead of ``4.20``).
#
# Internally, the module name resolves to a ``_ModuleView`` and the field
# name to a ``_FieldView``.  The Jinja2 ``Environment`` lives in the child
# process because compiled templates aren't picklable across ``spawn``.


class _FieldView:
    """A template proxy for a single observation field.

    Wraps a chronologically-ordered (oldest-first) list of raw field values.
    Indexing reinterprets the sequence as "steps back from now":

        view[0]   → latest value (last in buffer)
        view[-1]  → one step before latest
        view[-k]  → k steps before latest

    Used without indexing in a Jinja template (``{{ arm.x }}``), it renders
    the latest value via ``__str__``.  Piping a bare view through a filter
    (``{{ arm.x | cm }}``) also resolves to the latest value because
    :func:`_unwrap_view` is applied inside the filter wrapper.
    """

    __slots__ = ("_values", "_full_name")

    def __init__(self, values: List[Any], full_name: str = ""):
        # ``values`` is oldest-first, matching ``ObservationRingBuffer.get_last_k``.
        self._values = values
        self._full_name = full_name

    def _resolve(self, idx) -> Any:
        # Jinja passes ``[-1]`` indices as ``int``, but accept int-looking
        # strings too so the view behaves the same if rendered through
        # any other templating engine (e.g. ``str.format``) in tests.
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
    """A template proxy for a single observation module.

    Attribute / item access returns a :class:`_FieldView` for the requested
    field, so ``{{ arm.x }}`` resolves to ``arm_view.x`` which is a
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


def _unwrap_view(value: Any) -> Any:
    """Resolve a ``_FieldView`` to its latest value; pass everything else through.

    Lets users write ``{{ arm.x | cm }}`` as shorthand for
    ``{{ arm.x[0] | cm }}`` — the filter receives a plain float instead
    of a view, so filters can stay ignorant of the view machinery.
    """
    if isinstance(value, _FieldView):
        return value._resolve(0)
    return value


def _wrap_filter(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a user-supplied filter so its first arg is auto-unwrapped."""

    def _wrapped(value, *args, **kwargs):
        return fn(_unwrap_view(value), *args, **kwargs)

    return _wrapped


def _build_prompt_env(
    filters: Dict[str, Callable[[Any], Any]],
) -> jinja2.Environment:
    """Build the Jinja2 environment used to render prompt templates.

    Called from inside the child process (compiled templates carry
    references to the environment and are not picklable across the
    ``spawn`` start method used to launch :class:`ThinkProcess`).
    """
    env = jinja2.Environment(
        autoescape=False,
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # ``fmt`` applies a Python format spec — useful because Jinja's
    # built-in ``round`` returns a float, so ``4.2`` prints as ``"4.2"``
    # rather than ``"4.20"``.  ``fmt('.2f')`` gives the fixed-width
    # output that prompts typically want.
    env.filters["fmt"] = lambda v, spec="": format(_unwrap_view(v), spec)

    for name, fn in filters.items():
        env.filters[name] = _wrap_filter(fn)

    return env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_for_vlm(
    image: np.ndarray,
    max_dim: Optional[int],
) -> np.ndarray:
    """Aspect-preserving downscale so the longest side is ``<= max_dim`` px.

    Returns the input unchanged when ``max_dim`` is falsy or already
    satisfied.  Kept as a separate step from JPEG encoding so callers
    can overlay annotations on the *exact* image the VLM will receive
    (annotations done in the post-resize frame are in the same
    coordinate system as anything the VLM reports about that image).
    """
    import cv2

    if not max_dim or max_dim <= 0:
        return image
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / float(longest)
    new_size = (
        max(1, int(round(w * scale))),
        max(1, int(round(h * scale))),
    )
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _crop_black_borders(
    image: np.ndarray,
    luma_threshold: int = 8,
    margin: int = 0,
) -> np.ndarray:
    """Crop to the tight bbox of pixels whose max channel exceeds the threshold.

    Eliminates the large black borders that fisheye / circular wrist
    cameras leave around their illuminated region.  Computes the
    row/column masks of "non-black" pixels once and slices — no copy
    unless the caller mutates the result, because NumPy slicing
    returns a view.

    Returns the input unchanged when no pixel exceeds the threshold
    (e.g. fully black frame during startup).
    """
    if image.ndim == 3:
        lum = image.max(axis=2)
    else:
        lum = image
    mask = lum > luma_threshold
    if not mask.any():
        return image
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    if margin > 0:
        h, w = image.shape[:2]
        y0 = max(0, y0 - margin)
        y1 = min(h, y1 + margin)
        x0 = max(0, x0 - margin)
        x1 = min(w, x1 + margin)
    if (y0, y1, x0, x1) == (0, image.shape[0], 0, image.shape[1]):
        return image
    return image[y0:y1, x0:x1]


def _rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Swap R and B channels so cv2 (BGR) operations produce correct colours.

    The wrist-camera observation is published in RGB order; cv2's
    ``imencode`` / ``imshow`` both treat the first channel as blue,
    which produces a JPEG with red and blue swapped when given RGB
    input directly.  Using fancy indexing (``[..., ::-1]``) is faster
    and has no OpenCV dependency, so this works even on frames where
    the cv2 import fails early.
    """
    if image.ndim == 3 and image.shape[2] >= 3:
        return np.ascontiguousarray(image[..., ::-1])
    return image


def _draw_coordinate_grid(
    image: np.ndarray,
    color: tuple = (80, 80, 80),
) -> np.ndarray:
    """Overlay a light 25 / 50 / 75 pixel-percent reference grid.

    Gives the VLM explicit anchor points when it is asked to emit
    pixel-percent bbox / finger-tip coordinates.  Without this the
    model tends to hallucinate "round" values (``[30, 95]``, ``[50,
    50]``) that don't correspond to any real feature in the frame.

    The grid is drawn on a copy so upstream buffers are never mutated.
    Labels are rendered in two passes (black outline + coloured
    foreground) so they stay legible over bright and dark regions
    alike.
    """
    import cv2

    out = image.copy()
    h, w = out.shape[:2]
    for frac in (0.25, 0.5, 0.75):
        x = int(round(w * frac))
        y = int(round(h * frac))
        cv2.line(out, (x, 0), (x, h - 1), color, 1, cv2.LINE_AA)
        cv2.line(out, (0, y), (w - 1, y), color, 1, cv2.LINE_AA)

    for frac, label in ((0.25, "25"), (0.5, "50"), (0.75, "75")):
        x = int(round(w * frac))
        y = int(round(h * frac))
        for pos in ((x + 3, 11), (3, y + 11)):
            cv2.putText(
                out, label, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (0, 0, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                out, label, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                color, 1, cv2.LINE_AA,
            )
    return out


def _encode_jpeg_data_url(image: np.ndarray, jpeg_quality: int = 80) -> str:
    """Encode an already-sized numpy image as a JPEG base64 data URL."""
    import cv2

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    success, encoded = cv2.imencode(".jpg", image, encode_params)
    if not success:
        raise ValueError("Failed to JPEG-encode image")
    b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _pct_to_px(
    pt: Any, width: int, height: int
) -> Optional[tuple]:
    """Convert a ``[x_pct, y_pct]`` pair (0..100) to integer pixel coords.

    Returns ``None`` when the input isn't a 2-element numeric sequence
    so callers can skip drawing rather than crash on malformed VLM output.
    """
    if not isinstance(pt, (list, tuple)) or len(pt) != 2:
        return None
    try:
        x = float(pt[0])
        y = float(pt[1])
    except (TypeError, ValueError):
        return None
    return (
        max(0, min(width - 1, int(round(x / 100.0 * width)))),
        max(0, min(height - 1, int(round(y / 100.0 * height)))),
    )


def _annotate_image_with_prev_obs(
    image: np.ndarray,
    observation: Optional[Dict[str, Any]],
) -> np.ndarray:
    """Overlay the previous decide()'s observation onto *image*.

    Draws, in-place on a copy (original buffer is never mutated):

      - ``target_bbox``        — cyan rectangle with "prev target_bbox" label
      - ``left_finger_tip``    — green circle labelled "L"
      - ``right_finger_tip``   — red circle labelled "R"
      - line between the two finger tips (orange) when both present

    All coordinates in *observation* are expected to be pixel-percents
    (0..100) in the same frame as *image*.  Malformed / missing fields
    are silently skipped so a garbage observation from the VLM can't
    crash the think loop.

    Colour choice is BGR (OpenCV native) and picked for contrast
    against typical lab scenes; if the target happens to be cyan the
    rectangle outline will still be visible at 2 px thickness.
    """
    if not isinstance(observation, dict) or not observation:
        return image

    import cv2

    out = image.copy()
    h, w = out.shape[:2]

    bbox = observation.get("target_bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            x0 = int(round(float(bbox[0]) / 100.0 * w))
            y0 = int(round(float(bbox[1]) / 100.0 * h))
            x1 = int(round(float(bbox[2]) / 100.0 * w))
            y1 = int(round(float(bbox[3]) / 100.0 * h))
        except (TypeError, ValueError):
            pass
        else:
            x0, x1 = sorted((max(0, min(w - 1, x0)), max(0, min(w - 1, x1))))
            y0, y1 = sorted((max(0, min(h - 1, y0)), max(0, min(h - 1, y1))))
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 0), 2)
            cv2.putText(
                out, "prev target_bbox",
                (x0, max(10, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 0), 1, cv2.LINE_AA,
            )

    left = _pct_to_px(observation.get("left_finger_tip"), w, h)
    right = _pct_to_px(observation.get("right_finger_tip"), w, h)
    if left is not None:
        cv2.circle(out, left, 6, (0, 255, 0), 2)
        cv2.putText(
            out, "L", (left[0] + 8, left[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (0, 255, 0), 1, cv2.LINE_AA,
        )
    if right is not None:
        cv2.circle(out, right, 6, (0, 0, 255), 2)
        cv2.putText(
            out, "R", (right[0] + 8, right[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (0, 0, 255), 1, cv2.LINE_AA,
        )
    if left is not None and right is not None:
        cv2.line(out, left, right, (0, 165, 255), 1)

    return out


def _image_to_data_url(
    image: np.ndarray,
    max_dim: Optional[int] = None,
    jpeg_quality: int = 80,
) -> str:
    """Encode a numpy image as a JPEG base64 data URL for VLM input.

    Thin wrapper around :func:`_resize_for_vlm` +
    :func:`_encode_jpeg_data_url` kept for callers that don't need to
    annotate the image between resize and encode.
    """
    return _encode_jpeg_data_url(
        _resize_for_vlm(image, max_dim), jpeg_quality=jpeg_quality
    )


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
        system_prompt: Jinja2 template for the VLM ``system`` message.
            See the module docstring for the supported variables
            (observation fields + context vars).
        user_prompt: Jinja2 template for the VLM ``user`` message.
            Same variables as ``system_prompt``, plus ``{{ user_msg }}``,
            ``{{ available_skills }}``, ``{{ active_skills }}``,
            ``{{ prev_decide }}``, ``{{ recent_skills_status }}``.
        prompt_filters: Mapping ``filter_name → callable`` registered on
            the Jinja environment as filters.  Filter inputs are
            auto-unwrapped (a ``_FieldView`` without an index resolves
            to its latest value before the filter runs), so e.g.
            ``{"cm": lambda v: v * 100.0}`` lets templates write
            ``{{ arm.x | cm | fmt('.2f') }}``.
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
        prompt_filters: Optional[Dict[str, Callable[[Any], Any]]],
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
        self.prompt_filters: Dict[str, Callable[[Any], Any]] = dict(prompt_filters or {})
        self.frequency = frequency
        self.running_event = running_event

        # Populated at runtime inside the child process.  The VLM client is
        # NOT created here because it holds un-picklable ``_thread.RLock``
        # objects (via httpx) and this process is launched with the
        # ``spawn`` start method, which pickles the ``Process`` instance.
        self.active_skills: Dict[str, Skill] = {}
        self.skills_status_history: List[Dict] = []
        self.last_user_msg: Optional[str] = None
        self.client: Any = None

        # Jinja2 env + compiled templates are built lazily inside the
        # child process (see ``run()``) because compiled ``Template``
        # objects hold references to their ``Environment`` which
        # contains un-picklable state, and this process is launched
        # with the ``spawn`` start method.
        self._prompt_env: Optional[jinja2.Environment] = None
        self._system_tmpl: Optional[jinja2.Template] = None
        self._user_tmpl: Optional[jinja2.Template] = None

        # Snapshot of the previous decide() call — proprio + decision —
        # so the next prompt can show the VLM what it just saw and did.
        # Keeping only the last one (no image) to cap token usage.
        self._last_decide: Optional[Dict[str, Any]] = None

        # Stagnation detection — how many consecutive decide() calls
        # returned a decision that starts/stops the exact same skills
        # with the exact same params.  Surfaced to the template via
        # ``{{ stagnation_warning }}`` so prompts can tell the VLM to
        # stop repeating itself.
        self._consecutive_repeat_count: int = 0

        # skill_id → human-readable skill name, derived from the registry
        # so that observation logs show "arm_skill / running" instead of
        # raw integer ids.  Built lazily inside the child process.
        self._skill_id_to_name: Dict[int, str] = {}
        self._name_to_skill_id: Dict[str, int] = {}

        # Monotonic counter assigning a unique number to every skill
        # invocation (one per launch).  The prompt uses these numbers to
        # correlate "started" entries in the previous decide() call with
        # the RUNNING/COMPLETED records in ``recent_skills_status`` —
        # e.g. ``arm_skill[3](delta_pos: [0, 0, -5]): completed``.
        self._skill_invocation_counter: int = 0
        # skill_id → {"number": int, "params": dict} for the most recent
        # invocation of that skill.  Status records published by the
        # skill process are tagged with this invocation's metadata when
        # they are drained from the shared-memory buffer.
        self._current_invocation_by_skill_id: Dict[int, Dict[str, Any]] = {}

        # Capture the parent's log queue so the child process can forward
        # logging records back to the StatusBoard running in the main process.
        self._log_queue = get_log_queue()

        # Set to True once the OpenCV debug window has been created inside
        # this child process.  Guarded with a flag (rather than a
        # try/except around ``namedWindow`` on every call) so a headless
        # backend failure is logged exactly once and subsequent decide()
        # calls skip the window cheaply.
        self._debug_window_ready: bool = False
        self._debug_window_failed: bool = False

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
        skills_status: List[Dict],
    ) -> Dict:
        """
        Decide which skills to start or stop.

        Formats the subclass-supplied ``system_prompt`` / ``user_prompt``
        templates with the latest observation snapshot plus context
        variables (``user_msg``, ``available_skills``, ``active_skills``,
        ``prev_decide``, ``recent_skills_status``), optionally attaches the
        scene image, sends the result to the VLM, and parses its JSON
        response.

        Args:
            image: Latest camera image (H, W, C) to attach to the VLM call.
                Passing ``None`` sends a text-only turn.
            user_msg: Latest user command.
            skills_status: Recent skill status records (see ``_think_step``).

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
        status_text = (
            self._format_skills_status(skills_status[-5:]) if skills_status else "(none)"
        )
        prev_text = self._format_previous_call()

        observation_ns = self._build_observation_namespace()
        prev_obs_ns = self._build_prev_obs_namespace()
        has_prev_obs = bool(
            (self._last_decide or {}).get("obs_snapshot")
        )
        delta_obs_ns = self._build_delta_obs_namespace(observation_ns)
        stagnation_warning = self._format_stagnation_warning()

        fmt_vars: Dict[str, Any] = {
            "user_msg": user_msg,
            "available_skills": available_text,
            "active_skills": active_text,
            "prev_decide": prev_text,
            "recent_skills_status": status_text,
            "prev_obs": prev_obs_ns,
            "has_prev_obs": has_prev_obs,
            "delta_obs": delta_obs_ns,
            "has_delta_obs": has_prev_obs,
            "stagnation_warning": stagnation_warning,
            **observation_ns,
        }

        try:
            system_text = self._system_tmpl.render(**fmt_vars)
            user_text = self._user_tmpl.render(**fmt_vars)
        except (jinja2.TemplateError, LookupError, AttributeError, TypeError) as exc:
            logger.warning(
                "Prompt rendering failed (%s: %s); skipping decide() step. "
                "This is usually transient at startup while observation "
                "buffers are still filling.",
                type(exc).__name__, exc,
            )
            return {
                "skills_to_start": [],
                "skills_to_stop": [],
                "reasoning": "prompt_format_failure",
            }

        # logger.info("decide() system prompt:\n%s", system_text)
        logger.info("decide() user prompt:\n%s", user_text)

        image_data_url: Optional[str] = None
        if image is not None:
            # Image preparation pipeline.  The order matters:
            #   1) Crop fisheye black borders FIRST so the downstream
            #      resize uses the full max_dim budget on actual pixels
            #      (otherwise half the resized frame is wasted on black).
            #   2) Convert RGB → BGR SECOND, before anything that uses
            #      cv2.  The wrist-camera observation is RGB but cv2's
            #      imencode / imshow both assume BGR, so without this
            #      conversion red and blue end up swapped in the JPEG
            #      sent to the VLM and in the debug window alike.
            #   3) Resize into the VLM coordinate frame BEFORE drawing
            #      anything, so the grid / overlay line widths stay
            #      visible after encoding and pixel-percent coords are
            #      consistent with what the VLM reports.
            #   4) Grid comes BEFORE the prev-obs overlay so the overlay
            #      is drawn on top (grid lines never occlude the bbox).
            sent_image = image
            if self.vlm_config.get("image_crop_black_borders", True):
                sent_image = _crop_black_borders(sent_image)
            if self.vlm_config.get("image_is_rgb", True):
                sent_image = _rgb_to_bgr(sent_image)
            sent_image = _resize_for_vlm(
                sent_image, self.vlm_config.get("image_max_dim", 512)
            )
            if self.vlm_config.get("image_draw_coordinate_grid", True):
                sent_image = _draw_coordinate_grid(sent_image)
            # Previous-observation overlay (cyan bbox + L/R finger tips)
            # is only drawn when the VLM's decision schema actually
            # contains those geometric fields.  With the simpler
            # ``next_move`` schema there is nothing meaningful to
            # draw, so the overlay is skipped to keep the frame clean.
            if self.vlm_config.get("image_annotate_prev_obs", True):
                prev_obs_for_overlay = (
                    (self._last_decide or {}).get("decision") or {}
                ).get("observation")
                sent_image = _annotate_image_with_prev_obs(
                    sent_image,
                    prev_obs_for_overlay
                    if isinstance(prev_obs_for_overlay, dict)
                    else None,
                )
            if self.vlm_config.get("debug_show_annotated_image", False):
                self._show_debug_window(sent_image)
            image_data_url = _encode_jpeg_data_url(
                sent_image,
                jpeg_quality=self.vlm_config.get("image_jpeg_quality", 80),
            )

        # Response budget: the full JSON decision (observation + plan +
        # skills + reasoning) is typically 150-250 tokens.  300 leaves
        # a small cushion while keeping total context usage low for
        # small-context models (e.g. 4k-token VLMs), where a big system
        # prompt + a 400-token response budget can overflow the window.
        # Override with ``vlm_config["max_response_tokens"]`` per deployment.
        max_response_tokens = int(
            self.vlm_config.get("max_response_tokens", 300)
        )

        raw = self._chat(
            system_text, user_text, image_data_url,
            max_tokens=max_response_tokens,
        )
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
                system_text, repair, image_data_url,
                max_tokens=max_response_tokens,
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

        # Translate an optional high-level ``next_move`` string into the
        # concrete ``skills_to_start`` / ``skills_to_stop`` the rest of
        # the pipeline expects.  This lets controllers expose a much
        # smaller action vocabulary to the VLM (e.g. LEFT / RIGHT /
        # FORWARD / OPEN / CLOSE / DONE) and keep the base-frame sign
        # math in deterministic Python code instead of in the prompt —
        # the sign-inversion failure mode we hit with weak VLMs when
        # they had to compute ``delta_pos`` directly.  The map lives in
        # ``vlm_config["action_map"]`` so each controller can declare
        # its own vocabulary (keys are case-insensitive strings, values
        # are either a single skill-spec dict, a list of specs, or
        # ``None`` to mean "no skill this call").  When the map is not
        # configured, or the VLM did not emit a ``next_move``, the
        # decision is left untouched.
        action_map = self.vlm_config.get("action_map")
        raw_move = decision.get("next_move")
        if action_map and isinstance(raw_move, str):
            normalized_map = {
                str(k).strip().upper(): v for k, v in action_map.items()
            }
            key = raw_move.strip().upper()
            if key in normalized_map:
                spec = normalized_map[key]
                if spec is None:
                    translated_starts: List[Dict[str, Any]] = []
                elif isinstance(spec, dict):
                    translated_starts = [copy.deepcopy(spec)]
                elif isinstance(spec, list):
                    translated_starts = [
                        copy.deepcopy(s) for s in spec if isinstance(s, dict)
                    ]
                else:
                    translated_starts = []
                decision["skills_to_start"] = translated_starts
                decision.setdefault("skills_to_stop", [])
                decision["next_move"] = key
            else:
                logger.warning(
                    "VLM emitted unknown next_move=%r; allowed=%s. "
                    "Treating as no-op.",
                    raw_move, sorted(normalized_map.keys()),
                )
                decision["skills_to_start"] = []
                decision.setdefault("skills_to_stop", [])

        logger.info(
            "decide() result%s: next_move=%s start=%s stop=%s plan=%s "
            "step=%s scene=%s reasoning=%s",
            " (retry)" if retried else "",
            decision.get("next_move"),
            decision.get("skills_to_start"),
            decision.get("skills_to_stop"),
            decision.get("plan"),
            decision.get("current_step"),
            decision.get("scene") or decision.get("observation"),
            decision.get("reasoning"),
        )

        # Update the stagnation counter BEFORE overwriting ``_last_decide``
        # so we can compare the new decision's signature against the one
        # stashed on the previous call.  A "signature" is just the set of
        # (skill, params) tuples for starts + the set of stopped names —
        # distinct skill invocations with the same effect on the robot
        # count as a repeat.
        new_sig = self._decision_signature(decision)
        prev_sig = (self._last_decide or {}).get("signature")
        empty_sig = ((), ())
        if (
            prev_sig is not None
            and new_sig == prev_sig
            and new_sig != empty_sig
        ):
            self._consecutive_repeat_count += 1
        else:
            self._consecutive_repeat_count = 0

        # Snapshot this call so the next prompt can tell the VLM what
        # skills it just started/stopped, and so ``prev_obs.*`` in the
        # next template render resolves to the proprioception captured
        # *at this decide() call* (not the previous ring-buffer entry,
        # which is only one control-loop tick old).  We intentionally do
        # NOT stash the rendered user prompt: re-embedding it in the
        # next ``prev_decide`` block would recursively carry every
        # earlier prompt forward and grow the context unboundedly.
        self._last_decide = {
            "timestamp": time.time(),
            "decision": decision,
            "obs_snapshot": self._snapshot_observation_latest(observation_ns),
            "signature": new_sig,
        }

        return decision

    @staticmethod
    def _decision_signature(decision: Dict[str, Any]) -> Any:
        """Produce a hashable fingerprint of a decision for stagnation checks.

        We fingerprint the *effect* of the decision — primarily the
        normalized ``next_move`` string (when the controller exposes a
        ``next_move`` vocabulary), and otherwise the set of
        ``(skill_name, params)`` tuples that would be started / stopped.
        ``reasoning`` / ``plan`` / ``scene`` are ignored on purpose: the
        VLM often rephrases the same intent across calls and we still
        want to catch that as a repeat.
        """
        next_move = decision.get("next_move")
        if isinstance(next_move, str) and next_move.strip():
            return ("next_move", next_move.strip().upper())
        starts = tuple(sorted(
            (
                (s.get("skill_name") or ""),
                json.dumps(s.get("params") or {}, sort_keys=True, default=str),
            )
            for s in (decision.get("skills_to_start") or [])
        ))
        stops = tuple(sorted(
            (str(n) for n in (decision.get("skills_to_stop") or []))
        ))
        return (starts, stops)

    def _format_stagnation_warning(self) -> str:
        """Render the ``{{ stagnation_warning }}`` template variable.

        Fires after the same decision has been emitted twice in a row
        (i.e. the current call would be the third identical one), since
        the first repeat is usually a legitimate "keep nudging toward
        the target" case — only sustained repetition signals the VLM is
        stuck and should try a different axis / skill.
        """
        n = self._consecutive_repeat_count
        if n < 2:
            return "(no stagnation detected)"
        return (
            f"WARNING: your last {n + 1} decisions have been IDENTICAL. "
            f"Proprioception deltas above show whether that action is "
            f"actually making progress — if not, pick a DIFFERENT axis, "
            f"a DIFFERENT skill, or DIFFERENT params on the next call "
            f"instead of repeating the same decision."
        )

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

    @staticmethod
    def _snapshot_observation_latest(
        namespace: Dict[str, "_ModuleView"],
    ) -> Dict[str, Dict[str, Any]]:
        """Take a plain-dict snapshot of the latest value of each field.

        Flattens the ``_ModuleView`` / ``_FieldView`` structure to
        ``{module_name: {field_name: latest_value}}`` so it can be
        stashed on ``self._last_decide`` and later rebuilt into a
        fresh ``_ModuleView`` tree for the next render.  Keeping only
        the latest scalar (instead of the whole ring-buffer slice)
        caps memory and makes semantics clear: ``prev_obs.*`` is
        exactly the proprioception seen by the previous decide() call.
        """
        snap: Dict[str, Dict[str, Any]] = {}
        for module_name, module_view in namespace.items():
            fields = getattr(module_view, "_fields", {})
            per_field: Dict[str, Any] = {}
            for field_name, field_view in fields.items():
                try:
                    per_field[field_name] = field_view._resolve(0)
                except Exception:
                    continue
            if per_field:
                snap[module_name] = per_field
        return snap

    def _build_delta_obs_namespace(
        self, current_ns: Dict[str, "_ModuleView"]
    ) -> "_ModuleView":
        """Build ``delta_obs.module.field = now - prev_decide`` views.

        Uses the same snapshot stashed on :attr:`_last_decide` that
        :meth:`_build_prev_obs_namespace` consumes, but computes the
        difference between the latest value in *current_ns* and the
        snapshot value.  Non-numeric fields (e.g. strings, or anything
        that doesn't support subtraction) are silently dropped so
        template rendering can't blow up on them.

        Returns an empty namespace on the first decide() call or when a
        module/field is new this step — templates should still guard
        accesses with ``{% if has_delta_obs %}``.
        """
        prev = self._last_decide or {}
        snap: Dict[str, Dict[str, Any]] = prev.get("obs_snapshot") or {}
        modules: Dict[str, Any] = {}
        for module_name, fields_snap in snap.items():
            current_module = current_ns.get(module_name)
            if current_module is None:
                continue
            current_fields = getattr(current_module, "_fields", {})
            deltas: Dict[str, _FieldView] = {}
            for field_name, prev_val in fields_snap.items():
                cur_field = current_fields.get(field_name)
                if cur_field is None:
                    continue
                try:
                    cur_val = cur_field._resolve(0)
                except Exception:
                    continue
                try:
                    delta = cur_val - prev_val
                except TypeError:
                    continue
                deltas[field_name] = _FieldView(
                    [delta], f"delta_obs.{module_name}.{field_name}"
                )
            if deltas:
                modules[module_name] = _ModuleView(
                    deltas, f"delta_obs.{module_name}"
                )
        return _ModuleView(modules, "delta_obs")

    def _build_prev_obs_namespace(self) -> "_ModuleView":
        """Rebuild a ``_ModuleView`` tree from the last decide()'s snapshot.

        Exposes ``prev_obs.module.field`` in templates, resolving to the
        scalar value captured at the previous decide() call.  On the
        first call the namespace is empty — any template reference
        such as ``prev_obs.arm.x`` will raise a clear ``AttributeError``,
        which ``decide()`` catches and logs as a prompt-render warning.
        """
        prev = self._last_decide or {}
        snap: Dict[str, Dict[str, Any]] = prev.get("obs_snapshot") or {}
        modules: Dict[str, Any] = {}
        for module_name, fields in snap.items():
            module_fields: Dict[str, _FieldView] = {
                field_name: _FieldView([value], f"prev_obs.{module_name}.{field_name}")
                for field_name, value in fields.items()
            }
            modules[module_name] = _ModuleView(
                module_fields, f"prev_obs.{module_name}"
            )
        # ``_ModuleView._fields`` is declared as ``Dict[str, _FieldView]``
        # but its only behavior is dict lookup through ``__getattr__`` /
        # ``__getitem__``, so populating it with ``_ModuleView`` values
        # gives us a ``prev_obs.<module>.<field>`` chain for free.
        return _ModuleView(modules, "prev_obs")

    def _format_previous_call(self) -> str:
        """Render the actions decided in the previous decide() call.

        Returns ``"(none)"`` on the first call.  Only plan / current_step
        / started / stopped from the last decision are included.  We
        deliberately do NOT re-inject the previous ``observation`` block
        or ``reasoning`` string: small VLMs treat them as a copy-paste
        template and keep echoing the same numbers / phrasing across
        calls even when the cyan/L/R overlay clearly shows they were
        wrong.  Feedback on the previous observation is instead
        delivered purely visually, as the cyan target_bbox rectangle
        and the green 'L' / red 'R' finger-tip markers drawn onto the
        current image by ``_annotate_image_with_prev_obs``.  The
        proprioception delta block above already carries the numeric
        progress signal for the last action.
        """
        prev = self._last_decide
        if not prev:
            return "(none)"

        age = max(0.0, time.time() - float(prev.get("timestamp", 0.0)))

        decision = prev.get("decision", {}) or {}
        started = decision.get("skills_to_start") or []
        stopped = decision.get("skills_to_stop") or []
        plan = decision.get("plan") or []
        current_step = decision.get("current_step")

        if started:
            started_parts: List[str] = []
            for spec in started:
                name = spec.get("skill_name")
                params_str = self._format_params(spec.get("params"))
                invocation = spec.get("invocation_number")
                head = name
                if invocation is not None:
                    head = f"{head}[{invocation}]"
                if params_str:
                    head = f"{head}({params_str})"
                started_parts.append(f"    - {head}")
            started_lines = "\n".join(started_parts)
        else:
            started_lines = "    (none)"

        if stopped:
            stopped_lines = "\n".join(f"    - {name}" for name in stopped)
        else:
            stopped_lines = "    (none)"

        if plan:
            plan_lines = "\n".join(
                f"    [{i}] {step}" for i, step in enumerate(plan)
            )
            if current_step is not None:
                plan_block = (
                    f"  plan:\n{plan_lines}\n"
                    f"  current_step: {current_step}\n"
                )
            else:
                plan_block = f"  plan:\n{plan_lines}\n"
        else:
            plan_block = ""

        return (
            f"  {age:.1f}s ago\n"
            f"{plan_block}"
            f"  actions taken then:\n"
            f"    started:\n{started_lines}\n"
            f"    stopped:\n{stopped_lines}"
        )

    # ------------------------------------------------------------------
    # Skill status formatting
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
        self._name_to_skill_id = {name: sid for sid, name in mapping.items()}

    @staticmethod
    def _status_name(status: int) -> str:
        try:
            return SkillStatus(int(status)).name.lower()
        except ValueError:
            return f"status_{int(status)}"

    @staticmethod
    def _format_params(params: Optional[Dict[str, Any]]) -> str:
        """Render a skill's params as a compact ``key: value, ...`` string.

        Returns an empty string when *params* is falsy, so callers can
        decide whether to emit the surrounding ``(...)``.
        """
        if not params:
            return ""
        return ", ".join(f"{k}: {v}" for k, v in params.items())

    def _format_skills_status(self, records: List[Dict]) -> str:
        """Render recent skill status records for the decide() prompt.

        Each record is annotated with the invocation number and params
        of the launch it belongs to, so the VLM can correlate status
        updates with the entries in the previous decide() call, e.g.::

            - arm_skill[1](delta_pos: [0, 0, -5]): running
            - arm_skill[1](delta_pos: [0, 0, -5]): completed
            - gripper_skill[2](delta_pos: 1): running
        """
        lines: List[str] = []
        for rec in records:
            sid = int(rec.get("skill_id", 0))
            name = self._skill_id_to_name.get(sid) or f"skill_{sid}"
            status = self._status_name(rec.get("status", 0))
            invocation = rec.get("invocation")
            params_str = self._format_params(rec.get("params"))

            head = name
            if invocation is not None:
                head = f"{head}[{invocation}]"
            if params_str:
                head = f"{head}({params_str})"
            lines.append(f"- {head}: {status}")
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

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------

    def _show_debug_window(self, image: np.ndarray) -> None:
        """Display *image* in an OpenCV window (same pixels sent to the VLM).

        Enabled by ``vlm_config["debug_show_annotated_image"] = True``.
        The window is created lazily on the first call and reused
        afterwards.  Any GUI backend failure (no display, OpenCV built
        without highgui, etc.) is logged once and the window is disabled
        for the remainder of the process — it's a debugging aid, not a
        correctness requirement.
        """
        if self._debug_window_failed:
            return
        try:
            import cv2
            if not self._debug_window_ready:
                cv2.namedWindow(
                    "think/vlm input", cv2.WINDOW_NORMAL
                )
                self._debug_window_ready = True
            cv2.imshow("think/vlm input", image)
            # ``waitKey`` pumps the GUI event loop; without it the
            # window appears but never repaints.  1 ms is enough to
            # process pending events without noticeable latency at 2 Hz.
            cv2.waitKey(1)
        except Exception as exc:
            logger.warning(
                "Debug window disabled (%s: %s). Unset "
                "vlm_config['debug_show_annotated_image'] to silence.",
                type(exc).__name__, exc,
            )
            self._debug_window_failed = True

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

    def _assign_invocation(
        self, skill_name: str, params: Optional[Dict]
    ) -> Optional[int]:
        """Assign a new invocation number for *skill_name* and remember *params*.

        The number is monotonic across the whole think process, so the
        prompt's ``recent_skills_status`` and ``prev_decide`` blocks can
        line up pairs like ``arm_skill[3](...)`` regardless of which
        skill they belong to.  Returns ``None`` if the skill has no
        resolvable ``skill_id`` (shouldn't happen once the registry is
        built, but keeps status tagging safe).
        """
        sid = self._name_to_skill_id.get(skill_name)
        if sid is None:
            return None
        self._skill_invocation_counter += 1
        number = self._skill_invocation_counter
        self._current_invocation_by_skill_id[int(sid)] = {
            "number": number,
            "params": dict(params or {}),
        }
        return number

    def _launch_skill(
        self, skill_name: str, params: Dict
    ) -> Optional[int]:
        """Start a skill and return the invocation number, or ``None``.

        ``None`` is returned for unknown skills, remote skills that lack
        an ``action_type`` (can't dispatch), and controller skills that
        are already running (launch is a no-op).
        """
        spec = self.available_skills.get(skill_name)
        if spec is None:
            logger.warning(f"Unknown skill: {skill_name}")
            return None

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
                return None
            try:
                action = Action(action_type=int(action_type), data=dict(params or {}))
                self.action_queue.put(action)
                number = self._assign_invocation(skill_name, params)
                logger.info(
                    f"Enqueued action (action_type={int(action_type)}) "
                    f"to launch remote skill '{skill_name}' on {runner.name}"
                )
                return number
            except Exception as exc:
                logger.error(f"Failed to enqueue action for '{skill_name}': {exc}")
                return None

        # Controller-side skill — spawn it as a subprocess and track its
        # lifecycle locally.
        if skill_name in self.active_skills:
            logger.info(f"Skill '{skill_name}' already active, skipping")
            return None

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
        number = self._assign_invocation(skill_name, params)
        logger.info(f"Launched skill '{skill_name}' (PID {skill.pid})")
        return number

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
        # 1. Drain incoming skill-status records (shared memory) and queues.
        # Tag each record with the most recent invocation metadata for
        # its ``skill_id`` so the prompt formatter can show
        # ``arm_skill[3](delta_pos: [0, 0, -5]): completed`` — correlating
        # the status line with the launch that triggered it.
        for rec in self.skill_status_buffer.drain():
            inv = self._current_invocation_by_skill_id.get(int(rec.skill_id))
            self.skills_status_history.append(
                {
                    "skill_id": rec.skill_id,
                    "status": rec.status,
                    "timestamp": rec.timestamp,
                    "invocation": (inv or {}).get("number"),
                    "params": (inv or {}).get("params"),
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
            image, self.last_user_msg, self.skills_status_history
        )

        # 4. Act on the decision.  Launched skills receive an invocation
        # number that is stamped back onto the ``spec`` dict (the same
        # dict stored inside ``self._last_decide``), so the next
        # decide() call's ``prev_decide`` block can show it.
        for spec in decision.get("skills_to_start", []):
            number = self._launch_skill(
                spec["skill_name"], spec.get("params", {})
            )
            if number is not None:
                spec["invocation_number"] = number

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

        # Jinja2 env + pre-compiled prompt templates.  Same reasoning:
        # ``jinja2.Template`` holds references to the environment which
        # isn't safe to pickle across ``spawn``.
        self._prompt_env = _build_prompt_env(self.prompt_filters)
        self._system_tmpl = self._prompt_env.from_string(self.system_prompt)
        self._user_tmpl = self._prompt_env.from_string(self.user_prompt)

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

        if self._debug_window_ready:
            try:
                import cv2
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:
                pass

        logger.info("Think process stopped")
