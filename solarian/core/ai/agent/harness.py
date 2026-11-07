"""The supervising agent: an event-driven ReAct loop over the controller's tools.

Two threads, because the two halves run at different speeds:

* The **planner** thread sleeps until something happens — the user writes,
  an action ends, the monitor escalates, the operator takes over, or the
  optional heartbeat expires. An action that succeeds moves the plan on: the
  harness starts the next step whose call is complete without the planner.
  Otherwise the thread shows the planner model a fresh observation and the
  news, and runs the tool calls it makes until one starts an action. Skill
  tools return as soon as the action has started; the planner never blocks
  on the robot.
* The **monitor** thread ticks every ``monitor.period_s`` while actions run:
  it enforces timeouts and the controller's tripwires, and asks the monitor
  model each action's questions, finishing the action or waking the planner
  when a rule fires.

Everything that commands the robot runs on the controller's control thread
(:meth:`Controller.call_on_control_thread`), and every such command carries
the run's generation: cancelling, resetting or an operator taking over bumps
it, so a command decided before that moment is refused rather than executed
after it.

The host is the controller; the harness reads and calls only the methods
listed in :class:`AgentHost`.
"""

from __future__ import annotations

import copy
import json
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from solarian.core.ai.agent.model import (
    AssistantTurn,
    ImagePart,
    Message,
    ModelClient,
    ToolCall,
    ToolDef,
    ToolResult,
    Usage,
    UserTurn,
    make_model_client,
)
from solarian.core.ai.agent.monitor import (
    ESCALATE,
    FAIL,
    FINISH,
    Monitor,
    MonitorAnswer,
    MonitorEvidence,
    MonitorQuestion,
    RuleTracker,
    make_monitor,
    monitor_context,
)
from solarian.core.ai.agent.run_log import AGENT_DIR, RunLog
from solarian.core.ai.agent.vision import (
    Frame,
    View,
    close_up,
    encode_jpeg,
    make_view,
    mark_point,
    resized,
    target_box,
)
from solarian.core.clock_sync import synced_time
from solarian.core.configs import AgentConfig
from solarian.core.skill import SkillStatus, Tool
from solarian.core.ui.base import UIImage

logger = logging.getLogger(__name__)

#: How long a command may wait for the control thread before it is abandoned.
_CONTROL_TIMEOUT_S = 3.0
#: How long to wait for a skill of the same name to finish stopping before
#: starting the next one, so the old one's end is not taken for the new one's.
_RESTART_WAIT_S = 6.0
#: How long to wait for a seeded tracker to publish its first target.
_SEED_WAIT_S = 3.0
_TRANSCRIPT_LIMIT = 200
_FRAMES_KEPT = 30


class AgentHost(Protocol):
    #: :attr:`Controller.recording_state`: both sides' recording.
    recording_state: dict

    def controller_episode_dir(self) -> Path | None: ...

    def call_on_control_thread(self, fn: Callable[[], Any]) -> Future: ...

    def agent_inputs(self) -> tuple[list[UIImage], dict]: ...

    def start_skill(self, skill) -> str: ...

    def skill_invocation(self, name: str) -> int | None: ...

    def skill_active(self, name: str) -> bool: ...

    def stop_skill(self, name: str, reason: str = "") -> None: ...

    def start_service(self, module: str, field: str, **params) -> None: ...

    def stop_service(self, module: str, field: str) -> None: ...

    def field_value(self, module: str, field: str): ...

    def start_recording(self, task: str | None = None, session: str | None = None) -> None: ...

    def stop_recording(self) -> None: ...

    def on_agent_idle(self) -> None: ...

    def check_agent_action(self, action_id: str, tool: str, elapsed_s: float) -> tuple | None: ...


class RunStatus(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    AWAITING_USER = "awaiting_user"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def active(self) -> bool:
        return self in (RunStatus.THINKING, RunStatus.ACTING, RunStatus.AWAITING_USER, RunStatus.PAUSED)


class StaleCommand(Exception):
    """A command decided before a cancel, reset or takeover; refused, not run."""


@dataclass
class AgentAction:
    id: str
    tool: str
    #: The arguments the planner gave, in its own (model image) pixels.
    args: dict
    skill_name: str
    ends_by_itself: bool
    max_duration_s: float | None
    questions: list[MonitorQuestion]
    status: str = "requested"
    reason: str = ""
    invocation: int | None = None
    started: float = field(default_factory=time.monotonic)
    ended: float | None = None
    tracker: RuleTracker = field(default_factory=RuleTracker)
    reference: list[ImagePart] = field(default_factory=list)
    #: ``(time.monotonic(), frames, their run log file names)`` of this
    #: action's last monitor ticks, oldest first.
    recent: deque[tuple[float, list[ImagePart], list[str]]] = field(default_factory=deque)
    answers: list[MonitorAnswer] = field(default_factory=list)
    escalated: set[str] = field(default_factory=set)
    #: What a stop this harness asked for should be reported as when it lands.
    stop_outcome: str | None = None
    #: The follow-up tool a monitor finish starts once this action stops; the
    #: follow-up's end, not this one's, moves the run on.
    then: str | None = None

    @property
    def running(self) -> bool:
        return self.status in ("requested", "running")

    def elapsed_s(self) -> float:
        return (self.ended or time.monotonic()) - self.started

    def describe(self) -> str:
        text = f"{self.id} {self.tool} {self.status} ({self.elapsed_s():.0f}s)"
        if self.reason:
            text += f": {self.reason}"
        if self.running and self.answers:
            text += " [" + ", ".join(a.summary() for a in self.answers) + "]"
        return text

    def to_ui(self) -> dict:
        return {
            "id": self.id,
            "tool": self.tool,
            "status": self.status,
            "reason": self.reason,
            "elapsed_s": round(self.elapsed_s(), 1),
            "monitor": [a.summary() for a in self.answers],
        }


@dataclass
class Wake:
    kind: str
    text: str = ""


_SYSTEM_PROMPT = """\
You are the supervising agent of a robot. You turn the user's request into a plan and \
carry it out by calling tools that run skills on the real robot.

How the loop works:
- Every time you are woken you get a fresh observation: camera images with an id such \
as o12, and the robot's readings. The images have a faint pixel grid to help you read \
coordinates.
- A skill tool starts an action and returns at once with an action id. The action then \
runs on its own, and starting it ends your turn: you are woken when it fails or is \
stopped, when it succeeds and the plan needs you, when the monitor flags a problem, when \
the user writes, or when the operator intervenes{heartbeat}.
- Give each step of the plan the call that carries it out: `update_plan` steps take \
`tool` and `args`. When an action succeeds, the harness starts the next step whose call \
is complete without waking you. It wakes you for a step that needs a point in a fresh \
image (leave its args out and point when woken), for a step with no tool, such as \
checking the result, and when every step is done. Update the plan in the same reply as \
the action you start.
- Skills marked "runs until stopped" are watched by a fast monitor that finishes them \
when their success condition holds. You can stop any action with `stop_action`.
- When starting an action you may add `watch_for`: short visual statements of how it \
would look if it went wrong. The monitor wakes you if one becomes true, checking from \
6 s into the action. Each must be false the whole time the action goes as it should: \
"the object is on the table" is true during every approach, so it would wake you for \
nothing; "the fingers are closed with nothing between them" is not. The monitor may \
see only a close-up of the gripper, so describe what shows there. Skills that \
list their own monitoring already watch for failure, and rarely need one.
- The monitor is fast but often wrong. Treat its flags as hints: check them against \
the newest observation, and stop an action only when the image agrees. If the image \
shows the action progressing, reply without a tool call and it keeps running.

How to work:
- Look before you act. Point at things with [x, y] pixel coordinates in the image of \
the observation you quote as `observation_id`. A point is only valid for that image: \
after anything moves, use the newest observation.
- Keep a short plan with `update_plan`. The harness marks steps in progress and done \
as their actions run; mark failed or skipped steps yourself.
- Verify every step in the images. An action that ended "succeeded" may still have \
missed; one that failed or timed out may have worked or half-worked, so before \
undoing it (opening a gripper, moving away) check the newest image and build on what \
it shows. Recover: retry with a corrected \
point, nudge the arm or move it to a viewing pose to see better, or try another way. \
Do not repeat the same failing call more than a few times; ask the user instead.
- Starting an action on an arm preempts whatever else is running on it.
- When the task is done and you have checked it visually, call `finish_task`. If you \
need information or the task is impossible, `ask_user`. If anything looks unsafe, stop \
the actions and ask the user.
- Your text is shown in a small chat panel: keep it to a sentence or two.
"""

_HARNESS_TOOLS = [
    ToolDef(
        "observe",
        "Take a fresh observation (camera images and robot readings) right now.",
        {"type": "object", "properties": {}},
    ),
    ToolDef(
        "stop_action",
        "Stop a running action.",
        {
            "type": "object",
            "properties": {"action_id": {"type": "string"}},
            "required": ["action_id"],
        },
    ),
    ToolDef(
        "update_plan",
        "Replace the plan shown to the user and run by the harness. Keep it short; one step "
        "per skill call. When an action succeeds, the harness starts the next step whose "
        "`tool` and `args` are complete. Leave `args` out of a step that needs a point in a "
        "fresh image, and `tool` out of one that is yours, such as checking the result.",
        {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "done", "failed", "skipped"],
                            },
                            "tool": {"type": "string", "description": "The skill tool that carries the step out."},
                            "args": {"type": "object", "description": "Its arguments, when no later image is needed."},
                        },
                        "required": ["step", "status"],
                    },
                }
            },
            "required": ["steps"],
        },
    ),
    ToolDef(
        "ask_user",
        "Ask the user a question and wait for the answer. Running actions keep running.",
        {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    ),
    ToolDef(
        "finish_task",
        "End the task: success after you verified it in the images, or failure with what went wrong.",
        {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "summary": {"type": "string"},
            },
            "required": ["success", "summary"],
        },
    ),
]

_DEFERRED_TOOLS = ("ask_user",)


def _skill_tool_def(name: str, spec: Mapping[str, Any]) -> ToolDef:
    """The planner-facing definition of a skill tool, from its contract."""
    lines = [str(spec["description"]).strip()]
    if spec.get("ends_by_itself", True):
        lines.append("Ends by itself.")
    elif spec.get("monitor"):
        lines.append("Runs until stopped: the monitor finishes it on success, or stop it yourself.")
    elif spec.get("max_duration_s"):
        lines.append("Runs until its time limit, or until you stop it; then judge from the newest image "
                     "whether it succeeded.")
    else:
        lines.append("Runs until you stop it.")
    if spec.get("max_duration_s"):
        lines.append(f"Times out after {spec['max_duration_s']:.0f} s.")
    for example in spec.get("examples") or []:
        when = f" — {example['when']}" if example.get("when") else ""
        lines.append(f"Example: {json.dumps(example.get('params', {}))}{when}")
    parameters = copy.deepcopy(dict(spec.get("parameters") or {"type": "object", "properties": {}}))
    parameters.setdefault("type", "object")
    properties = parameters.setdefault("properties", {})
    required = list(parameters.get("required", []))
    if spec.get("image_points"):
        if spec.get("camera"):
            lines.append(f"Points are [x, y] pixels in the {spec['camera']} image of an observation.")
        properties["observation_id"] = {
            "type": "string",
            "description": "Id of the observation whose image the point(s) were read from, e.g. o12.",
        }
        required.append("observation_id")
    properties["watch_for"] = {
        "type": "array",
        "items": {"type": "string"},
        "description": (
            "Optional visual failure conditions the monitor should wake you for. Each must be "
            "false while the action goes as it should; checked from 6 s into the action."
        ),
    }
    parameters["required"] = required
    return ToolDef(name, "\n".join(lines), parameters)


def _missing_required(schema: Mapping[str, Any], args: Mapping[str, Any]) -> list[str]:
    return [key for key in schema.get("required", []) if key not in args]


def _armed_sides_recording(state: Mapping[str, Any]) -> bool:
    """Every side the switches armed has opened its episode.

    The controller's episode follows the robot's when both record, so the
    robot going active is not yet a directory the run can log into. A robot
    that cannot record is not waited for.
    """
    targets = state.get("targets") or {}
    armed = [
        side for side in ("robot", "controller")
        if targets.get(side) and (side != "robot" or (state.get("robot") or {}).get("enabled"))
    ]
    return bool(armed) and all((state.get(side) or {}).get("active") for side in armed)


class AgentHarness:
    def __init__(
        self,
        host: AgentHost,
        cfg: AgentConfig,
        tools: Mapping[str, Tool],
        planner: ModelClient | None = None,
        monitor: Monitor | None = None,
    ):
        self.host = host
        self.cfg = cfg
        self.tools: dict[str, Tool] = {}
        self.specs: dict[str, dict] = {}
        for name, tool in tools.items():
            spec = tool.spec()
            if not spec.get("description"):
                logger.warning("Tool '%s' (%s) has no description; the agent will not see it",
                               name, tool.skill.__name__)
                continue
            self.tools[name] = tool
            self.specs[name] = spec
        for name, spec in self.specs.items():
            for question in spec.get("monitor") or []:
                for rule in MonitorQuestion.from_spec(question).rules:
                    if rule.then and rule.then not in self.specs:
                        raise ValueError(f"{name}: monitor question {question['name']!r} "
                                         f"has then: {rule.then!r}, which is not a tool")
        self.tool_defs = [_skill_tool_def(n, s) for n, s in self.specs.items()] + _HARNESS_TOOLS
        heartbeat = cfg.planner_heartbeat_s
        self.system_prompt = _SYSTEM_PROMPT.format(
            heartbeat=f", or every {heartbeat:.0f} s while actions run" if heartbeat else "")
        if cfg.system_prompt.strip():
            self.system_prompt += "\nAbout this robot and its tools:\n" + cfg.system_prompt.strip() + "\n"
        self.planner = planner or make_model_client(cfg.planner)
        self.monitor = monitor if monitor is not None else (
            make_monitor(cfg.monitor) if cfg.monitor else None
        )
        self._monitor_side = cfg.monitor.image_max_side if cfg.monitor else 512
        self._monitor_period = cfg.monitor.period_s if cfg.monitor else 1.0
        self._monitor_history = max(1, cfg.monitor.history) if cfg.monitor else 1
        self._close_ups = cfg.monitor.close_ups if cfg.monitor else {}
        self._monitor_reference = cfg.monitor.reference if cfg.monitor else False
        self._mark_target = cfg.monitor.mark_target if cfg.monitor else False

        self._inbox: queue.Queue[Wake] = queue.Queue()
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._threads: list[threading.Thread] = []
        #: Bumped by cancel, reset and operator takeover. See the module docstring.
        self.generation = 0

        self.status = RunStatus.IDLE
        self.messages: list[Message] = []
        self.plan: list[dict] = []
        self.transcript: deque[dict] = deque(maxlen=_TRANSCRIPT_LIMIT)
        self.actions: list[AgentAction] = []
        self.task = ""
        #: ``<session dir>/<episode dir>`` of a recorded run; ``None`` when the
        #: controller is not recording it, and nothing of the run is kept.
        self.run_id: str | None = None
        self.run_log: RunLog | None = None
        #: Tokens and calls of the current run, per model.
        self.usage = Usage()
        self.monitor_usage = Usage()
        self.monitor_calls = 0
        #: Frames captured for the monitor this run: names the logged ``m<k>`` images.
        self._monitor_frames = 0
        self._turns = 0
        self._run_started = 0.0
        self._action_seq = 0
        self._obs_seq = 0
        self._frames: deque[Frame] = deque(maxlen=_FRAMES_KEPT)
        self._failures: dict[str, int] = {}
        #: Tool results of the last planner turn not yet sent back to it.
        self._pending_results: list[ToolResult] = []
        #: ``ask_user`` calls answered at the next wake.
        self._deferred: list[str] = []
        self._notes: list[str] = []
        self._logged_upto = 0
        self._opened_episode = False
        self._view: UIImage | None = None

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        for target, name in ((self._planner_loop, "agent-planner"), (self._monitor_loop, "agent-monitor")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def close(self) -> None:
        self._closed.set()
        self._inbox.put(Wake("close"))

    # ---- inputs, from any thread ---------------------------------------------

    def send_user_message(self, text: str) -> None:
        text = text.strip()
        if text:
            self._inbox.put(Wake("user", text))

    def cancel(self) -> None:
        self._inbox.put(Wake("cancel"))

    def reset(self) -> None:
        self._inbox.put(Wake("reset"))

    def on_skill_ended(self, invocation: int, status: SkillStatus, reason: str) -> None:
        """A skill run the controller started has ended. Only this harness's own runs matter."""
        with self._lock:
            action = next((a for a in self.actions if a.invocation == invocation and a.running), None)
            if action is None:
                return
            if status == SkillStatus.COMPLETED:
                outcome = "succeeded"
            elif status == SkillStatus.FAILED:
                outcome = "failed"
            elif action.stop_outcome is not None:
                outcome = action.stop_outcome
            elif reason.startswith("preempted"):
                outcome = "preempted"
            else:
                outcome = "stopped"
            action.status = outcome
            if status == SkillStatus.FAILED and action.reason and reason and reason != action.reason:
                # The skill failed on its own while the stop was on its way: its reason is the cause.
                action.reason = f"{action.reason}; the skill reported: {reason}"
            else:
                action.reason = action.reason or reason
            action.ended = time.monotonic()
            if outcome in ("failed", "timeout"):
                self._failures[action.tool] = self._failures.get(action.tool, 0) + 1
            elif outcome == "succeeded":
                self._failures[action.tool] = 0
            step = next((s for s in self.plan if s.get("action") == action.id), None)
            if outcome == "succeeded" and step is not None and action.then is None:
                step["status"] = "done"
        self._log("action_ended", action=action.id, status=outcome, reason=action.reason)
        self._say("tool", f"{action.id} {action.tool}: {outcome}" + (f" — {action.reason}" if action.reason else ""))
        text = f"Action {action.describe()}."
        if action.then is not None:
            # The rule that stopped it already woke the planner, or handed on to its follow-up.
            with self._lock:
                self._notes.append(text)
        elif outcome == "succeeded" and step is not None:
            self._inbox.put(Wake("step", text))
        else:
            self._inbox.put(Wake("action", text))

    def on_operator_action(self, description: str) -> None:
        """The operator commanded the robot directly; the run yields to them."""
        with self._lock:
            if not self.status.active or self.status == RunStatus.PAUSED:
                return
            self.generation += 1
            self.status = RunStatus.PAUSED
        self._log("operator", description=description)
        self._say("system", f"Paused: the operator {description}. Send a message to continue.")
        self._inbox.put(Wake("operator", f"The operator took control: {description}."))

    # ---- outputs, for the UI -------------------------------------------------

    def ui_state(self) -> dict:
        with self._lock:
            return {
                "enabled": True,
                "status": self.status.value,
                "task": self.task,
                "run_id": self.run_id,
                "planner": self.planner.model,
                "monitor": self.monitor.model if self.monitor else None,
                "dry_run": self.cfg.dry_run,
                "plan": [dict(step) for step in self.plan],
                "actions": [a.to_ui() for a in self.actions[-8:]],
                "transcript": list(self.transcript),
                "usage": self._usage(),
            }

    def _usage(self) -> dict:
        """Per model: which one, how many calls this run, and the tokens they took."""
        usage = {"planner": {"model": self.planner.model, "calls": self._turns, **self.usage.to_dict()}}
        if self.monitor is not None:
            usage["monitor"] = {"model": self.monitor.model, "calls": self.monitor_calls,
                                **self.monitor_usage.to_dict()}
        return usage

    def view_image(self) -> UIImage | None:
        """The image the planner last saw, with the points it chose drawn on."""
        return self._view

    def _show(self, rgb: np.ndarray) -> None:
        self._view = UIImage(label="agent/view", data=rgb, jpeg=encode_jpeg(rgb))

    # ---- planner thread --------------------------------------------------------

    def _planner_loop(self) -> None:
        while not self._closed.is_set():
            try:
                first = self._inbox.get(timeout=self._wake_timeout())
            except queue.Empty:
                first = self._timer_wake()
            wakes = [first]
            while True:
                try:
                    wakes.append(self._inbox.get_nowait())
                except queue.Empty:
                    break
            try:
                self._handle(wakes)
            except Exception as exc:
                logger.exception("Agent turn failed")
                self._say("system", f"Agent error: {exc}")
                with self._lock:
                    if self.status.active:
                        self.status = RunStatus.AWAITING_USER

    def _wake_timeout(self) -> float | None:
        with self._lock:
            if self.status != RunStatus.ACTING:
                return None
        return self.cfg.planner_heartbeat_s

    def _timer_wake(self) -> Wake:
        return Wake("heartbeat", f"Heartbeat: {self.cfg.planner_heartbeat_s:.0f} s with actions running.")

    def _handle(self, wakes: list[Wake]) -> None:
        if any(w.kind == "close" for w in wakes):
            return
        if any(w.kind == "reset" for w in wakes):
            self._end_run(RunStatus.CANCELLED, "Reset by the user.")
            with self._lock:
                self.messages, self.plan, self.actions = [], [], []
                self.transcript.clear()
                self._pending_results, self._deferred, self._notes = [], [], []
                self._frames.clear()
                self._view = None
                self.status = RunStatus.IDLE
            return
        if any(w.kind == "cancel" for w in wakes):
            self._end_run(RunStatus.CANCELLED, "Cancelled by the user.")
            return

        user_texts = [w.text for w in wakes if w.kind == "user"]
        for text in user_texts:
            self._say("user", text)
        notes = [w.text for w in wakes if w.kind != "user"]
        if self.status.active:
            notes += [f"The user says: {t}" for t in user_texts]
        elif user_texts:
            self._begin_run(" ".join(user_texts))
        else:
            return
        with self._lock:
            self._notes.extend(notes)
            status = self.status
        if status == RunStatus.PAUSED and not user_texts:
            return
        if status == RunStatus.AWAITING_USER and not user_texts:
            return
        if all(w.kind == "step" for w in wakes):
            note = self._advance_plan()
            if note is None:
                return
            if note:
                with self._lock:
                    self._notes.append(note)
        self._think()

    def _advance_plan(self) -> str | None:
        """Start the plan's next step when its call needs nothing from the planner.

        None when the harness has the run in hand: an action still runs, or the
        step started. Otherwise what the planner is needed for ("" for nothing
        in particular).
        """
        with self._lock:
            if any(a.running for a in self.actions):
                return None
            gen = self.generation
            index, step = next(((i, s) for i, s in enumerate(self.plan)
                                if s["status"] in ("pending", "in_progress")), (-1, None))
        if step is None:
            return "Every step of the plan is done; check the result." if self.plan else ""
        name = f"step {index + 1} ({step['step']})"
        tool = step.get("tool")
        if tool is None:
            return f"Next is {name}, which has no tool: it is yours."
        if self.specs[tool].get("image_points"):
            return f"Next is {name}: point its {tool} in the newest observation."
        if self.cfg.dry_run:
            return f"Next is {name}; in a dry run the harness starts nothing."
        try:
            started = json.loads(self._start_action(ToolCall(f"step-{index + 1}", tool, dict(step.get("args") or {})), gen))
        except (ToolError, StaleCommand) as exc:
            return f"Next is {name}, but its {tool} call could not start: {exc}"
        self._attach(started["action_id"], tool, step)
        self._log("plan_step", step=index + 1, action=started["action_id"], tool=tool)
        with self._lock:
            if gen == self.generation:
                self.status = RunStatus.ACTING
        return None

    def _attach(self, action_id: str, tool: str, step: dict | None = None) -> None:
        """Link an action to the step it carries out and mark that step in progress.
        By default: the first unfinished step naming its tool and not taken by
        another running action."""
        with self._lock:
            if step is None:
                running = {a.id for a in self.actions if a.running and a.id != action_id}
                step = next((s for s in self.plan if s.get("tool") == tool
                             and s["status"] in ("pending", "in_progress")
                             and s.get("action") not in running), None)
            if step is not None:
                step.update(status="in_progress", action=action_id)

    # ---- runs --------------------------------------------------------------------

    def _begin_run(self, task: str) -> None:
        session, episode = self._open_episode(task)
        episode_dir = self.host.controller_episode_dir() if session is not None else None
        # One run is one episode, so the episode's place in its session is the id.
        run_id = f"{episode_dir.parent.parent.name}/{episode_dir.name}" if episode_dir else None
        with self._lock:
            self.generation += 1
            self.task = task
            self.run_id = run_id
            self.status = RunStatus.THINKING
            self.actions = []
            self.plan = []
            self._failures = {}
            self._turns = 0
            self.usage = Usage()
            self.monitor_usage = Usage()
            self.monitor_calls = 0
            self._monitor_frames = 0
            self._run_started = time.monotonic()
            self._logged_upto = len(self.messages)
        # Only the controller records the agent's side. A run the robot alone
        # records keeps nothing of the agent; one nobody records keeps nothing.
        self.run_log = None if episode_dir is None else RunLog(
            episode_dir / AGENT_DIR,
            {
                "run_id": run_id,
                "task": task,
                "started": synced_time(),
                "session": session,
                "episode": episode,
                "planner": self.cfg.planner.model_dump(exclude={"api_key_env"}),
                "monitor": self.cfg.monitor.model_dump(exclude={"api_key_env"}) if self.cfg.monitor else None,
                "system_prompt": self.system_prompt,
                "tools": [t.__dict__ for t in self.tool_defs],
                "status": RunStatus.THINKING.value,
            },
        )
        self._log("run_started", task=task)
        if run_id is not None:
            self._say("system", f"Run {run_id} started")
        elif episode is not None:
            self._say("system", f"Run started on {session} episode {episode}; the agent's side is not recorded")
        else:
            self._say("system", "Run started; nothing is recorded")
        first = not self.messages
        self._user_turn().content.append(f"{'Task' if first else 'New task'} from the user: {task}")

    def _open_episode(self, task: str) -> tuple[str | None, int | None]:
        """One run is one episode: open one if anything records and none is open."""
        state = dict(self.host.recording_state)
        if not (self.cfg.record_runs and state.get("enabled")):
            return None, None
        self._opened_episode = False
        if not state.get("active"):
            self._control(lambda: self.host.start_recording(task=task), check=False)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not _armed_sides_recording(self.host.recording_state):
                time.sleep(0.05)
            state = dict(self.host.recording_state)
            if not state.get("active"):
                self._say("system", "No episode opened; this run is not recorded.")
                return None, None
            self._opened_episode = True
        return state.get("session"), int(state.get("episode", 0))

    def _end_run(self, status: RunStatus, summary: str) -> None:
        with self._lock:
            if not self.status.active:
                return
            self.generation += 1
            self.status = status
            running = [a for a in self.actions if a.running]
        for action in running:
            self._stop_action(action, "stopped", summary)
        if self._opened_episode:
            self._control(self.host.stop_recording, check=False)
            self._opened_episode = False
        self._control(self.host.on_agent_idle, check=False)
        self._say("system", summary)
        self._log("run_ended", status=status.value, summary=summary)
        if self.run_log is not None:
            self.run_log.update(status=status.value, summary=summary, ended=synced_time(), usage=self._usage())

    def _budget_exceeded(self) -> str | None:
        budgets = self.cfg.budgets
        if self._turns >= budgets.max_turns:
            return f"Stopped: the run used its {budgets.max_turns} planner turns."
        if time.monotonic() - self._run_started > budgets.max_minutes * 60.0:
            return f"Stopped: the run took longer than {budgets.max_minutes:.0f} minutes."
        return None

    # ---- thinking -------------------------------------------------------------------

    def _user_turn(self) -> UserTurn:
        """The user turn the next planner call will read: the last message if it
        is one, so user turns never follow each other."""
        if not self.messages or not isinstance(self.messages[-1], UserTurn):
            self.messages.append(UserTurn())
        return self.messages[-1]

    def _think(self) -> None:
        with self._lock:
            gen = self.generation
            self.status = RunStatus.THINKING
            notes, self._notes = self._notes, []
            owed = list(self._pending_results)
            owed += [ToolResult(cid, ["Woken up; see below."]) for cid in self._deferred]
            self._pending_results, self._deferred = [], []
        frame = self._observe()
        turn = self._user_turn()
        turn.tool_results = owed + turn.tool_results
        turn.content += [self._status_text(notes, frame), *frame.image_parts()]

        while True:
            exceeded = self._budget_exceeded()
            if exceeded:
                self._end_run(RunStatus.FAILED, exceeded)
                return
            reply = self._call_planner()
            if reply is None:
                with self._lock:
                    if self.status == RunStatus.THINKING:
                        self.status = RunStatus.AWAITING_USER
                return
            if reply.text:
                self._say("agent", reply.text)
            if gen != self.generation:
                self._pending_results = [
                    ToolResult(c.id, ["Not run: the run was interrupted before this call."], is_error=True)
                    for c in reply.tool_calls
                ]
                return
            if not reply.tool_calls:
                with self._lock:
                    self.status = RunStatus.ACTING if any(a.running for a in self.actions) else RunStatus.AWAITING_USER
                return

            results: list[ToolResult] = []
            deferred: list[str] = []
            finished = False
            for call in reply.tool_calls:
                if finished:
                    results.append(ToolResult(call.id, ["Not run: the task was finished."], is_error=True))
                    continue
                if call.name in _DEFERRED_TOOLS:
                    error = self._defer(call)
                    if error:
                        results.append(ToolResult(call.id, [error], is_error=True))
                    else:
                        deferred.append(call.id)
                    continue
                if call.name == "finish_task":
                    finished = True
                result = self._execute(call, gen)
                results.append(result)
            self._log_turn()

            if finished:
                self.messages.append(UserTurn(tool_results=results))
                success = bool(next(c for c in reply.tool_calls if c.name == "finish_task").arguments.get("success"))
                summary = str(next(c for c in reply.tool_calls if c.name == "finish_task").arguments.get("summary") or "")
                self._end_run(RunStatus.DONE if success else RunStatus.FAILED,
                              ("Done: " if success else "Failed: ") + summary)
                return
            names = {c.name for c in reply.tool_calls}
            note, acting = "", False
            if not ("observe" in names or any(r.is_error for r in results)):
                with self._lock:
                    acting = any(a.running for a in self.actions)
                if not (deferred or acting) and "update_plan" in names:
                    note = self._advance_plan()
                    acting = note is None
            if deferred or acting:
                # An action now runs on its own; its end, the monitor or the user wakes the planner.
                with self._lock:
                    self._pending_results = results
                    self._deferred = deferred
                    if gen == self.generation:
                        self.status = RunStatus.AWAITING_USER if deferred else RunStatus.ACTING
                return
            self.messages.append(UserTurn(tool_results=results, content=[note] if note else []))
            if gen != self.generation:
                return

    def _defer(self, call: ToolCall) -> str | None:
        question = str(call.arguments.get("question") or "").strip()
        if not question:
            return "ask_user needs a question."
        self._say("agent", question)
        self._log("ask_user", question=question)
        return None

    def _call_planner(self) -> AssistantTurn | None:
        self._trim_images()
        t0 = time.monotonic()
        try:
            reply = self.planner.complete(self.system_prompt, self.messages, self.tool_defs)
        except Exception as exc:
            logger.exception("Planner call failed")
            self._say("system", f"Planner call failed: {exc}. Send a message to retry.")
            self._log("planner_error", error=str(exc))
            return None
        latency = time.monotonic() - t0
        with self._lock:
            self._turns += 1
            self.usage.add(reply.usage)
        added = self.messages[self._logged_upto:]
        self.messages.append(reply)
        if self.run_log is not None:
            self.run_log.turn(added, reply, latency)
        self._logged_upto = len(self.messages)
        for call in reply.tool_calls:
            self._log("tool_call", id=call.id, name=call.name, arguments=call.arguments)
        return reply

    def _log_turn(self) -> None:
        if self.run_log is not None:
            self.run_log.update(status=self.status.value, plan=self.plan, usage=self._usage())

    def _trim_images(self) -> None:
        """Keep the newest images in context; older ones become a one-line stub.

        Trimmed in a batch once twice the limit is reached, rather than one by
        one, so the cached prefix survives most calls.
        """
        limit = max(1, self.cfg.max_images_in_context)
        slots: list[tuple[list, int]] = []
        for message in self.messages:
            if not isinstance(message, UserTurn):
                continue
            for parts in [message.content] + [r.content for r in message.tool_results]:
                slots += [(parts, i) for i, p in enumerate(parts) if isinstance(p, ImagePart)]
        if len(slots) <= 2 * limit:
            return
        for parts, i in slots[:-limit]:
            parts[i] = f"[image {parts[i].label} no longer shown]"

    def _status_text(self, notes: list[str], frame: Frame) -> str:
        lines = []
        if notes:
            lines.append("What happened: " + " ".join(notes))
        with self._lock:
            elapsed = time.monotonic() - self._run_started
            lines.append(f"Run time {elapsed:.0f} s, planner turn {self._turns + 1} of {self.cfg.budgets.max_turns}.")
            recent = self.actions[-6:]
            if recent:
                lines.append("Actions: " + "; ".join(a.describe() for a in recent) + ".")
            else:
                lines.append("No actions yet.")
            if self.plan:
                lines.append("Plan: " + "; ".join(
                    f"{i}. {s['step']} [{s['status']}" + (f", {s['action']}" if s.get("action") else "") + "]"
                    for i, s in enumerate(self.plan, 1)) + ".")
        lines.append(f"Observation {frame.obs_id}, robot readings: " + json.dumps(frame.state, separators=(",", ":")))
        return "\n".join(lines)

    # ---- observations ------------------------------------------------------------------

    def _capture(self) -> tuple[list[UIImage], dict]:
        images, state = self.host.agent_inputs()
        if self.cfg.cameras:
            images = [i for i in images if i.label in self.cfg.cameras]
        return images, state

    def _observe(self) -> Frame:
        images, state = self._capture()
        views = [make_view(i, self.cfg.image_max_side, self.cfg.image_grid) for i in images]
        with self._lock:
            self._obs_seq += 1
            frame = Frame(obs_id=f"o{self._obs_seq}", t_mono=time.monotonic(), t=synced_time(),
                          views=views, state=state)
            self._frames.append(frame)
        if views:
            self._show(views[0].rgb)
        if self.run_log is not None:
            for part in frame.image_parts():
                self.run_log.image(part)
            self._log("observation", obs_id=frame.obs_id, images=[p.label for p in frame.image_parts()])
        return frame

    def _frame(self, obs_id: str | None) -> Frame | None:
        with self._lock:
            if obs_id is None:
                return self._frames[-1] if self._frames else None
            return next((f for f in self._frames if f.obs_id == obs_id), None)

    # ---- tools ---------------------------------------------------------------------------

    def _execute(self, call: ToolCall, gen: int) -> ToolResult:
        if call.parse_error:
            return ToolResult(call.id, [call.parse_error], is_error=True)
        try:
            if call.name in self.tools:
                text = self._start_action(call, gen)
                self._attach(json.loads(text)["action_id"], call.name)
                return ToolResult(call.id, [text])
            if call.name == "observe":
                frame = self._observe()
                return ToolResult(call.id, [self._status_text([], frame), *frame.image_parts()])
            if call.name == "stop_action":
                return ToolResult(call.id, [self._stop_by_id(str(call.arguments.get("action_id") or ""))])
            if call.name == "update_plan":
                steps = [self._plan_step(s) for s in call.arguments.get("steps") or [] if isinstance(s, dict)]
                with self._lock:
                    # A rewritten plan keeps its running steps' links to their actions.
                    linked = {(s["step"], s.get("tool")): s["action"] for s in self.plan if s.get("action")}
                    running = {a.id: a.tool for a in self.actions if a.running}
                    for step in steps:
                        action_id = linked.get((step["step"], step.get("tool")))
                        if step["status"] == "in_progress" and action_id in running:
                            step["action"] = action_id
                    self.plan = steps
                    unlinked = set(running) - {s.get("action") for s in steps}
                for action_id in sorted(unlinked):
                    self._attach(action_id, running[action_id])
                self._log("plan", steps=steps)
                return ToolResult(call.id, ["Plan updated."])
            if call.name == "finish_task":
                return ToolResult(call.id, ["Task finished."])
            return ToolResult(call.id, [f"Unknown tool {call.name!r}."], is_error=True)
        except StaleCommand:
            return ToolResult(call.id, ["Not run: the run was interrupted."], is_error=True)
        except ToolError as exc:
            self._log("tool_error", id=call.id, name=call.name, error=str(exc))
            return ToolResult(call.id, [str(exc)], is_error=True)

    def _plan_step(self, raw: dict) -> dict:
        step = {"step": str(raw.get("step", "")), "status": str(raw.get("status", "pending"))}
        tool = raw.get("tool")
        if tool:
            if tool not in self.tools:
                raise ToolError(f"Step {step['step']!r}: {tool!r} is not a skill tool.")
            step["tool"] = str(tool)
            if isinstance(raw.get("args"), dict):
                step["args"] = dict(raw["args"])
        return step

    def _start_action(self, call: ToolCall, gen: int) -> str:
        name = call.name
        tool, spec = self.tools[name], self.specs[name]
        args = dict(call.arguments)
        watch_for = [str(w) for w in args.pop("watch_for", None) or [] if str(w).strip()]
        obs_id = args.pop("observation_id", None)
        missing = _missing_required(spec.get("parameters") or {}, args)
        if missing:
            raise ToolError(f"Missing required arguments: {', '.join(missing)}.")
        if self._failures.get(name, 0) >= self.cfg.budgets.retries_per_step:
            raise ToolError(
                f"{name} failed {self._failures[name]} times in a row. Change approach "
                "(reset the scene, a different point or pose) or ask_user."
            )

        skill_params = {**tool.params, **args}
        marked = None
        pointed: View | None = None
        if spec.get("image_points"):
            frame = self._frame(str(obs_id) if obs_id else None)
            if frame is None:
                raise ToolError(f"Unknown observation_id {obs_id!r}; call observe and use its id.")
            if frame.age_s() > self.cfg.max_point_age_s:
                raise ToolError(f"Observation {frame.obs_id} is {frame.age_s():.0f} s old; call observe and point again.")
            with self._lock:
                moved = any(a.started > frame.t_mono or (a.ended or 0) > frame.t_mono for a in self.actions)
            if moved:
                raise ToolError(f"The robot has moved since {frame.obs_id}; point on the newest observation.")
            camera = spec.get("camera")
            view = frame.view(str(camera) if camera else None)
            if view is None:
                raise ToolError(f"{frame.obs_id} has no {camera or 'camera'} image to point in.")
            marked = view.rgb
            for key in spec["image_points"]:
                point = args.get(key)
                if not (isinstance(point, (list, tuple)) and len(point) == 2
                        and all(isinstance(v, (int, float)) for v in point)):
                    raise ToolError(f"{key} must be [x, y] pixels in {frame.obs_id}'s image.")
                if not view.contains(point):
                    raise ToolError(f"{key} {list(point)} is outside the {view.size[0]}x{view.size[1]} image.")
                skill_params[key] = view.to_capture(point)
                marked = mark_point(marked, point, f"{key}")
            pointed = replace(view, rgb=marked)

        with self._lock:
            self._action_seq += 1
            action = AgentAction(
                id=f"a{self._action_seq}",
                tool=name,
                args=dict(call.arguments),
                skill_name=tool.skill.registry_name(),
                ends_by_itself=bool(spec.get("ends_by_itself", True)),
                max_duration_s=spec.get("max_duration_s"),
                questions=[MonitorQuestion.from_spec(q) for q in spec.get("monitor") or []]
                + [MonitorQuestion.watch_for(i, w) for i, w in enumerate(watch_for)],
                recent=deque(maxlen=self._monitor_history),
            )
        if marked is not None:
            self._show(marked)
        self._say("tool", f"{action.id} {name}({json.dumps(call.arguments)})")
        self._log("action_requested", action=action.id, tool=name, args=call.arguments, skill_params=skill_params)

        if self.cfg.dry_run:
            action.status, action.reason, action.ended = "dry_run", "dry run: not started", time.monotonic()
            with self._lock:
                self.actions.append(action)
            return json.dumps({"action_id": action.id, "status": "dry_run",
                               "note": "Dry run: nothing moved. Do not wait for it."})

        requested = time.monotonic()
        if self.host.skill_active(action.skill_name):
            self._control(lambda: self.host.stop_skill(action.skill_name, reason=f"replaced by {action.id}"), gen=gen)
            deadline = time.monotonic() + _RESTART_WAIT_S
            while self.host.skill_active(action.skill_name) and time.monotonic() < deadline:
                time.sleep(0.05)

        services = tool.skill.services_for(skill_params)
        target_area = None
        if services:
            seeded_at = synced_time()
            self._control(lambda: [self.host.start_service(m, f, **p) for m, f, p in services], gen=gen)
            values = [self._wait_for_field(module, field_name, seeded_at) for module, field_name, _ in services]
            limit = spec.get("max_target_area")
            if limit is not None:
                masks = [np.asarray(v) for v in values if v is not None and np.ndim(v) == 2]
                target_area = max((np.count_nonzero(m) / m.size for m in masks), default=None)
                problem = None
                if any(v is None for v in values):
                    problem = "The tracker found nothing at that point; point again on the object."
                elif target_area is not None and target_area > limit:
                    problem = (f"The tracker selected {target_area:.0%} of the image, not one object: the point "
                               "missed it. Point again at the middle of the object.")
                if problem:
                    self._control(lambda: [self.host.stop_service(m, f) for m, f, _ in services], gen=gen)
                    self._log("target_rejected", action=action.id,
                              area=None if target_area is None else round(target_area, 4))
                    raise ToolError(problem)
        seeded = time.monotonic()

        def start() -> int | None:
            skill = tool.skill(params=skill_params)
            started = self.host.start_skill(skill)
            invocation = self.host.skill_invocation(started)
            action.invocation = invocation
            action.status = "running"
            with self._lock:
                self.actions.append(action)
            return invocation

        try:
            self._control(start, gen=gen)
        except (ToolError, StaleCommand):
            raise
        except Exception as exc:
            raise ToolError(f"Could not start {name}: {exc}") from exc
        self._log("action_started", action=action.id, seed_s=round(seeded - requested, 2),
                  start_s=round(time.monotonic() - seeded, 2),
                  **({} if target_area is None else {"target_area": round(target_area, 4)}))
        if self._frames and self._monitor_reference:
            # The marked view shows the monitor the target before the tracker's first mask.
            action.reference = [
                resized(pointed if pointed is not None and v.label == pointed.label else v, self._monitor_side)
                for v in self._frames[-1].views
            ]
        result = {
            "action_id": action.id,
            "status": "running",
            "ends_by_itself": action.ends_by_itself,
        }
        if action.max_duration_s:
            result["times_out_after_s"] = action.max_duration_s
        if action.questions and self.monitor is not None:
            result["monitoring"] = [q.name for q in action.questions]
        return json.dumps(result)

    def _wait_for_field(self, module: str, field_name: str, since: float):
        """The first non-empty value of ``module.field_name`` stamped after ``since``, or ``None``."""
        deadline = time.monotonic() + _SEED_WAIT_S
        while time.monotonic() < deadline:
            data = self.host.field_value(module, field_name)
            if data is not None and data.value is not None and np.any(data.value) \
                    and float(data.timestamp or 0.0) >= since:
                return data.value
            time.sleep(0.05)
        self._log("seed_timeout", module=module, field=field_name)
        return None

    def _stop_by_id(self, action_id: str) -> str:
        with self._lock:
            action = next((a for a in self.actions if a.id == action_id), None)
        if action is None:
            raise ToolError(f"No action {action_id!r}.")
        if not action.running:
            return f"{action_id} already ended: {action.status}."
        self._stop_action(action, "stopped", "stopped by the agent")
        return f"Stopping {action_id}."

    def _stop_action(self, action: AgentAction, outcome: str, reason: str) -> None:
        """Ask the controller to stop ``action``; its end is reported as ``outcome``."""
        with self._lock:
            if not action.running or action.stop_outcome is not None:
                return
            action.stop_outcome = outcome
            action.reason = reason
        invocation = action.invocation

        def stop() -> None:
            if self.host.skill_invocation(action.skill_name) == invocation:
                self.host.stop_skill(action.skill_name, reason=reason)

        self._control(stop, check=False)
        self._log("action_stop", action=action.id, outcome=outcome, reason=reason)

    def _control(self, fn: Callable[[], Any], gen: int | None = None, check: bool = True):
        """Run ``fn`` on the control thread; refuse it if the generation moved on."""
        def guarded():
            if check and gen is not None and gen != self.generation:
                raise StaleCommand()
            return fn()

        future = self.host.call_on_control_thread(guarded)
        try:
            return future.result(timeout=_CONTROL_TIMEOUT_S)
        except FutureTimeout:
            raise ToolError("The controller did not respond; is the control loop running?") from None

    # ---- monitor thread -----------------------------------------------------------------

    def _monitor_loop(self) -> None:
        while not self._closed.is_set():
            t0 = time.monotonic()
            try:
                self._monitor_tick()
            except Exception:
                logger.exception("Agent monitor tick failed")
            wait = max(0.0, self._monitor_period - (time.monotonic() - t0))
            with self._lock:
                deadlines = [a.max_duration_s - a.elapsed_s() for a in self.actions
                             if a.running and a.stop_outcome is None and a.max_duration_s]
            if deadlines:
                # A policy that only ends at its time limit should not run a period past it.
                wait = min(wait, max(0.05, min(deadlines)))
            self._closed.wait(timeout=wait)

    def _monitor_tick(self) -> None:
        with self._lock:
            if not self.status.active or self.status == RunStatus.PAUSED:
                return
            running = [a for a in self.actions if a.running and a.stop_outcome is None]
        for action in running:
            elapsed = action.elapsed_s()
            if action.max_duration_s and elapsed > action.max_duration_s:
                if action.questions and self.monitor is not None:
                    reason = (f"stopped after {action.max_duration_s:.0f} s without the monitor confirming "
                              "success; it may have succeeded anyway, so judge it from the newest image")
                else:
                    reason = (f"ran its {action.max_duration_s:.0f} s limit; it does not end by itself, "
                              "so judge from the newest image whether it succeeded")
                self._stop_action(action, "timeout", reason)
                continue
            tripped = self.host.check_agent_action(action.id, action.tool, elapsed)
            if tripped is not None:
                do, reason, *then = tripped
                self._fire(action, do, reason, by_monitor=False, then=then[0] if then else None)
        if self.monitor is None:
            return
        watched = [a for a in running if a.questions and a.stop_outcome is None]
        if not watched:
            return
        images, state = self._capture()
        frames = []
        for image in images:
            target = target_box(image) if self._mark_target else None
            if image.label in self._close_ups:
                frames.append(close_up(image, self._close_ups[image.label], target))
            else:
                view = make_view(image, self._monitor_side, grid=False, overlays=False, target=target)
                frames.append(ImagePart(jpeg=view.jpeg, label=view.label))
        now = time.monotonic()
        for action in watched:
            with self._lock:
                self._monitor_frames += 1
                frame = self._monitor_frames
            logged = []
            if self.run_log is not None:
                logged = [self.run_log.image(ImagePart(f.jpeg, f"m{frame} {f.label}")) for f in frames]
            action.recent.append((now, frames, logged))
            if self._monitor_history > 1 and len(action.recent) == 1:
                # Choices are defined by what changed since the previous frame: the
                # first one is kept for the next tick, not judged on its own.
                self._log("monitor_frame", action=action.id, images=logged)
                continue
            with self._lock:
                self.monitor_calls += 1
            evidence = MonitorEvidence(
                context=self._monitor_context(action),
                reference=action.reference,
                frames=[
                    ImagePart(jpeg=f.jpeg,
                              label=f"{f.label} " + ("current" if t == now else f"previous, {now - t:.1f} s ago"))
                    for t, tick, _ in action.recent
                    for f in tick
                ],
                data={"elapsed_s": round(action.elapsed_s(), 1), "readings": state},
                cache_key=f"{self.generation}-{action.id}",
            )
            answers, usage = self.monitor.assess(evidence, action.questions)
            with self._lock:
                self.monitor_usage.add(usage)
                action.answers = answers
            if self.run_log is not None:
                # Every frame the monitor was shown, oldest first: the earlier ones were logged by earlier ticks.
                shown = [name for _, _, names in action.recent for name in names]
                self._log("monitor", action=action.id, answers=[a.to_dict() for a in answers], images=shown)
            for firing in action.tracker.update(action.questions, answers, evidence.data["elapsed_s"]):
                self._fire(action, firing.rule.do, f"{firing.question.name}: {firing.reason()}",
                           then=firing.rule.then)

    def _fire(self, action: AgentAction, do: str, reason: str, by_monitor: bool = True,
              then: str | None = None) -> None:
        """``by_monitor`` False: the controller's ``check_agent_action`` fired, not a vision model.

        ``then``: a tool to start without arguments once a finished or failed action has stopped.
        """
        source = "the monitor" if by_monitor else "a controller check"
        if do in (FINISH, FAIL):
            ended = "finished" if do == FINISH else "failed"
            if then is not None:
                action.then = then
            self._stop_action(action, "succeeded" if do == FINISH else "failed", reason)
            self._say("tool", f"{action.id} {ended} by {source} — {reason}")
            if then is None:
                return
            with self._lock:
                gen = self.generation
            try:
                started = json.loads(self._start_action(ToolCall(f"{action.id}-then", then, {}), gen))
            except (ToolError, StaleCommand) as exc:
                self._log("follow_up_failed", action=action.id, tool=then, error=str(exc))
                self._inbox.put(Wake("monitor", f"{action.id} {action.tool} {ended}, but its follow-up "
                                                f"{then} could not start: {exc}"))
                return
            self._log("follow_up", action=action.id, follow_up=started["action_id"], tool=then)
            if do == FINISH:
                # The follow-up carries the finished action's step on, or takes the next step
                # when the plan already names it there; its end moves the run on.
                with self._lock:
                    step = next((s for s in self.plan if s.get("action") == action.id), None)
                    if step is not None:
                        later = self.plan[self.plan.index(step) + 1:]
                        following = next((s for s in later if s["status"] in ("pending", "in_progress")), None)
                        if following is not None and following.get("tool") == then:
                            step["status"] = "done"
                            step = following
                    self._notes.append(
                        f"The monitor finished {action.id} {action.tool} ({reason}; it can be wrong) and "
                        f"started {started['action_id']} {then}, the step its skill always takes next. "
                        "Check the newest image: if it shows the action did not succeed, recover from there.")
                if step is not None:
                    self._attach(started["action_id"], then, step)
                return
            text = (f"{source[0].upper()}{source[1:]} stopped {action.id} {action.tool} as failed "
                    f"({reason}) and started {started['action_id']} {then} to recover. "
                    "Check the newest image before trying again.")
            self._inbox.put(Wake("monitor", text))
        elif do == ESCALATE:
            key = reason.split(":", 1)[0]
            with self._lock:
                if key in action.escalated:
                    return
                action.escalated.add(key)
            self._log("escalate", action=action.id, reason=reason, source="monitor" if by_monitor else "check")
            self._say("tool", f"{action.id} {source}: {reason}")
            if by_monitor:
                text = (f"The monitor (a fast vision model; it can be wrong) flagged {action.id} "
                        f"{action.tool}: {reason}. Check the newest image before acting.")
            else:
                text = f"A controller check on {action.id} {action.tool}: {reason}."
            self._inbox.put(Wake("monitor", text))

    def _monitor_context(self, action: AgentAction) -> str:
        with self._lock:
            step = next((s["step"] for s in self.plan if s.get("action") == action.id), None)
            if step is None:
                step = next((s["step"] for s in self.plan if s["status"] == "in_progress"), "")
        return monitor_context(self.task, action.tool, self.specs[action.tool], action.args, step)

    # ---- bookkeeping ----------------------------------------------------------------------

    def _say(self, role: str, text: str) -> None:
        with self._lock:
            self.transcript.append({"role": role, "text": text, "t": synced_time()})

    def _log(self, kind: str, **data) -> None:
        if self.run_log is not None:
            self.run_log.event(kind, **data)


class ToolError(Exception):
    """A tool call the harness refused; its message goes back to the planner."""
