"""The fast loop: typed questions about the camera while an action runs.

A policy skill does not know when it is done or when it is failing, and the
planner is too slow and too expensive to look every second. The monitor asks
a small, cheap model a fixed set of questions on every tick — "is the object
held?", "which phase is the grasp in?" — and a rule on each answer decides
whether to finish the action, wake the planner, or do nothing.

Questions come from two places: the skill's own ``monitor`` contract (what
success and failure look like for that skill), and ``watch_for`` conditions
the planner adds when it starts an action. Rules are debounced: a threshold
has to hold for ``consecutive`` ticks, because one blurry frame is not a
dropped object.
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from solarian.core.ai.agent.model import (
    ImagePart,
    ModelClient,
    Usage,
    UserTurn,
    make_model_client,
)
from solarian.core.configs import MonitorConfig

logger = logging.getLogger(__name__)

FINISH = "finish"
FAIL = "fail"
ESCALATE = "escalate"
#: Seconds into an action before a planner's ``watch_for`` condition counts.
WATCH_FOR_AFTER_S = 6.0


@dataclass(frozen=True)
class MonitorRule:
    #: :data:`FINISH` stops the action and marks it succeeded, :data:`FAIL`
    #: stops it and marks it failed, :data:`ESCALATE` wakes the planner and
    #: leaves it running.
    do: str
    threshold: float = 0.8
    consecutive: int = 2
    #: For a choice question, the value that fires the rule.
    value: str | None = None
    #: Seconds into the action before the rule counts ticks at all: a failure
    #: cannot be judged while the arm is still approaching.
    after_s: float = 0.0
    #: For a rule that stops the action, a tool started without arguments as
    #: soon as it stops, before the planner is woken: the step that always
    #: follows, such as carrying a picked object home.
    then: str | None = None


@dataclass(frozen=True)
class MonitorQuestion:
    name: str
    instructions: str
    #: ``predicate`` (probability the statement is true) or ``choice``.
    type: str = "predicate"
    #: ``(value, description)`` pairs of a choice question.
    choices: tuple[tuple[str, str], ...] = ()
    rules: tuple[MonitorRule, ...] = ()

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> MonitorQuestion:
        """Build from a skill's ``monitor`` entry (a yaml mapping).

        ``choices`` is ``{value: description}``; ``when`` is one rule or a list
        of them, each ``{do, threshold, consecutive, value, after_s, then}``.
        """
        choices = spec.get("choices") or {}
        if isinstance(choices, Mapping):
            pairs = tuple((str(k), str(v)) for k, v in choices.items())
        else:
            pairs = tuple((str(c["value"]), str(c.get("description", ""))) for c in choices)
        when = spec.get("when") or []
        rules = []
        for w in [when] if isinstance(when, Mapping) else when:
            rule = MonitorRule(
                do=str(w["do"]),
                threshold=float(w.get("threshold", 0.8)),
                consecutive=int(w.get("consecutive", 2)),
                value=str(w["value"]) if w.get("value") is not None else None,
                after_s=float(w.get("after_s", 0.0)),
                then=str(w["then"]) if w.get("then") else None,
            )
            if rule.do not in (FINISH, FAIL, ESCALATE):
                raise ValueError(f"monitor question {spec['name']!r}: when.do must be finish, fail or escalate")
            if rule.then and rule.do == ESCALATE:
                raise ValueError(f"monitor question {spec['name']!r}: an escalate rule leaves the action "
                                 "running, so it cannot have `then`")
            rules.append(rule)
        question_type = str(spec.get("type", "predicate"))
        if question_type == "choice" and not pairs:
            raise ValueError(f"monitor question {spec['name']!r} is a choice with no choices")
        return cls(
            name=str(spec["name"]),
            instructions=str(spec["instructions"]),
            type=question_type,
            choices=pairs,
            rules=tuple(rules),
        )

    @classmethod
    def watch_for(cls, index: int, condition: str) -> MonitorQuestion:
        """A planner-supplied failure condition: escalate when it holds.

        Quiet for the first seconds, like a skill's own failure rules: a
        condition such as "the object is on the table" is true during every
        approach.
        """
        return cls(
            name=f"watch_{index}",
            instructions=f"Is this true in the latest frame? {condition}",
            rules=(MonitorRule(do=ESCALATE, threshold=0.75, consecutive=2, after_s=WATCH_FOR_AFTER_S),),
        )


@dataclass
class MonitorAnswer:
    name: str
    #: Predicate: probability the statement is true. Choice: probability of ``choice``.
    probability: float | None = None
    choice: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    refused: bool = False

    def summary(self) -> str:
        if self.refused:
            return f"{self.name}: refused"
        if self.choice is not None:
            return f"{self.name}={self.choice} ({self.probability or 0.0:.2f})"
        return f"{self.name} p={self.probability or 0.0:.2f}"

    def to_dict(self) -> dict:
        # Identity, not ==: a probability of 0.0 equals False and must stay.
        return {k: v for k, v in self.__dict__.items() if v is not None and v is not False and v != {}}


@dataclass
class MonitorEvidence:
    #: What is going on, in words: task, plan step, action, elapsed time.
    context: str
    #: Frames at the action's start. Identical for every tick of one action,
    #: so they sit in the cacheable prefix.
    reference: list[ImagePart]
    #: The newest frames.
    frames: list[ImagePart]
    #: Robot readings and action progress.
    data: dict
    #: One cache partition per action.
    cache_key: str


class Monitor(ABC):
    def __init__(self, cfg: MonitorConfig):
        self.cfg = cfg
        self.model = cfg.model
        self.instructions = _INSTRUCTIONS
        if cfg.mark_target:
            self.instructions += " " + _TARGET_NOTE
        if cfg.instructions.strip():
            self.instructions += "\n" + cfg.instructions.strip()

    @abstractmethod
    def assess(
        self, evidence: MonitorEvidence, questions: list[MonitorQuestion]
    ) -> tuple[list[MonitorAnswer], Usage]:
        """The answers, and the tokens the call cost."""


_INSTRUCTIONS = (
    "You watch a robot's camera while it runs one action. Judge only from what is "
    "visible in the frames and the readings provided. The frames come oldest to "
    "newest: the previous ones, each labelled with its age, then the current one. "
    "Compare them to judge what changed. A frame labelled close-up is a fixed region "
    "of the camera at full resolution, such as the gripper and what it holds."
)
_TARGET_NOTE = (
    "A thin cyan box marks the action's target, the object it acts on or the place it acts "
    "at, where a tracker following it has it; the box is drawn just outside it, so the "
    "target itself is not covered. Trackers can slip onto something else or lose part of "
    "the target, so trust what the frame shows over the box."
)
_REFERENCE_CAPTION = "from when the action started, with the planner's point on the target"


def monitor_context(task: str, tool: str, spec: Mapping[str, Any], args: Mapping[str, Any], step: str = "") -> str:
    """What the monitor is told about the running action, in words.

    The description's first sentence only: the rest tells the planner how to
    call the tool. Pointing arguments are left out, since their pixels are in
    an image the monitor does not see.
    """
    description = re.split(r"(?<=\.)\s", str(spec.get("description") or "").strip(), maxsplit=1)[0]
    hidden = {"observation_id", "watch_for", *(spec.get("image_points") or ())}
    shown = {k: v for k, v in args.items() if k not in hidden}
    return (
        f"Task: {task}\n"
        + (f"Current step: {step}\n" if step else "")
        + f"Running action: {tool}: {description}"
        + (f"\nArguments: {json.dumps(shown)}" if shown else "")
    )


class DecisionsMonitor(Monitor):
    """OpenAI's ``/v1/decisions``: typed answers, roughly 10x faster than a chat model.

    Called over plain HTTP rather than through the SDK, which may predate the
    endpoint. The reference frames end the cacheable prefix with an explicit
    breakpoint, keyed per action, so every tick of one action re-reads them
    from cache.
    """

    def __init__(self, cfg: MonitorConfig):
        super().__init__(cfg)
        import httpx

        key = os.environ.get(cfg.api_key_env or "OPENAI_API_KEY")
        self._http = httpx.Client(
            base_url=(cfg.base_url or "https://api.openai.com/v1").rstrip("/"),
            headers={"Authorization": f"Bearer {key}"},
            timeout=cfg.timeout_s,
        )

    def request_body(self, evidence: MonitorEvidence, questions: list[MonitorQuestion]) -> dict:
        shared: list[dict] = [{"type": "input_text", "text": evidence.context}]
        for image in evidence.reference:
            shared.append({"type": "input_text", "text": f"Reference frame {image.label}, {_REFERENCE_CAPTION}:"})
            shared.append({"type": "input_image", "image_url": image.data_url()})
        shared[-1] = {**shared[-1], "prompt_cache_breakpoint": {"mode": "explicit"}}
        current: list[dict] = []
        for image in evidence.frames:
            current.append({"type": "input_text", "text": f"Frame {image.label}:"})
            current.append({"type": "input_image", "image_url": image.data_url()})
        current.append({"type": "input_json", "data": evidence.data})
        return {
            "model": self.model,
            "input": [
                {"role": "developer", "content": [{"type": "input_text", "text": self.instructions}]},
                {"role": "user", "content": shared},
                {"role": "user", "content": current},
            ],
            "questions": [self._question(q) for q in questions],
            "prompt_cache_key": evidence.cache_key,
        }

    @staticmethod
    def _question(question: MonitorQuestion) -> dict:
        body: dict[str, Any] = {
            "type": question.type,
            "name": question.name,
            "instructions": question.instructions,
        }
        if question.type == "choice":
            body["choices"] = [{"value": v, "description": d} for v, d in question.choices]
        return body

    @staticmethod
    def parse_answers(payload: dict) -> list[MonitorAnswer]:
        answers = []
        for raw in payload.get("answers") or []:
            name = str(raw.get("name") or "")
            kind = raw.get("type")
            if kind == "predicate":
                answers.append(MonitorAnswer(name=name, probability=float(raw.get("probability") or 0.0)))
            elif kind == "choice":
                probabilities = {
                    str(p.get("value")): float(p.get("probability") or 0.0)
                    for p in raw.get("probabilities") or []
                }
                choice = str(raw.get("choice"))
                answers.append(
                    MonitorAnswer(
                        name=name,
                        choice=choice,
                        probability=probabilities.get(choice, float(raw.get("confidence") or 0.0)),
                        probabilities=probabilities,
                    )
                )
            else:
                answers.append(MonitorAnswer(name=name, refused=True))
        return answers

    @staticmethod
    def parse_usage(payload: dict) -> Usage:
        usage = payload.get("usage") or {}
        details = usage.get("input_tokens_details") or {}
        return Usage(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_read_tokens=int(details.get("cached_tokens") or 0),
        )

    def assess(self, evidence, questions):
        response = self._http.post("/decisions", json=self.request_body(evidence, questions))
        response.raise_for_status()
        payload = response.json()
        return self.parse_answers(payload), self.parse_usage(payload)


class ChatMonitor(Monitor):
    """Any chat model asked for the same typed answers as JSON.

    The fallback when the Decisions API is not available: slower and pricier
    per tick, but the questions, rules and debounce are the same.
    """

    def __init__(self, cfg: MonitorConfig, client: ModelClient | None = None):
        super().__init__(cfg)
        self._client = client or make_model_client(cfg)

    @staticmethod
    def prompt(evidence: MonitorEvidence, questions: list[MonitorQuestion]) -> str:
        lines = [
            evidence.context,
            "Readings: " + json.dumps(evidence.data),
            "Answer every question. Reply with JSON only, shaped as",
            '{"<name>": {"probability": <0..1>}} for a predicate (probability it is true) and',
            '{"<name>": {"choice": "<value>", "probability": <0..1>}} for a choice.',
            "Questions:",
        ]
        for q in questions:
            lines.append(f"- {q.name} ({q.type}): {q.instructions}")
            if q.type == "choice":
                lines += ["  Choices:", *(f"  - {v}: {d}" for v, d in q.choices)]
        return "\n".join(lines)

    @staticmethod
    def parse_answers(text: str, questions: list[MonitorQuestion]) -> list[MonitorAnswer]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            payload = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            payload = {}
        answers = []
        for q in questions:
            raw = payload.get(q.name)
            if not isinstance(raw, dict):
                answers.append(MonitorAnswer(name=q.name, refused=True))
                continue
            probability = float(raw.get("probability") or 0.0)
            if q.type == "choice":
                answers.append(MonitorAnswer(name=q.name, choice=str(raw.get("choice")), probability=probability))
            else:
                answers.append(MonitorAnswer(name=q.name, probability=probability))
        return answers

    def assess(self, evidence, questions):
        content: list = [self.instructions]
        if evidence.reference:
            content += [f"Reference frames, {_REFERENCE_CAPTION}:", *evidence.reference]
        content += ["Frames, oldest first:", *evidence.frames]
        content.append(self.prompt(evidence, questions))
        turn = self._client.complete("You answer questions about robot camera frames.", [UserTurn(content=content)], [])
        return self.parse_answers(turn.text, questions), turn.usage


def make_monitor(cfg: MonitorConfig) -> Monitor:
    if cfg.provider == "openai_decisions":
        return DecisionsMonitor(cfg)
    return ChatMonitor(cfg)


@dataclass
class Firing:
    question: MonitorQuestion
    rule: MonitorRule
    answer: MonitorAnswer

    def reason(self) -> str:
        ticks = "tick" if self.rule.consecutive == 1 else f"{self.rule.consecutive} ticks"
        return f"{self.answer.summary()} for {ticks}"


class RuleTracker:
    """Debounce: a rule fires once its condition has held ``consecutive`` ticks in a row.

    Ticks before the rule's ``after_s`` neither count nor fire.
    """

    def __init__(self) -> None:
        self._streaks: dict[tuple[str, int], int] = {}

    def update(
        self, questions: list[MonitorQuestion], answers: list[MonitorAnswer], elapsed_s: float
    ) -> list[Firing]:
        by_name = {a.name: a for a in answers}
        fired = []
        for question in questions:
            answer = by_name.get(question.name)
            if answer is None:
                continue
            for i, rule in enumerate(question.rules):
                if elapsed_s < rule.after_s:
                    continue
                key = (question.name, i)
                self._streaks[key] = self._streaks.get(key, 0) + 1 if self._holds(rule, answer) else 0
                if self._streaks[key] >= rule.consecutive:
                    fired.append(Firing(question, rule, answer))
        return fired

    @staticmethod
    def _holds(rule: MonitorRule, answer: MonitorAnswer) -> bool:
        if answer.refused or answer.probability is None:
            return False
        if rule.value is not None and answer.choice != rule.value:
            return False
        return answer.probability >= rule.threshold
