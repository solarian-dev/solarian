"""The AI controller's harness, against a fake controller, planner and monitor.

The planner and monitor threads are not started: each test hands the harness
its wakes and monitor ticks itself, so every step is deterministic.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import cv2
import msgpack
import numpy as np
import pytest

from solarian.core.ai.agent.harness import AgentHarness, RunStatus, Wake, _skill_tool_def
from solarian.core.ai.agent.model import (
    AnthropicClient,
    AssistantTurn,
    ImagePart,
    ModelClient,
    OpenAIChatClient,
    OpenAIResponsesClient,
    ToolCall,
    ToolResult,
    Usage,
    UserTurn,
)
from solarian.core.ai.agent.monitor import (
    ChatMonitor,
    DecisionsMonitor,
    Monitor,
    MonitorAnswer,
    MonitorEvidence,
    MonitorQuestion,
    RuleTracker,
    monitor_context,
)
from solarian.core.ai.agent.run_log import merge_into_session
from solarian.core.clock_sync import synced_time
from solarian.core.configs import AgentConfig, ModelEndpointConfig, MonitorConfig
from solarian.core.skill import Skill, SkillStatus, Tool
from solarian.core.ai.agent.vision import mark_target, target_box
from solarian.core.ui.base import UIImage

CAPTURE_W, CAPTURE_H = 1280, 960


class PickSkill(Skill):
    config_name = "pick"
    description = "Pick the object at a point."
    parameters = {
        "type": "object",
        "properties": {"point": {"type": "array", "items": {"type": "number"}}},
        "required": ["point"],
    }
    image_points = ("point",)
    ends_by_itself = False
    monitor = (
        {"name": "held", "type": "predicate", "instructions": "The object is held.",
         "when": {"do": "finish", "threshold": 0.8, "consecutive": 2}},
    )

    @classmethod
    def services_for(cls, params):
        return [(params["camera"], "target_mask", {"point": params["point"]})]

    def control_step(self, dt):
        return None


class HomeSkill(Skill):
    config_name = "home"
    description = "Move home."
    max_duration_s = 0.01

    def control_step(self, dt):
        return None


class HiddenSkill(Skill):
    config_name = "hidden"

    def control_step(self, dt):
        return None


class FakeHost:
    """The controller, as the harness sees it: commands run at once, on the calling thread."""

    def __init__(self, root=None):
        #: Where the controller's recordings would go; ``None`` records nothing.
        self.root = root
        self.recording_state: dict = {
            "targets": {"controller": False, "robot": False},
            "enabled": False,
            "active": False,
            "episode": 0,
            "session": None,
            "robot": {"enabled": False, "active": False, "episode": 0, "session": None},
            "controller": {"active": False, "episode": 0, "session": None},
        }
        self.harness: AgentHarness | None = None
        self.started: list[Skill] = []
        self.stopped: list[tuple[str, str]] = []
        self.services: list[tuple] = []
        self.services_stopped: list[tuple] = []
        #: What a seeded tracker publishes; ``None`` publishes nothing.
        self.target = np.ones(2)
        self.running: dict[str, int] = {}
        self.invocations = 0
        self.idle_calls = 0
        self.tripwire = None
        self.image = UIImage(label="cam/image", data=np.full((CAPTURE_H, CAPTURE_W, 3), 90, np.uint8))

    def call_on_control_thread(self, fn) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn())
        except Exception as exc:  # noqa: BLE001 - the harness reads it off the future
            future.set_exception(exc)
        return future

    def agent_inputs(self):
        return [self.image], {"arm": {"z": "0.100"}}

    def start_skill(self, skill) -> str:
        self.invocations += 1
        self.running[skill.skill_name] = self.invocations
        self.started.append(skill)
        return skill.skill_name

    def skill_invocation(self, name):
        return self.running.get(name)

    def skill_active(self, name) -> bool:
        return name in self.running

    def stop_skill(self, name, reason=""):
        self.stopped.append((name, reason))
        invocation = self.running.pop(name, None)
        if invocation is not None:
            self.harness.on_skill_ended(invocation, SkillStatus.STOPPED, reason)

    def end(self, name, status=SkillStatus.COMPLETED, reason=""):
        """The skill ended on its own, as the controller would report it."""
        self.harness.on_skill_ended(self.running.pop(name), status, reason)

    def start_service(self, module, field, **params):
        self.services.append((module, field, params))

    def stop_service(self, module, field):
        self.services_stopped.append((module, field))

    def field_value(self, module, field):
        return SimpleNamespace(value=self.target, timestamp=synced_time())

    def arm(self, *, controller=False, robot=False):
        """Flip the switches, with a robot that can record."""
        state = self.recording_state
        state["targets"] = {"controller": controller, "robot": robot}
        state["robot"]["enabled"] = True
        state["enabled"] = controller or robot

    def start_recording(self, task=None, session=None):
        state = self.recording_state
        for side in ("robot", "controller"):
            if state["targets"][side]:
                state[side].update(active=True, session="0001_abcdef_S1", episode=3)
        state.update(active=True, session="0001_abcdef_S1", episode=3)

    def stop_recording(self):
        state = self.recording_state
        state["active"] = state["robot"]["active"] = state["controller"]["active"] = False

    def controller_episode_dir(self):
        state = self.recording_state
        if not state["controller"]["active"]:
            return None
        episode_dir = self.root / state["controller"]["session"] / "episodes" / f"{state['controller']['episode']:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        return episode_dir

    def on_agent_idle(self):
        self.idle_calls += 1

    def check_agent_action(self, action_id, tool, elapsed_s):
        return self.tripwire


class ScriptedPlanner(ModelClient):
    """Replies from a script; each entry is a list of tool calls, or text."""

    def __init__(self, script):
        super().__init__(ModelEndpointConfig(model="scripted"))
        self.script = list(script)
        self.seen: list[list] = []

    def complete(self, system, messages, tools):
        self.seen.append(list(messages))
        step = self.script.pop(0) if self.script else "Waiting."
        if isinstance(step, str):
            return AssistantTurn(text=step)
        return AssistantTurn(tool_calls=[
            ToolCall(id=f"c{len(self.seen)}_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(step)
        ])


class FixedMonitor(Monitor):
    def __init__(self, probability=0.0, **cfg):
        super().__init__(MonitorConfig(model="fixed", **cfg))
        self.probability = probability
        self.calls = 0
        self.evidence: list[MonitorEvidence] = []

    def assess(self, evidence, questions):
        self.calls += 1
        self.evidence.append(evidence)
        answers = [MonitorAnswer(name=q.name, probability=self.probability) for q in questions]
        return answers, Usage(input_tokens=100, output_tokens=5)


def _harness(tmp_path, script, monitor=None, **cfg):
    host = FakeHost(tmp_path)
    tools = {
        "pick": Tool(PickSkill, {"camera": "cam", "arm": "arm"}),
        "home": Tool(HomeSkill, {"arm": "arm"}),
        "hidden": Tool(HiddenSkill),
    }
    config = AgentConfig(
        planner=ModelEndpointConfig(model="scripted"), monitor=monitor.cfg if monitor else None, **cfg
    )
    harness = AgentHarness(host, config, tools, planner=ScriptedPlanner(script), monitor=monitor)
    host.harness = harness
    return harness, host


def _next(harness) -> None:
    """Handle everything the harness has been woken with."""
    wakes = []
    while not harness._inbox.empty():
        wakes.append(harness._inbox.get_nowait())
    harness._handle(wakes)


def _results(turn: UserTurn) -> dict[str, ToolResult]:
    return {r.call_id: r for r in turn.tool_results}


PICK_AT_CENTRE = ("pick", {"point": [384, 288], "observation_id": "o1"})


def test_a_tool_without_a_description_stays_off_the_list(tmp_path):
    harness, _ = _harness(tmp_path, [])
    names = {t.name for t in harness.tool_defs}
    assert {"pick", "home", "observe", "update_plan", "finish_task"} <= names
    assert "hidden" not in names and "wait" not in names


def test_a_task_runs_a_skill_until_the_planner_finishes(tmp_path):
    harness, host = _harness(tmp_path, [
        [("update_plan", {"steps": [{"step": "pick", "status": "in_progress"}]}), PICK_AT_CENTRE],
        [("finish_task", {"success": True, "summary": "picked"})],
    ])
    host.arm(controller=True)
    harness._handle([Wake("user", "pick it")])

    assert harness.status == RunStatus.ACTING
    (skill,) = host.started
    # Planner pixels are on a 768-wide view of a 1280-wide capture.
    assert skill.params["point"] == pytest.approx([640.0, 480.0])
    assert host.services == [("cam", "target_mask", {"point": skill.params["point"]})]
    assert harness.plan == [{"step": "pick", "status": "in_progress"}]

    host.end(skill.skill_name)
    _next(harness)

    assert harness.status == RunStatus.DONE
    assert host.idle_calls == 1
    # Starting the pick ended the turn; its results came back when the pick ended.
    woken = harness.planner.seen[1][-1]
    assert [r.call_id for r in woken.tool_results] == ["c1_0", "c1_1"]
    assert any("succeeded" in p for p in woken.content if isinstance(p, str))
    log = harness.run_log.dir
    assert json.loads((log / "run.json").read_text())["status"] == "done"
    kinds = [json.loads(line)["kind"] for line in (log / "events.jsonl").read_text().splitlines()]
    assert kinds[0] == "run_started" and "action_ended" in kinds and kinds[-1] == "run_ended"
    assert sorted(p.name for p in (log / "turns").iterdir()) == ["0001.json", "0002.json"]


def test_a_point_on_an_image_from_before_the_robot_moved_is_refused(tmp_path):
    harness, host = _harness(tmp_path, [[PICK_AT_CENTRE], [PICK_AT_CENTRE], "ok"])
    harness._handle([Wake("user", "pick it")])
    host.end(host.started[0].skill_name)
    _next(harness)

    (result,) = _results(harness.planner.seen[2][-1]).values()
    assert result.is_error and "moved since o1" in result.content[0]
    assert len(host.started) == 1


def test_a_point_outside_the_image_is_refused(tmp_path):
    harness, host = _harness(tmp_path, [[("pick", {"point": [5000, 10], "observation_id": "o1"})], "ok"])
    harness._handle([Wake("user", "pick it")])

    (result,) = _results(harness.planner.seen[1][-1]).values()
    assert result.is_error and "outside" in result.content[0]
    assert host.started == []


HOME_STEP = {"step": "go home", "status": "pending", "tool": "home", "args": {}}


def test_the_harness_runs_complete_plan_steps_without_the_planner(tmp_path):
    harness, host = _harness(tmp_path, [
        [("update_plan", {"steps": [HOME_STEP, {**HOME_STEP, "step": "home again"},
                                    {"step": "check", "status": "pending"}]})],
        "Checked.",
    ])
    host.arm(controller=True)
    harness._handle([Wake("user", "go home twice")])

    assert len(harness.planner.seen) == 1 and harness.status == RunStatus.ACTING
    assert [(s["status"], s.get("action")) for s in harness.plan] == [("in_progress", "a1"), ("pending", None),
                                                                       ("pending", None)]
    host.end("home")
    _next(harness)
    assert len(harness.planner.seen) == 1
    assert [s["status"] for s in harness.plan] == ["done", "in_progress", "pending"]

    host.end("home")
    _next(harness)
    assert len(harness.planner.seen) == 2
    notes = " ".join(p for p in harness.planner.seen[1][-1].content if isinstance(p, str))
    assert "a2 home succeeded" in notes and "Next is step 3 (check), which has no tool" in notes
    assert "Plan: 1. go home [done, a1]; 2. home again [done, a2]; 3. check [pending]." in notes
    events = [json.loads(line) for line in (harness.run_log.dir / "events.jsonl").read_text().splitlines()]
    assert [(e["step"], e["action"]) for e in events if e["kind"] == "plan_step"] == [(1, "a1"), (2, "a2")]


def test_a_step_that_needs_a_point_wakes_the_planner_to_point_it(tmp_path):
    harness, host = _harness(tmp_path, [
        [("update_plan", {"steps": [HOME_STEP, {"step": "pick", "status": "pending", "tool": "pick"}]})],
        [("pick", {"point": [384, 288], "observation_id": "o2"})],
        "All done.",
    ])
    harness._handle([Wake("user", "pick it")])
    host.end("home")
    _next(harness)

    notes = " ".join(p for p in harness.planner.seen[1][-1].content if isinstance(p, str))
    assert "Next is step 2 (pick): point its pick" in notes
    assert harness.plan[1]["action"] == "a2" and harness.plan[1]["status"] == "in_progress"

    host.end("pick")
    _next(harness)
    notes = " ".join(p for p in harness.planner.seen[2][-1].content if isinstance(p, str))
    assert "Every step of the plan is done" in notes


def test_a_failed_step_wakes_the_planner_and_stays_in_progress(tmp_path):
    harness, host = _harness(tmp_path, [[("update_plan", {"steps": [HOME_STEP, HOME_STEP]})], "Hm."])
    harness._handle([Wake("user", "go home")])
    host.end("home", SkillStatus.FAILED, "blocked")
    _next(harness)

    assert len(harness.planner.seen) == 2 and len(host.started) == 1
    assert [s["status"] for s in harness.plan] == ["in_progress", "pending"]


def test_an_escalation_wakes_the_planner_and_the_queue_waits(tmp_path):
    harness, host = _harness(tmp_path, [[("update_plan", {"steps": [HOME_STEP, HOME_STEP]})], "Looks fine."],
                             monitor=FixedMonitor())
    harness._handle([Wake("user", "go home")])
    harness._fire(harness.actions[0], "escalate", "progress=failed (0.80) for 2 ticks")
    _next(harness)

    assert len(harness.planner.seen) == 2 and len(host.started) == 1
    assert harness.status == RunStatus.ACTING and harness.actions[0].running


def test_a_monitor_finish_hands_its_step_to_the_follow_up_and_the_plan_goes_on(tmp_path):
    harness, host = _harness(tmp_path, [
        [("update_plan", {"steps": [{"step": "pick", "status": "in_progress", "tool": "pick"},
                                    {"step": "check", "status": "pending"}]}),
         PICK_AT_CENTRE],
        "Checked.",
    ], monitor=FixedMonitor())
    harness._handle([Wake("user", "pick it")])
    harness._fire(harness.actions[0], "finish", "held: held (0.90) for 2 ticks", then="home")

    assert harness.plan[0]["action"] == "a2" and harness._inbox.empty()
    host.end("home")
    _next(harness)
    assert [s["status"] for s in harness.plan] == ["done", "pending"]
    notes = " ".join(p for p in harness.planner.seen[1][-1].content if isinstance(p, str))
    assert "finished a1 pick" in notes and "Next is step 2 (check)" in notes


def test_a_follow_up_takes_the_next_step_when_the_plan_names_it_there(tmp_path):
    harness, host = _harness(tmp_path, [
        [("update_plan", {"steps": [{"step": "pick", "status": "in_progress", "tool": "pick"},
                                    {**HOME_STEP, "step": "carry home"}, {"step": "check", "status": "pending"}]}),
         PICK_AT_CENTRE],
        "Checked.",
    ], monitor=FixedMonitor())
    harness._handle([Wake("user", "pick it")])
    harness._fire(harness.actions[0], "finish", "held: held (0.90) for 2 ticks", then="home")

    assert [(s["status"], s.get("action")) for s in harness.plan] == [("done", "a1"), ("in_progress", "a2"),
                                                                       ("pending", None)]
    host.end("home")
    _next(harness)
    assert [type(s).__name__ for s in host.started] == ["PickSkill", "HomeSkill"]
    assert len(harness.planner.seen) == 2


def test_a_failed_action_with_a_follow_up_wakes_the_planner_once_and_keeps_the_skills_reason(tmp_path):
    harness, host = _harness(tmp_path, [[PICK_AT_CENTRE]])
    harness._handle([Wake("user", "pick it")])
    host.stop_skill = lambda name, reason="": host.stopped.append((name, reason))
    harness._fire(harness.actions[0], "fail", "the arm has not moved for 6 s", by_monitor=False, then="home")
    host.end("pick", SkillStatus.FAILED, "Timed out waiting for observation buffers to fill")

    wakes = [harness._inbox.get_nowait() for _ in range(harness._inbox.qsize())]
    assert [w.kind for w in wakes] == ["monitor"]
    pick = harness.actions[0]
    assert pick.status == "failed"
    assert pick.reason == ("the arm has not moved for 6 s; the skill reported: "
                           "Timed out waiting for observation buffers to fill")
    assert any("Timed out waiting" in note for note in harness._notes)


def test_a_plan_step_naming_no_skill_is_refused(tmp_path):
    harness, _ = _harness(tmp_path, [[("update_plan", {"steps": [{"step": "x", "tool": "fly"}]})], "ok"])
    harness._handle([Wake("user", "fly")])

    (result,) = _results(harness.planner.seen[1][-1]).values()
    assert result.is_error and "'fly' is not a skill tool" in result.content[0]


def test_without_a_heartbeat_only_events_wake_the_planner(tmp_path):
    harness, _ = _harness(tmp_path, [[PICK_AT_CENTRE]], planner_heartbeat_s=None)
    harness._handle([Wake("user", "pick it")])

    assert harness.status == RunStatus.ACTING and harness._wake_timeout() is None
    assert "while actions run" not in harness.system_prompt


def test_the_monitor_finishes_a_skill_after_its_rule_holds_twice(tmp_path):
    monitor = FixedMonitor(probability=0.9)
    harness, host = _harness(tmp_path, [[PICK_AT_CENTRE]], monitor=monitor)
    harness._handle([Wake("user", "pick it")])

    harness._monitor_tick()
    harness._monitor_tick()
    assert host.stopped == []
    harness._monitor_tick()

    assert monitor.calls == 2
    assert host.stopped and host.stopped[0][1].startswith("held")
    assert harness.actions[0].status == "succeeded"
    usage = harness.ui_state()["usage"]
    assert usage["monitor"]["model"] == "fixed"
    assert (usage["monitor"]["calls"], usage["monitor"]["input_tokens"]) == (2, 200)
    assert usage["planner"]["calls"] == 1


def test_a_skill_that_overruns_its_duration_is_stopped_as_timed_out(tmp_path):
    harness, _ = _harness(tmp_path, [[("home", {})]])
    harness._handle([Wake("user", "go home")])
    time.sleep(0.02)
    harness._monitor_tick()

    assert harness.actions[0].status == "timeout"
    assert "judge from the newest image whether it succeeded" in harness.actions[0].reason
    assert harness._failures["home"] == 1


def test_a_monitored_skill_that_overruns_says_the_monitor_did_not_confirm_it(tmp_path):
    harness, _ = _harness(tmp_path, [[PICK_AT_CENTRE]], monitor=FixedMonitor())
    harness._handle([Wake("user", "pick it")])
    harness.actions[0].max_duration_s = 0.01
    time.sleep(0.02)
    harness._monitor_tick()

    assert harness.actions[0].status == "timeout"
    assert "without the monitor confirming success" in harness.actions[0].reason


def test_a_timeout_fires_at_its_deadline_not_a_monitor_period_later(tmp_path):
    harness, _ = _harness(tmp_path, [[("home", {})]], monitor=FixedMonitor(period_s=3))
    harness._handle([Wake("user", "go home")])
    harness.actions[0].max_duration_s = 0.2
    loop = threading.Thread(target=harness._monitor_loop, daemon=True)
    loop.start()
    try:
        deadline = time.monotonic() + 1.0
        while harness.actions[0].status != "timeout" and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        harness._closed.set()
        loop.join(timeout=1.0)

    assert harness.actions[0].status == "timeout"


def test_a_tool_that_only_times_out_tells_the_planner_to_judge_the_result(tmp_path):
    harness, _ = _harness(tmp_path, [])
    harness.specs["pick"].update(monitor=[], max_duration_s=11)
    pick = _skill_tool_def("pick", harness.specs["pick"])

    assert "Runs until its time limit" in pick.description and "monitor" not in pick.description


def _pick_with_area_limit(tmp_path, target, script):
    harness, host = _harness(tmp_path, script)
    harness.specs["pick"]["max_target_area"] = 0.05
    host.target = target
    host.arm(controller=True)
    harness._handle([Wake("user", "pick it")])
    return harness, host


def _events(harness, kind):
    lines = (harness.run_log.dir / "events.jsonl").read_text().splitlines()
    return [e for e in map(json.loads, lines) if e["kind"] == kind]


def test_a_mask_covering_much_of_the_image_is_refused_before_anything_moves(tmp_path):
    mask = np.zeros((CAPTURE_H, CAPTURE_W), np.uint8)
    mask[: CAPTURE_H // 2] = 1
    harness, host = _pick_with_area_limit(tmp_path, mask, [[PICK_AT_CENTRE], "ok"])

    (result,) = _results(harness.planner.seen[1][-1]).values()
    assert result.is_error and "selected 50% of the image" in result.content[0]
    assert host.started == [] and host.services_stopped == [("cam", "target_mask")]
    assert [e["area"] for e in _events(harness, "target_rejected")] == [0.5]


def test_an_empty_mask_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("solarian.core.ai.agent.harness._SEED_WAIT_S", 0.1)
    harness, host = _pick_with_area_limit(tmp_path, np.zeros((CAPTURE_H, CAPTURE_W), np.uint8),
                                          [[PICK_AT_CENTRE], "ok"])

    (result,) = _results(harness.planner.seen[1][-1]).values()
    assert result.is_error and "found nothing" in result.content[0]
    assert host.started == []


def test_a_mask_the_size_of_an_object_starts_the_pick_and_logs_its_area(tmp_path):
    mask = np.zeros((CAPTURE_H, CAPTURE_W), np.uint8)
    mask[:96, :128] = 1
    harness, host = _pick_with_area_limit(tmp_path, mask, [[PICK_AT_CENTRE]])

    assert [type(s).__name__ for s in host.started] == ["PickSkill"] and host.services_stopped == []
    assert [e["target_area"] for e in _events(harness, "action_started")] == [0.01]


def test_a_tripwire_escalates_once_per_reason(tmp_path):
    harness, host = _harness(tmp_path, [[PICK_AT_CENTRE]])
    harness._handle([Wake("user", "pick it")])
    host.tripwire = ("escalate", "empty grasp: closed on nothing")
    harness._monitor_tick()
    harness._monitor_tick()

    wakes = [harness._inbox.get_nowait() for _ in range(harness._inbox.qsize())]
    assert [w.kind for w in wakes] == ["monitor"]
    assert wakes[0].text.startswith("A controller check on a1 pick: empty grasp")
    assert harness.actions[0].running


def test_a_monitor_flag_reaches_the_planner_as_a_hint(tmp_path):
    harness, _ = _harness(tmp_path, [[PICK_AT_CENTRE]], monitor=FixedMonitor())
    harness._handle([Wake("user", "pick it")])
    harness._fire(harness.actions[0], "escalate", "progress=failed (0.80) for 3 ticks")

    (wake,) = [harness._inbox.get_nowait() for _ in range(harness._inbox.qsize())]
    assert "a fast vision model; it can be wrong" in wake.text
    assert wake.text.endswith("Check the newest image before acting.")


def test_a_rule_counts_no_ticks_before_its_after_s():
    question = MonitorQuestion.from_spec({
        "name": "progress", "type": "choice", "instructions": "Phase.",
        "choices": {"approaching": "", "failed": ""},
        "when": {"do": "escalate", "value": "failed", "threshold": 0.7, "consecutive": 2, "after_s": 6},
    })
    failed = [MonitorAnswer(name="progress", choice="failed", probability=0.9)]
    tracker = RuleTracker()

    assert question.rules[0].after_s == 6.0
    assert tracker.update([question], failed, 2.0) == []
    assert tracker.update([question], failed, 4.0) == []
    assert tracker.update([question], failed, 6.0) == []
    (firing,) = tracker.update([question], failed, 7.5)
    assert firing.question is question


def test_one_question_can_finish_on_one_choice_and_escalate_on_another():
    question = MonitorQuestion.from_spec({
        "name": "progress", "type": "choice", "instructions": "Phase.",
        "choices": {"holding": "", "failed": ""},
        "when": [
            {"do": "finish", "value": "holding", "threshold": 0.85, "consecutive": 1, "then": "home"},
            {"do": "escalate", "value": "failed", "threshold": 0.7, "consecutive": 2},
        ],
    })
    tracker = RuleTracker()
    failed = [MonitorAnswer(name="progress", choice="failed", probability=0.9)]

    assert tracker.update([question], failed, 1.0) == []
    (escalation,) = tracker.update([question], failed, 2.0)
    assert escalation.rule.do == "escalate"
    (finish,) = tracker.update([question], [MonitorAnswer(name="progress", choice="holding", probability=0.9)], 3.0)
    assert (finish.rule.do, finish.rule.then) == ("finish", "home")
    assert finish.reason() == "progress=holding (0.90) for tick"
    with pytest.raises(ValueError, match="cannot have `then`"):
        MonitorQuestion.from_spec({"name": "x", "instructions": "x", "when": {"do": "escalate", "then": "home"}})
    (fail,) = MonitorQuestion.from_spec({"name": "x", "instructions": "x",
                                         "when": {"do": "fail", "then": "home"}}).rules
    assert (fail.do, fail.then) == ("fail", "home")


def test_a_check_that_fails_an_action_starts_its_follow_up(tmp_path):
    harness, host = _harness(tmp_path, [[PICK_AT_CENTRE]])
    harness._handle([Wake("user", "pick it")])
    host.tripwire = ("fail", "the arm has not moved for 5 s", "home")
    harness._monitor_tick()

    pick, home = harness.actions
    assert pick.stop_outcome == "failed" and pick.reason == "the arm has not moved for 5 s"
    assert home.tool == "home" and home.running
    wakes = [harness._inbox.get_nowait() for _ in range(harness._inbox.qsize())]
    assert "A controller check stopped a1 pick as failed" in wakes[-1].text
    assert "started a2 home" in wakes[-1].text


def test_a_finish_rule_with_then_starts_the_next_tool(tmp_path):
    harness, host = _harness(tmp_path, [[PICK_AT_CENTRE]], monitor=FixedMonitor())
    harness._handle([Wake("user", "pick it")])
    pick = harness.actions[0]
    harness._fire(pick, "finish", "progress: progress=holding (0.90) for tick", then="home")

    assert [type(s).__name__ for s in host.started] == ["PickSkill", "HomeSkill"]
    assert pick.stop_outcome == "succeeded"
    assert harness.actions[1].tool == "home" and harness.actions[1].running
    # The planner sleeps on until the follow-up ends, then hears about both.
    assert harness._inbox.empty()
    notes = " ".join(harness._notes)
    assert "a1 pick succeeded" in notes
    assert "finished a1 pick" in notes and "started a2 home" in notes
    host.end("home")
    assert harness._inbox.get_nowait().kind == "action"


def test_a_then_naming_no_tool_is_refused_at_start(tmp_path, monkeypatch):
    monkeypatch.setattr(PickSkill, "monitor", (
        {"name": "held", "type": "predicate", "instructions": "Held.", "when": {"do": "finish", "then": "fly"}},
    ))
    with pytest.raises(ValueError, match="then: 'fly'"):
        _harness(tmp_path, [])


def test_the_monitor_sees_its_recent_frames_oldest_first(tmp_path):
    monitor = FixedMonitor(history=2)
    harness, host = _harness(tmp_path, [[PICK_AT_CENTRE]], monitor=monitor)
    host.arm(controller=True)
    harness._handle([Wake("user", "pick it")])

    for _ in range(3):
        harness._monitor_tick()

    # The first frame has nothing to compare with: it is kept, not judged.
    second, third = ([f.label for f in e.frames] for e in monitor.evidence)
    assert len(third) == 2 and third[0].startswith("cam/image previous, ") and third[1] == "cam/image current"
    assert second[0].endswith("s ago")
    events = [json.loads(line) for line in (harness.run_log.dir / "events.jsonl").read_text().splitlines()]
    (kept,) = [e["images"] for e in events if e["kind"] == "monitor_frame"]
    logged = [e["images"] for e in events if e["kind"] == "monitor"]
    assert [len(images) for images in logged] == [2, 2]
    # The run log names each frame once: a tick shows the previous one's frame and its own.
    assert logged[0] == [kept[0], logged[0][1]] and kept[0].startswith("m1_")
    assert logged[1] == [logged[0][1], logged[1][1]] and logged[1][1].startswith("m3_")
    assert harness.ui_state()["usage"]["monitor"]["calls"] == 2
    # A probability of 0 is logged, not dropped.
    assert all(a["probability"] == 0.0 for e in events if e["kind"] == "monitor" for a in e["answers"])


def test_a_camera_with_a_close_up_shows_the_monitor_only_that_region_at_capture_resolution(tmp_path):
    monitor = FixedMonitor(close_ups={"cam/image": (0.5, 0.5, 1.0, 1.0)}, reference=False)
    harness, host = _harness(tmp_path, [[PICK_AT_CENTRE]], monitor=monitor)
    harness._handle([Wake("user", "pick it")])
    host.image.data[CAPTURE_H // 2:, CAPTURE_W // 2:] = 250
    mask = np.zeros((CAPTURE_H, CAPTURE_W), np.uint8)
    mask[CAPTURE_H * 3 // 4:, CAPTURE_W * 3 // 4:] = 1
    host.image.masks = [mask]
    harness._monitor_tick()
    harness._monitor_tick()

    (evidence,) = monitor.evidence
    assert evidence.reference == []
    previous, current = (f.label for f in evidence.frames)
    assert previous.startswith("cam/image close-up previous, ") and current == "cam/image close-up current"
    close = cv2.imdecode(np.frombuffer(evidence.frames[-1].jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert close.shape[:2] == (CAPTURE_H // 2, CAPTURE_W // 2)
    # No tracker tint: the mask's quarter is as grey as the rest.
    assert np.abs(close.astype(int) - 250).max() < 8


def test_the_target_box_hugs_the_mask_without_covering_any_of_it():
    rgb = np.full((60, 80, 3), 90, np.uint8)
    mask = np.zeros((60, 80), np.uint8)
    mask[20:31, 30:45] = 1
    mask[25, 50] = 1
    box = target_box(UIImage(label="cam/image", data=rgb, masks=[mask]))

    assert box == (30, 20, 50, 30)
    marked = mark_target(rgb, box)
    assert (marked[mask > 0] == 90).all()
    changed = np.argwhere((marked != rgb).any(axis=2))
    assert changed[:, 0].min() == 17 and changed[:, 0].max() == 33
    assert changed[:, 1].min() == 27 and changed[:, 1].max() == 53
    # One pixel wide: the inside of the box is untouched.
    assert not (marked[18:33, 28:53] != rgb[18:33, 28:53]).any()
    assert (mark_target(rgb, (200, 200, 220, 220)) == rgb).all()


def test_mark_target_boxes_the_tracked_object_in_the_close_up(tmp_path):
    monitor = FixedMonitor(close_ups={"cam/image": (0.5, 0.5, 1.0, 1.0)}, reference=False, mark_target=True)
    harness, host = _harness(tmp_path, [[PICK_AT_CENTRE]], monitor=monitor)
    harness._handle([Wake("user", "pick it")])
    mask = np.zeros((CAPTURE_H, CAPTURE_W), np.uint8)
    cy, cx = CAPTURE_H * 3 // 4, CAPTURE_W * 3 // 4
    mask[cy - 10:cy + 11, cx - 10:cx + 11] = 1
    host.image.masks = [mask]
    harness._monitor_tick()
    harness._monitor_tick()

    close = cv2.cvtColor(cv2.imdecode(np.frombuffer(monitor.evidence[0].frames[-1].jpeg, np.uint8),
                                      cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(int)
    y, x = cy - CAPTURE_H // 2, cx - CAPTURE_W // 2
    assert close[y, x + 13][1] > 150 and close[y, x + 13][0] < 120
    assert np.abs(close[y, x] - 90).max() < 12
    assert "thin cyan box" in monitor.instructions


def test_monitor_frames_carry_no_tracker_overlay(tmp_path):
    monitor = FixedMonitor()
    harness, host = _harness(tmp_path, [[PICK_AT_CENTRE]], monitor=monitor)
    harness._handle([Wake("user", "pick it")])
    mask = np.zeros((CAPTURE_H, CAPTURE_W), np.uint8)
    mask[: CAPTURE_H // 2] = 1
    host.image.masks = [mask]
    harness._monitor_tick()
    harness._monitor_tick()

    frame = cv2.imdecode(np.frombuffer(monitor.evidence[0].frames[-1].jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert np.abs(frame.astype(int) - 90).max() < 8
    assert monitor.evidence[0].reference


def test_a_watch_for_condition_waits_like_a_failure_rule():
    question = MonitorQuestion.watch_for(0, "The battery is back on the table.")
    tracker = RuleTracker()
    true = [MonitorAnswer(name="watch_0", probability=0.95)]

    assert question.rules[0].after_s == 6.0
    assert tracker.update([question], true, 1.5) == [] and tracker.update([question], true, 3.0) == []
    assert tracker.update([question], true, 6.0) == []
    assert len(tracker.update([question], true, 7.5)) == 1


def test_an_answer_keeps_a_zero_probability_and_drops_what_is_unset():
    assert MonitorAnswer(name="held", probability=0.0).to_dict() == {"name": "held", "probability": 0.0}
    assert MonitorAnswer(name="held", refused=True).to_dict() == {"name": "held", "refused": True}


def test_the_monitors_reference_shows_where_the_planner_pointed(tmp_path):
    harness, _ = _harness(tmp_path, [[PICK_AT_CENTRE]], monitor=FixedMonitor())
    harness._handle([Wake("user", "pick it")])

    (reference,) = harness.actions[0].reference
    rgb = cv2.imdecode(np.frombuffer(reference.jpeg, np.uint8), cv2.IMREAD_COLOR)
    # The 768-wide planner view, resized to the monitor's 512: the point lands at (256, 192).
    assert np.abs(rgb[192, 246:267].astype(int) - 90).max() > 60
    assert np.abs(rgb[20, 20].astype(int) - 90).max() < 20


def test_an_operator_takeover_pauses_the_run_until_the_user_writes(tmp_path):
    harness, _ = _harness(tmp_path, [[PICK_AT_CENTRE], "Resuming."])
    harness._handle([Wake("user", "pick it")])
    generation = harness.generation
    harness.on_operator_action("pressed w")

    assert harness.status == RunStatus.PAUSED and harness.generation == generation + 1
    _next(harness)
    assert len(harness.planner.seen) == 1

    harness._handle([Wake("user", "carry on")])
    assert len(harness.planner.seen) == 2
    notes = " ".join(p for p in harness.planner.seen[1][-1].content if isinstance(p, str))
    assert "operator took control" in notes and "The user says: carry on" in notes


def test_cancel_stops_what_runs_and_hands_the_arm_back(tmp_path):
    harness, host = _harness(tmp_path, [[PICK_AT_CENTRE]])
    harness._handle([Wake("user", "pick it")])
    harness._handle([Wake("cancel")])

    assert harness.status == RunStatus.CANCELLED
    assert [name for name, _ in host.stopped] == ["pick"]
    assert harness.actions[0].status == "stopped"
    assert host.idle_calls == 1


def test_reset_forgets_the_conversation(tmp_path):
    harness, _ = _harness(tmp_path, ["Hello."])
    harness._handle([Wake("user", "hi")])
    harness._handle([Wake("reset")])

    assert harness.status == RunStatus.IDLE
    assert harness.messages == [] and list(harness.transcript) == []


def test_a_run_out_of_turns_fails(tmp_path):
    harness, _ = _harness(tmp_path, [[("observe", {})]] * 5, budgets={"max_turns": 2})
    harness._handle([Wake("user", "look around")])

    assert harness.status == RunStatus.FAILED
    assert len(harness.planner.seen) == 2


def test_a_tool_that_keeps_failing_is_refused(tmp_path):
    harness, _ = _harness(tmp_path, [], budgets={"retries_per_step": 1})
    harness._handle([Wake("user", "go home")])
    harness._failures["home"] = 1
    result = harness._execute(ToolCall("x", "home", {}), harness.generation)

    assert result.is_error and "failed 1 times" in result.content[0]


def test_a_dry_run_starts_nothing(tmp_path):
    harness, host = _harness(tmp_path, [[PICK_AT_CENTRE], "done"], dry_run=True)
    harness._handle([Wake("user", "pick it")])

    assert host.started == [] and harness.actions[0].status == "dry_run"


def test_a_recorded_run_logs_beside_the_controllers_episode(tmp_path):
    """One run is one episode, and its id is where to find it: session/episode."""
    harness, host = _harness(tmp_path, [[("finish_task", {"success": True, "summary": "ok"})]])
    host.arm(controller=True, robot=True)
    harness._handle([Wake("user", "do it")])

    assert harness.run_log.dir == tmp_path / "0001_abcdef_S1" / "episodes" / "000003" / "agent"
    assert harness.run_id == "0001_abcdef_S1/000003"
    assert json.loads((harness.run_log.dir / "run.json").read_text())["run_id"] == harness.run_id
    assert host.recording_state["active"] is False


def test_a_run_only_the_robot_records_keeps_nothing_of_the_agent(tmp_path):
    harness, host = _harness(tmp_path, [[("finish_task", {"success": True, "summary": "ok"})]])
    host.arm(robot=True)
    harness._handle([Wake("user", "do it")])

    assert harness.run_log is None and harness.run_id is None
    assert list(tmp_path.iterdir()) == []
    assert any("not recorded" in entry["text"] for entry in harness.ui_state()["transcript"])


def test_an_unrecorded_run_writes_nothing(tmp_path):
    harness, _ = _harness(tmp_path, [[("finish_task", {"success": True, "summary": "ok"})]])
    harness._handle([Wake("user", "do it")])

    assert harness.run_log is None and list(tmp_path.iterdir()) == []


def test_the_ui_state_crosses_msgpack(tmp_path):
    harness, _ = _harness(tmp_path, [[PICK_AT_CENTRE]])
    harness._handle([Wake("user", "pick it")])

    state = msgpack.unpackb(msgpack.packb(harness.ui_state()))
    assert state["status"] == "acting" and state["actions"][0]["tool"] == "pick"
    assert harness.view_image().jpeg[:2] == b"\xff\xd8"


def test_merge_copies_agent_logs_into_the_robots_session(tmp_path):
    """Both sides share the session directory name, so it is the join key."""
    root = tmp_path / "recordings" / "controller"
    (root / "0001_abcdef_S1" / "episodes" / "000003" / "agent").mkdir(parents=True)
    (root / "0001_abcdef_S1" / "episodes" / "000003" / "agent" / "run.json").write_text("{}")
    session = tmp_path / "recordings" / "robot" / "0001_abcdef_S1"
    (session / "episodes" / "000003").mkdir(parents=True)

    assert merge_into_session(root, session) == [session / "episodes" / "000003" / "agent"]
    assert (session / "episodes" / "000003" / "agent" / "run.json").exists()


# ---- wire formats ---------------------------------------------------------------------


def _conversation():
    image = ImagePart(jpeg=b"\xff\xd8jpeg", label="o1 cam/image")
    return [
        UserTurn(content=["Task: pick", image]),
        AssistantTurn(text="Looking.", tool_calls=[ToolCall("t1", "observe", {})]),
        UserTurn(tool_results=[ToolResult("t1", ["Observation o2", image])], content=["note"]),
    ]


def test_anthropic_messages_carry_tool_results_and_cache_the_end():
    client = AnthropicClient.__new__(AnthropicClient)
    out = client._messages(_conversation())

    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[1]["content"][1] == {"type": "tool_use", "id": "t1", "name": "observe", "input": {}}
    result = out[2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "t1"
    assert any(block["type"] == "image" for block in result["content"])
    assert out[2]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_openai_messages_move_tool_images_into_a_user_message():
    client = OpenAIChatClient.__new__(OpenAIChatClient)
    out = client._messages("system", _conversation())

    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "user"]
    assert "[image o1 cam/image follows]" in out[3]["content"]
    assert out[4]["content"][1]["type"] == "image_url"


def test_responses_input_replays_its_own_items_and_rebuilds_others():
    client = OpenAIResponsesClient.__new__(OpenAIResponsesClient)
    out = client._input(_conversation())

    assert out[0]["role"] == "user"
    assert out[1] == {"role": "assistant", "content": "Looking."}
    assert out[2] == {"type": "function_call", "call_id": "t1", "name": "observe", "arguments": "{}"}
    assert out[3]["type"] == "function_call_output" and "[image o1 cam/image follows]" in out[3]["output"]
    assert out[4]["content"][1]["type"] == "input_image"

    reasoning = {"type": "reasoning", "id": "rs_1", "encrypted_content": "x", "summary": []}
    own = AssistantTurn(tool_calls=[ToolCall("c1", "observe", {})], provider="openai_responses",
                        native=[reasoning, {"type": "function_call", "call_id": "c1", "name": "observe", "arguments": "{}"}])
    assert client._input([own])[0] == reasoning


def test_the_monitor_is_told_what_it_can_use_of_the_action():
    spec = {"description": "Grasp the object and lift it. Point at it in the wrist image.", "image_points": ["point"]}
    args = {"point": [405, 394], "observation_id": "o2", "watch_for": ["dropped"], "speed": 0.5}

    context = monitor_context("tidy up", "pick", spec, args, step="pick the cup")
    assert context == ("Task: tidy up\nCurrent step: pick the cup\n"
                       'Running action: pick: Grasp the object and lift it.\nArguments: {"speed": 0.5}')
    assert monitor_context("t", "pick", spec, {"point": [1, 2]}).endswith("lift it.")

    question = MonitorQuestion.from_spec({"name": "phase", "type": "choice", "instructions": "Phase.",
                                          "choices": {"approach": "moving in", "lift": "rising"}})
    evidence = MonitorEvidence(context=context, reference=[], frames=[], data={}, cache_key="k")
    assert ChatMonitor.prompt(evidence, [question]).endswith(
        "- phase (choice): Phase.\n  Choices:\n  - approach: moving in\n  - lift: rising")


def test_decisions_usage_reads_the_reply_and_tolerates_its_absence():
    usage = DecisionsMonitor.parse_usage(
        {"usage": {"input_tokens": 1200, "output_tokens": 30, "input_tokens_details": {"cached_tokens": 900}}}
    )
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (1200, 30, 900)
    assert DecisionsMonitor.parse_usage({}).input_tokens == 0


def test_decisions_request_and_answers():
    monitor = DecisionsMonitor(MonitorConfig(model="gpt-6-luna", instructions="The fingers are orange."))
    frame = ImagePart(jpeg=b"\xff\xd8", label="cam/image")
    questions = [
        MonitorQuestion.from_spec({"name": "held", "type": "predicate", "instructions": "Held.",
                                   "when": {"do": "finish"}}),
        MonitorQuestion.from_spec({"name": "phase", "type": "choice", "instructions": "Phase.",
                                   "choices": {"approach": "moving in", "lift": "rising"}}),
    ]
    body = monitor.request_body(
        MonitorEvidence(context="ctx", reference=[frame], frames=[frame], data={"elapsed_s": 2}, cache_key="r-a1"),
        questions,
    )

    assert body["model"] == "gpt-6-luna" and body["prompt_cache_key"] == "r-a1"
    developer = body["input"][0]["content"][0]["text"]
    assert "close-up" in developer and developer.endswith("\nThe fingers are orange.")
    assert body["input"][1]["content"][-1]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert body["input"][2]["content"][-1] == {"type": "input_json", "data": {"elapsed_s": 2}}
    assert body["questions"][1]["choices"][0] == {"value": "approach", "description": "moving in"}

    answers = DecisionsMonitor.parse_answers({"answers": [
        {"name": "held", "type": "predicate", "probability": 0.91},
        {"name": "phase", "type": "choice", "choice": "lift",
         "probabilities": [{"value": "lift", "probability": 0.7}, {"value": "approach", "probability": 0.3}]},
        {"name": "other", "type": "refusal"},
    ]})
    assert answers[0].probability == 0.91
    assert answers[1].choice == "lift" and answers[1].probability == 0.7
    assert answers[2].refused
