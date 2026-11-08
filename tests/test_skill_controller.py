"""Controller skill orchestration: preemption, occupancy, inline merge, remote status."""

import json
from collections import deque
from types import SimpleNamespace

import pytest

from solarian.core.action import Action, ActionType
from solarian.core.configs import ControllerRecordingConfig, ModuleConfig
from solarian.core.event import Event, EventType, event_handler
from solarian.core.execution import ExecutionMode, ExecutionPlace
from solarian.core.skill import Skill, SkillStatus
from solarian.core.controller import (
    _REMOTE_STOP_RETRY_S,
    _REMOTE_STOP_TIMEOUT_S,
    Controller,
)
from solarian.core.keyboard_module import KeyboardModule
from solarian.core.recording.episode_log import EpisodeLog
from solarian.utils.shared_memory.observation_buffer import ObservationBuffer


@pytest.fixture(autouse=True)
def schema_free_actions():
    """Exercise the schema-free ``Action`` path.

    Importing any project registers its ``Action`` subclass as the global
    active one, which would validate these fake module/field names against
    that project's schema.
    """
    previous = Action._active_subclass
    Action._active_subclass = None
    yield
    Action._active_subclass = previous


RIGHT_ARM = "right_arm"
RIGHT_GRIPPER = "right_gripper"


class HoldSkill(Skill):
    """Streams one field forever at the modules named in ``params``."""

    execution_mode = ExecutionMode.INLINE

    def __init__(self, name, params):
        super().__init__(
            name=name, params=params, execution_mode=ExecutionMode.INLINE
        )

    def control_step(self, dt):
        return Action(
            action_type=ActionType.MODULE_ACTION,
            data={
                next(iter(self.commanded_module_names())): {
                    "action_type": int(ActionType.SERVO_L),
                    "data": {"x": {"value": 1.0}},
                }
            },
        )


class OneShotSkill(Skill):
    """Emits one exclusive move at each writer named in ``params``, then completes."""

    execution_mode = ExecutionMode.INLINE

    def __init__(self, name=None, params=None):
        super().__init__(
            name=name, params=params, execution_mode=ExecutionMode.INLINE
        )
        self._sent = False

    def setup(self):
        self._sent = False

    def control_step(self, dt):
        if not self._sent:
            self._sent = True
            return Action(
                action_type=ActionType.MODULE_ACTION,
                data={
                    name: {
                        "action_type": int(ActionType.MOVE_L),
                        "data": {
                            "x": {
                                "value": self.params.get("value", 1.0),
                            }
                        },
                        "params": {"speed": None},
                    }
                    for name in self.commanded_module_names()
                },
            )
        self.finish()
        return None


class RemoteSkill(Skill):
    execution_place = ExecutionPlace.ROBOT
    skill_id = 7
    config_name = "cup_arrange"

    def control_step(self, dt):
        return None


class FakeCommandQueue:
    """Stands in for the shared-memory queue the think process writes to."""

    def __init__(self):
        self._items: deque = deque()

    def put(self, action):
        self._items.append(action)

    def empty(self) -> bool:
        return not self._items

    def get(self):
        return self._items.popleft()


class TeleopCtrl(Controller):
    """Controller with only skill orchestration wired up (no zmq / shared memory)."""

    skills = (HoldSkill, OneShotSkill, RemoteSkill)

    def __init__(self):
        self._init_skill_orchestration()
        self.cfg = SimpleNamespace(
            fps=30, waypoint_lookahead_s=0.25, recording=ControllerRecordingConfig()
        )
        self._init_recording()
        self.skill_command_queue = FakeCommandQueue()
        self.skill_status_buffer = None
        self.modules = {}
        self.observation_buffers = {}
        self.keyboard = self.attach(KeyboardModule, "keyboard")
        self._skill_registry = self._collect_controller_skills()

    def attach(self, module_class, name):
        """Add a controller-side module the way ``__init__``/``initialize`` do."""
        module = module_class(ModuleConfig(frequency=60.0), name)
        self.modules[name] = module
        buffer = ObservationBuffer.local()
        self.observation_buffers[name] = buffer
        module.observation_buffer = buffer
        module.start()
        return module

    @event_handler(EventType.KEY_DOWN)
    def start_bound_skill(self, event: Event) -> None:
        """A key starts; ``start_skill`` handles the preemption."""
        match event.data.get("key"):
            case "h":
                self.start_skill(
                    OneShotSkill(
                        name="move_home",
                        params={"arm": RIGHT_ARM, "gripper": RIGHT_GRIPPER},
                    )
                )
            case "m":
                self.start_skill(
                    HoldSkill(
                        "leader_follow",
                        {"arm": RIGHT_ARM, "gripper": RIGHT_GRIPPER},
                    )
                )
            case "c":
                self.start_skill(
                    RemoteSkill(params={"arm": RIGHT_ARM, "gripper": RIGHT_GRIPPER})
                )

    def hold_everything(self) -> str:
        """The skill a key would preempt, standing in for a default teleop."""
        return self.start_skill(
            HoldSkill("hold_all", {"arm": RIGHT_ARM, "gripper": RIGHT_GRIPPER})
        )

    def tick(self, char: str = "w"):
        """One iteration in the order ``control_loop`` runs it."""
        self.keyboard.on_press(char)
        events = self._drain_events()
        self._tick_local_modules()
        self._fan_out_events(events)
        return self.control_step()

    def press(self, char: str) -> None:
        """Press and drain, so handlers run exactly as they do on a tick."""
        self.keyboard.on_press(char)
        self._drain_events()

    def release(self, char: str) -> None:
        self.keyboard.on_release(char)
        self._drain_events()


def test_space_stops_all_and_nothing_comes_back():
    """Space is the operator's stop; no skill resumes behind their back."""
    ctrl = TeleopCtrl()
    ctrl.hold_everything()
    assert "hold_all" in ctrl.running_skills

    ctrl.press(" ")

    assert ctrl.running_skills == {}
    ctrl._tick_skills()
    assert ctrl.running_skills == {}


def test_a_bound_key_preempts_whatever_holds_the_modules():
    ctrl = TeleopCtrl()
    ctrl.hold_everything()

    ctrl.press("m")

    assert "leader_follow" in ctrl.running_skills
    assert "hold_all" not in ctrl.running_skills


def test_pressing_the_same_key_again_re_runs_rather_than_stopping():
    """A key starts, it does not toggle — otherwise nothing commands the arm."""
    ctrl = TeleopCtrl()
    ctrl.press("m")
    first = ctrl._local_skills["leader_follow"]

    ctrl.press("m")

    assert "leader_follow" in ctrl.running_skills
    assert ctrl._local_skills["leader_follow"] is not first


def test_unbound_keys_are_tracked_as_held():
    ctrl = TeleopCtrl()
    ctrl.press("w")
    assert ctrl.held_keys == {"w"}
    ctrl.release("w")
    assert ctrl.held_keys == set()


def test_held_keys_only_change_on_drain():
    """Held keys are state the control loop reads, so an input thread pressing
    mid-tick must not change what the tick already saw."""
    ctrl = TeleopCtrl()
    ctrl.keyboard.on_press("w")
    assert ctrl.held_keys == set()
    ctrl._drain_events()
    assert ctrl.held_keys == {"w"}


def test_a_local_module_publishes_into_its_own_buffer():
    ctrl = TeleopCtrl()
    ctrl.press("w")
    ctrl._tick_local_modules()
    assert ctrl.observations["keyboard"].latest()["held_keys"] == {"w"}


def test_local_modules_are_read_once_per_tick():
    """Two skills following the same module must not mean two hardware reads."""
    reads = []

    class CountingModule(KeyboardModule):
        def update_observations(self):
            reads.append(1)
            self.observation_buffer.put({"n": len(reads)})

    ctrl = TeleopCtrl()
    ctrl.attach(CountingModule, "leader")
    ctrl._tick_local_modules()
    assert len(reads) == 1
    assert ctrl.observations["leader"].latest() == {"n": 1}


def test_a_failing_module_does_not_break_the_tick():
    class BrokenModule(KeyboardModule):
        def update_observations(self):
            raise RuntimeError("device unplugged")

    ctrl = TeleopCtrl()
    ctrl.attach(BrokenModule, "broken")
    ctrl._tick_local_modules()
    assert ctrl.observations["broken"].latest() is None
    assert ctrl.observations["keyboard"].latest() is not None


def test_disjoint_modules_run_together():
    ctrl = TeleopCtrl()
    ctrl.start_skill(HoldSkill("arm_only", {"arm": RIGHT_ARM}))
    ctrl.start_skill(HoldSkill("grip_only", {"gripper": RIGHT_GRIPPER}))
    assert set(ctrl.running_skills) == {"arm_only", "grip_only"}


def test_overlapping_modules_preempt():
    ctrl = TeleopCtrl()
    ctrl.start_skill(
        HoldSkill("arm_and_grip", {"arm": RIGHT_ARM, "gripper": RIGHT_GRIPPER})
    )
    ctrl.start_skill(HoldSkill("grip_only", {"gripper": RIGHT_GRIPPER}))
    assert "grip_only" in ctrl.running_skills
    assert "arm_and_grip" not in ctrl.running_skills


def test_running_skills_includes_robot_status():
    ctrl = TeleopCtrl()
    ctrl.press("c")
    ctrl._tick_skills()
    ctrl._ingest_skill_status_records(
        [SimpleNamespace(skill_id=7, status=int(SkillStatus.RUNNING))]
    )
    rec = ctrl.running_skills["cup_arrange"]
    assert rec.execution_place is ExecutionPlace.ROBOT
    assert rec.modules == {"right_arm", "right_gripper"}


def test_inline_skill_actions_are_merged():
    ctrl = TeleopCtrl()
    ctrl.start_skill(HoldSkill("arm_only", {"arm": RIGHT_ARM}))
    ctrl.start_skill(HoldSkill("grip_only", {"gripper": RIGHT_GRIPPER}))
    action = ctrl._tick_skills()
    assert set(action.to_dict()["data"]) == {"right_arm", "right_gripper"}


class OffLoopSkill(Skill):
    """Stands in for a skill some other scheduler steps.

    ``start`` is a no-op so the test drives the step itself, which is what the
    controller sees of any skill it does not tick: actions simply appear in
    the queues.
    """

    execution_mode = ExecutionMode.THREAD

    def __init__(self, name="off_loop", params=None):
        super().__init__(
            name=name,
            params=params or {"gripper": RIGHT_GRIPPER},
            execution_mode=ExecutionMode.THREAD,
        )

    def start(self):
        pass

    def control_step(self, dt):
        return Action(
            action_type=ActionType.MODULE_ACTION,
            data={
                self.params["gripper"]: {
                    "action_type": int(ActionType.SERVO_J),
                    "data": {"x": {"value": 9.0}},
                }
            },
        )


def test_a_skill_the_loop_does_not_tick_merges_with_one_it_does():
    """Both write to queues, so scheduling no longer decides who gets merged."""
    ctrl = TeleopCtrl()
    ctrl.start_skill(HoldSkill("arm_only", {"arm": RIGHT_ARM}))
    ctrl.start_skill(OffLoopSkill("grip_worker"))

    worker = ctrl._local_skills["grip_worker"]
    worker.dispatch(worker.step())
    data = ctrl._tick_skills().to_dict()["data"]

    assert set(data) == {"right_arm", "right_gripper"}
    assert data["right_gripper"]["data"]["x"]["value"] == 9.0


def test_a_skill_returning_a_non_module_action_says_so(caplog):
    """Starting another skill is the host's call, so this cannot go silently."""

    class MisbehavingSkill(OffLoopSkill):
        def control_step(self, dt):
            return Action(action_type=ActionType.START_SKILL, data={"name": "other"})

    ctrl = TeleopCtrl()
    ctrl.start_skill(MisbehavingSkill("misbehaving"))
    skill = ctrl._local_skills["misbehaving"]

    with caplog.at_level("WARNING"):
        skill.dispatch(skill.step())

    assert "non-module action" in caplog.text
    assert ctrl._tick_skills() is None


def test_a_one_shot_leaves_nothing_running_when_it_completes():
    """The consequence of dropping auto-resume: the operator restarts teleop."""
    ctrl = TeleopCtrl()
    ctrl.hold_everything()
    ctrl.press("h")
    assert "move_home" in ctrl.running_skills
    assert "hold_all" not in ctrl.running_skills

    ctrl._tick_skills()
    ctrl._tick_skills()

    assert ctrl.running_skills == {}


def test_the_key_handler_builds_the_skill_it_starts():
    ctrl = TeleopCtrl()

    ctrl.press("h")
    action = ctrl._tick_skills()

    arm = action.to_dict()["data"]["right_arm"]
    assert arm["action_type"] == int(ActionType.MOVE_L)
    assert arm["params"]["speed"] is None


def test_a_remote_skill_is_sent_as_start_skill():
    """The controller launches nothing; the robot's generic handler does."""
    ctrl = TeleopCtrl()

    ctrl.press("c")
    action = ctrl._tick_skills()

    assert int(action.action_type) == int(ActionType.START_SKILL)
    assert action.data["skill"] == "cup_arrange"
    assert action.data["params"]["arm"] == RIGHT_ARM
    assert action.data["params"]["gripper"] == RIGHT_GRIPPER
    assert "modules" not in action.data
    assert "cup_arrange" in ctrl.running_skills


def test_stopping_a_remote_skill_is_sent_as_stop_skill():
    ctrl = TeleopCtrl()
    ctrl.press("c")
    ctrl._tick_skills()

    ctrl.stop_skill("cup_arrange")
    action = ctrl._tick_skills()

    assert int(action.action_type) == int(ActionType.STOP_SKILL)
    assert action.data["skill"] == "cup_arrange"


def stopping_since(ctrl, name, first_offset=0.0, last_offset=0.0):
    """Backdate a pending stop so the retry and deadline logic can be exercised."""
    first, last = ctrl._remote_stopping[name]
    ctrl._remote_stopping[name] = (first - first_offset, last - last_offset)


def test_a_remote_skill_stays_preemptable_until_the_robot_confirms():
    """Asking is not stopping. Drop the record early and the module set goes
    with it, so nothing can ever preempt the skill again."""
    ctrl = TeleopCtrl()
    ctrl.press("c")
    ctrl._tick_skills()

    ctrl.stop_skill("cup_arrange")

    assert ctrl.running_skills["cup_arrange"].modules == {RIGHT_ARM, RIGHT_GRIPPER}

    ctrl._ingest_skill_status_records(
        [SimpleNamespace(skill_id=7, status=int(SkillStatus.STOPPED))]
    )

    assert "cup_arrange" not in ctrl.running_skills
    assert ctrl._remote_stopping == {}


def test_an_unconfirmed_stop_is_asked_for_again():
    """One STOP_SKILL is not a stop: every hop of the link is CONFLATE, and the
    move that preempted the skill is sent on the very next tick."""
    ctrl = TeleopCtrl()
    ctrl.press("c")
    ctrl._tick_skills()
    ctrl.stop_skill("cup_arrange")
    assert int(ctrl._tick_skills().action_type) == int(ActionType.STOP_SKILL)

    assert ctrl._tick_skills() is None
    stopping_since(ctrl, "cup_arrange", last_offset=_REMOTE_STOP_RETRY_S)
    again = ctrl._tick_skills()

    assert int(again.action_type) == int(ActionType.STOP_SKILL)
    assert again.data["skill"] == "cup_arrange"


def test_the_controller_gives_up_asking_after_the_deadline(caplog):
    """The confirmation travels the same conflating link, so it can be lost too."""
    ctrl = TeleopCtrl()
    ctrl.press("c")
    ctrl._tick_skills()
    ctrl.stop_skill("cup_arrange")
    ctrl._tick_skills()

    stopping_since(ctrl, "cup_arrange", first_offset=_REMOTE_STOP_TIMEOUT_S + 1.0)
    with caplog.at_level("WARNING"):
        assert ctrl._tick_skills() is None

    assert "never confirmed" in caplog.text
    assert ctrl.running_skills == {}


def test_commanded_modules_come_from_params():
    skill = HoldSkill(
        "named_only", {"arm": RIGHT_ARM, "gripper": RIGHT_GRIPPER, "leader": "leader_arm"}
    )

    assert skill.commanded_module_names() == {"right_arm", "right_gripper"}


def test_the_think_process_names_a_skill_and_the_controller_builds_it():
    ctrl = TeleopCtrl()
    ctrl.skill_command_queue.put(
        Action(
            action_type=ActionType.START_SKILL,
            data={"skill": "OneShotSkill", "params": {"arm": RIGHT_ARM}},
        )
    )

    action = ctrl._tick_skills()

    assert "OneShotSkill" in ctrl.running_skills
    assert action.to_dict()["data"]["right_arm"]["params"]["speed"] is None


def test_a_name_the_registry_does_not_have_is_refused(caplog):
    ctrl = TeleopCtrl()
    ctrl.skill_command_queue.put(
        Action(action_type=ActionType.START_SKILL, data={"skill": "nope"})
    )

    with caplog.at_level("WARNING"):
        assert ctrl._tick_skills() is None

    assert "unknown skill 'nope'" in caplog.text
    assert ctrl.running_skills == {}


class RecordingSkill(Skill):
    """Keeps whatever it was handed, so a test can assert on the arguments."""

    execution_mode = ExecutionMode.INLINE

    def __init__(self, name="recorder", params=None):
        super().__init__(
            name=name,
            params=params or {"arm": RIGHT_ARM},
            execution_mode=ExecutionMode.INLINE,
        )
        self.calls = []

    def control_step(self, dt):
        self.calls.append((self.observations, list(self.events), dt))
        return None


def test_a_skill_reads_the_tick_observations_and_events_off_itself():
    ctrl = TeleopCtrl()
    ctrl.start_skill(RecordingSkill())
    ctrl.tick()

    observations, events, dt = ctrl._local_skills["recorder"].calls[-1]
    assert observations["keyboard"].latest()["held_keys"] == {"w"}
    assert [e.data["key"] for e in events] == ["w"]
    assert dt > 0


def test_an_inline_skill_is_wired_to_the_controllers_buffers():
    """The attribute must not be quietly empty just because it runs inline."""
    ctrl = TeleopCtrl()
    skill = RecordingSkill()
    ctrl.start_skill(skill)

    assert ctrl._local_skills["recorder"].observations is ctrl.observation_buffers


def test_events_a_skipped_tick_would_drop_are_kept_for_the_next_step():
    ctrl = TeleopCtrl()
    ctrl.start_skill(RecordingSkill())
    # A queued action makes the tick return before it steps any skill.
    ctrl._pending_actions.append(
        Action(action_type=ActionType.MODULE_ACTION, data={})
    )
    ctrl.tick()

    assert ctrl._local_skills["recorder"].calls == []

    ctrl._tick_skills()

    assert [e.data["key"] for e in ctrl._local_skills["recorder"].events] == ["w"]


def test_every_skill_on_a_tick_sees_the_same_observations():
    ctrl = TeleopCtrl()
    ctrl.start_skill(RecordingSkill("arm_watch", {"arm": RIGHT_ARM}))
    ctrl.start_skill(RecordingSkill("grip_watch", {"gripper": RIGHT_GRIPPER}))
    ctrl.tick()

    first = ctrl._local_skills["arm_watch"].calls[-1][0]
    second = ctrl._local_skills["grip_watch"].calls[-1][0]
    assert first is second


def test_finish_stops_the_skill_but_its_last_action_is_still_sent():
    class LastGaspSkill(OneShotSkill):
        def control_step(self, dt):
            self.finish()
            return super().control_step(dt)

    ctrl = TeleopCtrl()
    ctrl.start_skill(
        LastGaspSkill(name="last_gasp", params={"arm": RIGHT_ARM, "value": 7.0})
    )
    action = ctrl._tick_skills()

    assert action.to_dict()["data"]["right_arm"]["data"]["x"]["value"] == 7.0
    assert "last_gasp" not in ctrl.running_skills


class ChunkSkill(Skill):
    """A policy scheduling a chunk of successive waypoints in one step."""

    execution_mode = ExecutionMode.INLINE

    def __init__(self, name="chunk", params=None, values=(0.0, 1.0, 2.0)):
        params = dict(params or {"arm": RIGHT_ARM})
        params["values"] = values
        super().__init__(
            name=name, params=params, execution_mode=ExecutionMode.INLINE
        )

    def control_step(self, dt):
        module = self.params.get("arm") or self.params["gripper"]
        return [
            Action(
                action_type=ActionType.MODULE_ACTION,
                data={
                    module: {
                        "action_type": int(ActionType.SCHEDULE_L),
                        "data": {"x": {"value": v, "timestamp": 0.0}},
                    }
                },
            )
            for v in self.params["values"]
        ]


def test_a_step_may_return_several_actions():
    """Every waypoint in the chunk reaches the robot, in order."""
    ctrl = TeleopCtrl()
    ctrl.start_skill(ChunkSkill())

    sent = [ctrl._tick_skills() for _ in range(3)]

    assert [a.to_dict()["data"]["right_arm"]["data"]["x"]["value"] for a in sent] == [
        0.0,
        1.0,
        2.0,
    ]


def test_a_chunk_does_not_starve_another_skill_on_the_same_tick():
    """The two skills' first waypoints still travel in the same message."""
    ctrl = TeleopCtrl()
    ctrl.start_skill(ChunkSkill(name="arm_chunk", values=(0.0, 1.0)))
    ctrl.start_skill(
        ChunkSkill(
            name="gripper_chunk",
            params={"gripper": RIGHT_GRIPPER},
            values=(10.0, 11.0),
        )
    )

    first = ctrl._tick_skills().to_dict()["data"]

    assert first["right_arm"]["data"]["x"]["value"] == 0.0
    assert first["right_gripper"]["data"]["x"]["value"] == 10.0


def test_a_shorter_chunk_does_not_pad_the_longer_one():
    """A one-shot skill contributes to the first slot and nothing after it."""
    ctrl = TeleopCtrl()
    ctrl.start_skill(ChunkSkill(name="arm_chunk", values=(0.0, 1.0, 2.0)))
    ctrl.start_skill(
        ChunkSkill(
            name="gripper_once",
            params={"gripper": RIGHT_GRIPPER},
            values=(10.0,),
        )
    )

    sent = [ctrl._tick_skills().to_dict()["data"] for _ in range(3)]

    assert [s["right_arm"]["data"]["x"]["value"] for s in sent] == [0.0, 1.0, 2.0]
    assert "right_gripper" in sent[0]
    assert [s for s in sent[1:] if "right_gripper" in s] == []


# ---- the agent's view of skill lifecycles --------------------------------------------


class FakeAgent:
    def __init__(self):
        self.ended: list[tuple] = []
        self.operator: list[str] = []
        self.messages: list[str] = []
        self.cancelled = 0

    def on_skill_ended(self, invocation, status, reason):
        self.ended.append((invocation, status, reason))

    def on_operator_action(self, description):
        self.operator.append(description)

    def send_user_message(self, text):
        self.messages.append(text)

    def cancel(self):
        self.cancelled += 1


class FailingSkill(Skill):
    execution_mode = ExecutionMode.INLINE

    def control_step(self, dt):
        self.fail("no target in view")


def _agent_ctrl():
    ctrl = TeleopCtrl()
    ctrl.agent = FakeAgent()
    ctrl.running = True
    return ctrl


def _as_agent(ctrl, fn):
    """Run ``fn`` the way the harness does: queued, then drained on the control thread."""
    future = ctrl.call_on_control_thread(fn)
    ctrl._run_control_calls()
    return future.result(timeout=0)


def test_an_agent_started_skill_reports_how_it_was_preempted():
    ctrl = _agent_ctrl()
    name = _as_agent(ctrl, lambda: ctrl.start_skill(
        HoldSkill("hold_all", {"arm": RIGHT_ARM, "gripper": RIGHT_GRIPPER})
    ))
    invocation = ctrl.skill_invocation(name)
    assert ctrl.agent.operator == []

    ctrl.press("m")

    assert ctrl.agent.operator == ["started leader_follow"]
    assert ctrl.agent.ended == [(invocation, SkillStatus.STOPPED, "preempted by leader_follow")]


def test_an_agent_started_skill_reports_its_completion():
    ctrl = _agent_ctrl()
    name = _as_agent(ctrl, lambda: ctrl.start_skill(
        OneShotSkill(name="move_home", params={"arm": RIGHT_ARM})
    ))
    invocation = ctrl.skill_invocation(name)

    ctrl._tick_skills()
    ctrl._tick_skills()

    assert ctrl.agent.ended == [(invocation, SkillStatus.COMPLETED, "")]
    assert ctrl.skill_invocation(name) is None


def test_a_skill_that_fails_reports_its_reason():
    ctrl = _agent_ctrl()
    name = _as_agent(ctrl, lambda: ctrl.start_skill(FailingSkill(params={"arm": RIGHT_ARM})))
    invocation = ctrl.skill_invocation(name)

    ctrl._tick_skills()
    ctrl._tick_skills()

    assert ctrl.agent.ended == [(invocation, SkillStatus.FAILED, "no target in view")]


def test_each_run_of_a_skill_is_its_own_invocation():
    ctrl = _agent_ctrl()
    start = lambda: ctrl.start_skill(HoldSkill("hold_all", {"arm": RIGHT_ARM}))  # noqa: E731
    first = ctrl.skill_invocation(_as_agent(ctrl, start))
    second = ctrl.skill_invocation(_as_agent(ctrl, start))

    assert second != first
    assert ctrl.agent.ended == [(first, SkillStatus.STOPPED, "preempted by hold_all")]


def test_a_control_call_without_a_running_loop_fails_at_once():
    ctrl = TeleopCtrl()
    ctrl.running = False

    with pytest.raises(RuntimeError):
        ctrl.call_on_control_thread(lambda: None).result(timeout=0)


def test_space_is_an_operator_takeover():
    ctrl = _agent_ctrl()
    _as_agent(ctrl, lambda: ctrl.start_skill(HoldSkill("hold_all", {"arm": RIGHT_ARM})))

    ctrl.press(" ")

    assert ctrl.agent.operator == ["stopped all skills"]
    assert ctrl.agent.ended[0][1:] == (SkillStatus.STOPPED, "operator pressed space")


def test_chat_commands_reach_the_agent():
    from solarian.core.ui.base import UICommand, UICommandType

    ctrl = _agent_ctrl()
    ctrl.handle_ui_command(UICommand(UICommandType.AGENT_MESSAGE, {"text": "put the batteries away"}))
    ctrl.handle_ui_command(UICommand(UICommandType.AGENT_CANCEL))

    assert ctrl.agent.messages == ["put the batteries away"]
    assert ctrl.agent.cancelled == 1


# ---- the episode log -----------------------------------------------------------------


def _logged_ctrl(tmp_path):
    ctrl = _agent_ctrl()
    ctrl.controller_recorder = SimpleNamespace(log=EpisodeLog(tmp_path / "logs"))
    return ctrl


def _logged(tmp_path):
    lines = (tmp_path / "logs" / "events.jsonl").read_text().splitlines()
    return [{k: v for k, v in json.loads(line).items() if k != "t"} for line in lines]


def test_the_episode_logs_every_command_and_who_gave_it(tmp_path):
    ctrl = _logged_ctrl(tmp_path)
    _as_agent(ctrl, lambda: ctrl.start_skill(HoldSkill("hold_all", {"arm": RIGHT_ARM})))
    ctrl.press("m")

    assert _logged(tmp_path) == [
        {"kind": "skill_started", "by": "agent", "skill": "hold_all", "params": {"arm": RIGHT_ARM}},
        {"kind": "skill_started", "by": "operator", "skill": "leader_follow",
         "params": {"arm": RIGHT_ARM, "gripper": RIGHT_GRIPPER}},
        {"kind": "skill_stop", "by": "operator", "skill": "hold_all", "reason": "preempted by leader_follow"},
        {"kind": "skill_ended", "skill": "hold_all", "status": "stopped", "reason": "preempted by leader_follow"},
    ]


def test_the_episode_logs_trackers_started_and_stopped(tmp_path):
    ctrl = _logged_ctrl(tmp_path)
    _as_agent(ctrl, lambda: ctrl.start_service("right_wrist_camera", "target_mask", point=[120, 80]))
    ctrl.stop_service("right_wrist_camera", "target_mask")

    assert _logged(tmp_path) == [
        {"kind": "service_started", "by": "agent", "module": "right_wrist_camera", "field": "target_mask",
         "params": {"point": [120, 80]}},
        {"kind": "service_stopped", "by": "operator", "module": "right_wrist_camera", "field": "target_mask"},
    ]


def test_nothing_is_logged_outside_an_episode():
    ctrl = _agent_ctrl()
    ctrl.press("m")
    ctrl.start_service("right_wrist_camera", "target_mask", point=[1, 2])

    assert ctrl.episode_log is None
