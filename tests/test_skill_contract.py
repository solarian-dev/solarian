"""Skill control contract: (observations, events, dt) -> actions, done via finish()."""

import pytest

from solarian.core.action import Action, ActionType
from solarian.core.skill import Skill, SkillStatus
from solarian.core.execution import ExecutionMode, WorkerExit


@pytest.fixture(autouse=True)
def schema_free_actions():
    """Exercise the schema-free ``Action`` path (see test_skill_controller)."""
    previous = Action._active_subclass
    Action._active_subclass = None
    yield
    Action._active_subclass = previous


class FakeBuffer:
    def __init__(self, value):
        self.value = value
        self.reads = 0

    def get(self):
        self.reads += 1
        return self.value


class FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class FakeStatusBuffer:
    def __init__(self):
        self.records = []

    def put(self, skill_id, status, timestamp):
        self.records.append(int(status))


def waypoint(module_name, value):
    return Action(
        action_type=ActionType.MODULE_ACTION,
        data={
            module_name: {
                "action_type": int(ActionType.SCHEDULE_L),
                "data": {"x": {"value": value, "timestamp": 0.0}},
            }
        },
    )


class RobotSkill(Skill):
    """Robot-side skill: writes straight into per-module queues."""

    execution_mode = ExecutionMode.INLINE

    def __init__(self, steps=1, **kwargs):
        super().__init__(name="robot_skill", **kwargs)
        self.steps = steps
        self.seen = []

    def control_step(self, dt):
        self.seen.append((self.observations, list(self.events), dt))
        if len(self.seen) >= self.steps:
            self.finish()
        return waypoint("arm", float(len(self.seen)))


def make_skill(cls=RobotSkill, **kwargs):
    return cls(
        observations={"arm": FakeBuffer("arm-obs"), "camera": FakeBuffer("frame")},
        action_queues={"arm": FakeQueue()},
        skills_status_buffer=FakeStatusBuffer(),
        **kwargs,
    )


class TestCompletion:
    def test_a_fresh_skill_is_not_finished(self):
        assert not make_skill().finished

    def test_finish_ends_the_worker_loop(self):
        skill = make_skill(steps=1)
        skill.worker_setup(reconnect=False)
        assert skill.worker_tick() is False

    def test_the_step_that_finishes_still_gets_its_action_out(self):
        skill = make_skill(steps=1)
        skill.worker_setup(reconnect=False)
        skill.worker_tick()
        assert len(skill.action_queues["arm"].items) == 1

    def test_restarting_clears_a_previous_completion(self):
        skill = make_skill(steps=1)
        skill.worker_setup(reconnect=False)
        skill.worker_tick()
        assert skill.finished
        skill.begin()
        assert not skill.finished


class TestArguments:
    def test_dt_is_measured_between_steps(self):
        skill = make_skill(steps=3)
        skill.begin()
        skill.step()
        skill.step()
        assert skill.seen[1][2] > 0

    def test_first_dt_falls_back_to_the_nominal_period(self):
        skill = make_skill(steps=3, frequency=20.0)
        skill.begin()
        skill.step()
        assert skill.seen[0][2] == pytest.approx(0.05)


class TestObservations:
    def test_buffers_are_keyed_by_module_name(self):
        skill = make_skill(steps=3)
        assert skill.observations["arm"].get() == "arm-obs"

    def test_unread_modules_are_never_fetched(self):
        """A skill watching joints must not pay to copy a camera frame."""
        skill = make_skill(steps=3)
        skill.observations["arm"].get()
        assert skill.observations["camera"].reads == 0

    def test_a_module_with_no_buffer_reads_as_missing(self):
        skill = make_skill(steps=3)
        assert skill.observations.get("gripper") is None

    def test_the_step_reads_the_buffers_the_host_wired_up(self):
        skill = make_skill(steps=3)
        skill.begin()
        skill.step()
        assert skill.seen[0][0] is skill.observations


class TestDispatch:
    def test_module_actions_go_to_their_module_queue(self):
        skill = make_skill(steps=3)
        skill.dispatch(waypoint("arm", 1.0))
        assert len(skill.action_queues["arm"].items) == 1

    def test_a_list_is_dispatched_in_order(self):
        skill = make_skill(steps=3)
        skill.dispatch([waypoint("arm", 1.0), waypoint("arm", 2.0)])
        values = [
            item.to_dict()["data"]["x"]["value"]
            for item in skill.action_queues["arm"].items
        ]
        assert values == [1.0, 2.0]

    def test_nothing_to_do_is_a_valid_step_result(self):
        skill = make_skill(steps=3)
        skill.dispatch(None)
        assert skill.action_queues["arm"].items == []

    def test_an_unknown_module_is_skipped_not_raised(self):
        skill = make_skill(steps=3)
        skill.dispatch(waypoint("no_such_module", 1.0))
        assert skill.action_queues["arm"].items == []

    def test_a_non_module_action_is_reported_not_dropped(self, caplog):
        """A skill commands modules; starting another skill is the host's call."""
        skill = make_skill(steps=3)

        with caplog.at_level("WARNING"):
            skill.dispatch(Action(action_type=ActionType.START_SKILL, data={}))

        assert "non-module action" in caplog.text
        assert skill.action_queues["arm"].items == []


class TestStatusReporting:
    def test_a_completed_run_reports_completed(self):
        skill = make_skill(steps=1)
        skill.worker_setup(reconnect=False)
        skill.worker_tick()
        skill.worker_teardown(WorkerExit.COMPLETED)
        assert skill.skills_status_buffer.records == [
            int(SkillStatus.RUNNING),
            int(SkillStatus.COMPLETED),
        ]
