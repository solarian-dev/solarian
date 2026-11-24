"""The robot starts skills by import path and stops them by run, without a handler per skill."""

from __future__ import annotations

import pytest

from solarian.core.action import Action, ActionType
from solarian.core.actuator_module import ActuatorModule
from solarian.core.skill import Skill
from solarian.core.robot import Robot


def module(name):
    instance = ActuatorModule.__new__(ActuatorModule)
    instance.name = name
    # What the module answers to, which is what the robot asks before putting
    # a lifecycle action on its queue.
    instance._register_action_handlers()
    return instance


class FakeQueue:
    """A module's action queue, keeping what was put on it for the test."""

    def __init__(self):
        self.actions = []

    def put(self, action):
        self.actions.append(action)


class WaveSkill(Skill):
    """Stands in for a robot-side skill; ``start`` is a no-op so nothing forks."""

    config_name = "wave"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.did_start = False
        self.was_asked_to_stop = False
        self.was_terminated = False
        self.join_calls = 0

    def start(self):
        self.did_start = True

    def is_alive(self) -> bool:
        return self.did_start and not (self.was_asked_to_stop or self.was_terminated)

    @property
    def pid(self):
        return None

    def request_stop(self):
        self.was_asked_to_stop = True

    def terminate(self):
        self.was_terminated = True

    def join(self, timeout=None):
        self.join_calls += 1

    def control_step(self, dt):
        return None


class ArmWaveSkill(WaveSkill):
    config_name = "arm_wave"


class StubbornWaveSkill(WaveSkill):
    """Still alive after being asked to stop, the way an inference in flight is."""

    config_name = "stubborn_wave"

    def is_alive(self) -> bool:
        return self.did_start and not self.was_terminated


CLASSES = {cls.config_name: cls for cls in (WaveSkill, ArmWaveSkill, StubbornWaveSkill)}


class FakeRobot(Robot):
    """Enough of a robot for the skill handlers: no zmq, no shared memory."""

    def __init__(self):
        self.modules = {"arm": module("arm"), "gripper": module("gripper")}
        self.observation_buffers = {"arm": object()}
        self.action_queues = {"arm": FakeQueue(), "gripper": FakeQueue()}
        self.skills_status_buffer = None
        self.disabled_modules = set()
        self._unrouted_action_modules = set()
        self._launch_failures = {}
        self._local_skills = {}
        self._stopping_skills = {}
        self._deferred_starts = {}
        # No cores to reserve: this robot has no encoder.
        self._all_cores = []

    def control_loop(self):
        pass

    def disconnect(self):
        pass


def start(robot, skill, run_id=1, path=None, **data):
    """Send what the controller's ``start_skill`` sends for ``skill``."""
    if path is None:
        path = f"{__name__}:{CLASSES[skill].__qualname__}"
    data = {"skill": skill, "path": path, "run_id": run_id, **data}
    robot.start_named_skill(Action(action_type=ActionType.START_SKILL, data=data), 0.0)


def stop(robot, name, run_id=1):
    robot.stop_named_skill(
        Action(action_type=ActionType.STOP_SKILL, data={"skill": name, "run_id": run_id}), 0.0
    )


def test_a_named_skill_is_looked_up_and_started():
    robot = FakeRobot()

    start(robot, skill="wave", params={"amplitude": 2.0})

    skill = robot._local_skills["wave"]
    assert skill.did_start
    assert skill.params["amplitude"] == 2.0


def test_a_started_skill_is_wired_to_the_robots_shared_memory():
    robot = FakeRobot()

    start(robot, skill="wave")

    skill = robot._local_skills["wave"]
    assert skill.observations is robot.observation_buffers
    assert skill.action_queues is robot.action_queues
    assert skill.event_inbox is not None


def test_module_names_travel_in_params():
    """The names the controller sent are what the skill addresses."""
    robot = FakeRobot()

    start(robot, skill="wave", params={"arm": "arm", "gripper": "gripper"})

    skill = robot._local_skills["wave"]
    assert skill.commanded_module_names() == {"arm", "gripper"}
    assert skill.params["arm"] == "arm"


def test_absent_writer_params_command_nothing():
    robot = FakeRobot()

    start(robot, skill="arm_wave")

    assert robot._local_skills["arm_wave"].commanded_module_names() == set()


def test_the_skill_carries_the_run_id_its_status_rows_report():
    robot = FakeRobot()

    start(robot, skill="wave", run_id=42)

    assert robot._local_skills["wave"].skill_id == 42


@pytest.mark.parametrize("path", ["nope", "no_such_module:Wave", f"{__name__}:Missing", "builtins:dict"])
def test_a_path_that_is_not_an_importable_skill_is_refused(caplog, path):
    robot = FakeRobot()

    with caplog.at_level("WARNING"):
        start(robot, skill="nope", path=path)

    assert "Cannot start skill 'nope'" in caplog.text
    assert robot._local_skills == {}


def test_starting_a_skill_that_is_already_running_is_a_no_op():
    robot = FakeRobot()
    start(robot, skill="wave")
    first = robot._local_skills["wave"]

    start(robot, skill="wave")

    assert robot._local_skills["wave"] is first


def test_a_newer_run_replaces_one_whose_stop_was_lost():
    """The STOP for the old run was conflated away; dropping the new start would
    leave the controller waiting on a run the robot never had."""
    robot = FakeRobot()
    start(robot, skill="wave", params={"arm": "arm"}, run_id=1)
    first = robot._local_skills["wave"]

    start(robot, skill="wave", params={"arm": "arm"}, run_id=2)
    robot._reap_stopping_skills()

    assert first.was_asked_to_stop
    assert robot._local_skills["wave"].skill_id == 2


def test_a_retried_stop_for_a_replaced_run_does_not_stop_its_replacement():
    robot = FakeRobot()
    start(robot, skill="wave", run_id=1)
    stop(robot, "wave", run_id=1)
    robot._reap_stopping_skills()
    start(robot, skill="wave", run_id=2)

    stop(robot, "wave", run_id=1)

    assert robot._local_skills["wave"].skill_id == 2
    assert not robot._local_skills["wave"].was_asked_to_stop


def test_stop_skill_asks_the_skill_to_stop_and_forgets_it():
    robot = FakeRobot()
    start(robot, skill="wave")
    skill = robot._local_skills["wave"]

    stop(robot, "wave")

    assert skill.was_asked_to_stop
    assert robot._local_skills == {}


def test_stopping_a_skill_does_not_join_it_on_the_control_loop():
    """Joining here holds the loop, and a CONFLATE link forgets every command
    the controller sends while it is held — including the move that preempted
    this skill."""
    robot = FakeRobot()
    start(robot, skill="wave", params={"arm": "arm"})
    skill = robot._local_skills["wave"]

    stop(robot, "wave")

    assert skill.join_calls == 0
    assert "wave" in robot._stopping_skills


def test_reaping_a_stopped_skill_cancels_the_waypoints_it_left():
    """A chunk outlives its author, so the next skill would have to out-argue it."""
    robot = FakeRobot()
    start(robot, skill="wave", params={"arm": "arm"})
    stop(robot, "wave")

    robot._reap_stopping_skills()

    assert robot._stopping_skills == {}
    assert [int(a.action_type) for a in robot.action_queues["arm"].actions] == [
        int(ActionType.CANCEL_TRAJECTORY)
    ]
    assert robot.action_queues["gripper"].actions == []


def test_a_skill_still_finishing_its_step_is_not_cancelled_yet():
    """Its last tick can still dispatch a chunk, which a cancel now would miss."""
    robot = FakeRobot()
    start(robot, skill="stubborn_wave", params={"arm": "arm"})
    stop(robot, "stubborn_wave")

    robot._reap_stopping_skills()

    assert "stubborn_wave" in robot._stopping_skills
    assert robot.action_queues["arm"].actions == []


def test_a_skill_that_will_not_exit_is_terminated_and_still_cancelled():
    robot = FakeRobot()
    start(robot, skill="stubborn_wave", params={"arm": "arm"})
    stop(robot, "stubborn_wave")
    skill = robot._stopping_skills["stubborn_wave"][0]

    robot._reap_stopping_skills(force=True)

    assert skill.was_terminated
    assert robot._stopping_skills == {}
    assert [int(a.action_type) for a in robot.action_queues["arm"].actions] == [
        int(ActionType.CANCEL_TRAJECTORY)
    ]


def test_a_skill_that_is_still_stopping_is_restarted_once_it_has_stopped():
    """Two processes commanding the same modules would both own the hardware,
    and dropping the start would leave the controller waiting on a skill that
    never runs."""
    robot = FakeRobot()
    start(robot, skill="stubborn_wave", params={"arm": "arm"})
    first = robot._local_skills["stubborn_wave"]
    stop(robot, "stubborn_wave")

    start(robot, skill="stubborn_wave", params={"arm": "arm", "amplitude": 3.0}, run_id=2)

    assert robot._local_skills == {}
    assert robot._stopping_skills["stubborn_wave"][0] is first

    robot._reap_stopping_skills(force=True)

    second = robot._local_skills["stubborn_wave"]
    assert second is not first and second.did_start and second.params["amplitude"] == 3.0
    assert [int(a.action_type) for a in robot.action_queues["arm"].actions] == [
        int(ActionType.CANCEL_TRAJECTORY)
    ]


def test_a_restart_waiting_on_a_stop_is_dropped_when_it_is_stopped_too():
    robot = FakeRobot()
    start(robot, skill="stubborn_wave", params={"arm": "arm"})
    stop(robot, "stubborn_wave")
    start(robot, skill="stubborn_wave", params={"arm": "arm"}, run_id=2)

    stop(robot, "stubborn_wave", run_id=2)
    robot._reap_stopping_skills(force=True)

    assert robot._local_skills == {}


def test_both_handlers_are_registered_on_every_robot():
    handlers = {}
    for cls in reversed(Robot.__mro__):
        for name, value in cls.__dict__.items():
            if callable(value) and hasattr(value, "_handles_action"):
                handlers[int(getattr(value, "_handles_action"))] = name

    assert handlers[int(ActionType.START_SKILL)] == "start_named_skill"
    assert handlers[int(ActionType.STOP_SKILL)] == "stop_named_skill"
