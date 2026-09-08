"""A robot whose controller goes quiet stops its skills and holds safe.

The cloud deployment puts the internet between the controller and the follower,
so the interesting cases here are the two that a LAN never made us choose
between: silence because the operator is idle, and silence because the link is
gone.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from solarian.core.action import ActionType, ModuleAction
from solarian.core.actuator_module import ActuatorModule
from solarian.core.configs import ModuleConfig, SafeState
from solarian.core.robot import Robot
from solarian.utils.trajectory_interpolator import ScalarTrajectoryInterpolator


class FakeQueue:
    def __init__(self):
        self.actions: list[ModuleAction] = []

    def put(self, action: ModuleAction) -> None:
        self.actions.append(action)

    def action_types(self) -> list[int]:
        return [int(a.action_type) for a in self.actions]


def module(name: str, *, link_lost: SafeState = SafeState.HOLD) -> ActuatorModule:
    instance = ActuatorModule.__new__(ActuatorModule)
    instance.name = name
    instance.cfg = ModuleConfig(frequency=30.0, link_lost=link_lost)
    instance.interp = None
    instance.last_waypoint_time = 0.0
    instance._register_action_handlers()
    return instance


class Camera:
    """A sensor module: no actuators, so no HOLD_SAFE handler."""

    def __init__(self):
        self.name = "camera"
        self.action_handlers: dict[int, str] = {}


class FakeRobot(Robot):
    """Enough of a robot for the watchdog: no zmq, no shared memory."""

    def __init__(self, link_timeout_s: float | None):
        connection = type("Connection", (), {"link_timeout_s": link_timeout_s})()
        self.cfg = type("Cfg", (), {"connection": connection})()
        self.modules = {"arm": module("arm"), "camera": Camera()}
        self.action_queues = {"arm": FakeQueue(), "camera": FakeQueue()}
        self._local_skills = {}
        self._last_command_ns = None
        self._link_lost = False
        self.stopped_all_skills = 0

    def stop_all_skills(self, timeout: float = 5.0, wait: bool = True) -> None:
        # The watchdog must not block on a skill that is still finishing a step.
        assert wait is False
        self.stopped_all_skills += 1

    def control_loop(self):
        pass

    def disconnect(self):
        pass


@pytest.fixture(autouse=True)
def fresh_robot_singleton():
    """``Robot.__new__`` hands out one instance per process; don't keep the slot."""
    Robot._instance = None
    yield
    Robot._instance = None


def silent_for(robot: FakeRobot, seconds: float) -> None:
    robot._last_command_ns = time.monotonic_ns() - int(seconds * 1e9)


def test_silence_past_the_timeout_stops_skills_and_holds_modules_safe():
    robot = FakeRobot(link_timeout_s=1.0)
    silent_for(robot, 1.5)

    robot._check_action_link()

    assert robot.stopped_all_skills == 1
    assert robot.action_queues["arm"].action_types() == [int(ActionType.HOLD_SAFE)]


def test_a_module_with_nothing_to_hold_is_not_asked_to():
    robot = FakeRobot(link_timeout_s=1.0)
    silent_for(robot, 1.5)

    robot._check_action_link()

    assert robot.action_queues["camera"].actions == []


def test_an_idle_controller_that_still_heartbeats_is_left_alone():
    robot = FakeRobot(link_timeout_s=1.0)
    silent_for(robot, 0.2)

    robot._check_action_link()

    assert robot.stopped_all_skills == 0
    assert robot.action_queues["arm"].actions == []


def test_holding_safe_happens_once_per_outage():
    robot = FakeRobot(link_timeout_s=1.0)
    silent_for(robot, 1.5)

    for _ in range(5):
        robot._check_action_link()

    assert robot.stopped_all_skills == 1
    assert len(robot.action_queues["arm"].actions) == 1


def test_a_fresh_action_rearms_the_watchdog():
    robot = FakeRobot(link_timeout_s=1.0)
    silent_for(robot, 1.5)
    robot._check_action_link()

    robot._last_command_ns = time.monotonic_ns()
    robot._check_action_link()
    silent_for(robot, 1.5)
    robot._check_action_link()

    assert robot.stopped_all_skills == 2
    assert len(robot.action_queues["arm"].actions) == 2


def test_a_robot_nothing_has_commanded_yet_has_no_link_to_lose():
    robot = FakeRobot(link_timeout_s=1.0)

    robot._check_action_link()

    assert robot.stopped_all_skills == 0
    assert robot.action_queues["arm"].actions == []


def test_no_timeout_configured_disables_the_watchdog():
    robot = FakeRobot(link_timeout_s=None)
    silent_for(robot, 60.0)

    robot._check_action_link()

    assert robot.stopped_all_skills == 0
    assert robot.action_queues["arm"].actions == []


@pytest.fixture
def moving_gripper():
    """A module driving toward a waypoint one second out."""
    gripper = module("gripper")
    gripper.interp = ScalarTrajectoryInterpolator(
        times=np.array([0.0, 1.0]), values=np.array([0.0, 1.0])
    )
    return gripper


def test_hold_freezes_the_target_where_the_module_is_now(moving_gripper):
    moving_gripper._handle_hold_safe(ModuleAction(ActionType.HOLD_SAFE), t_now=0.25)

    assert moving_gripper.interp(0.25) == pytest.approx(0.25)
    # The waypoint it was driving toward no longer pulls it anywhere.
    assert moving_gripper.interp(1.0) == pytest.approx(0.25)


def test_relax_cuts_torque_and_drops_the_trajectory(moving_gripper):
    moving_gripper.cfg.link_lost = SafeState.RELAX
    relaxed = []
    moving_gripper.disable_torque = lambda: relaxed.append(True)

    moving_gripper._handle_hold_safe(ModuleAction(ActionType.HOLD_SAFE), t_now=0.25)

    assert relaxed == [True]
    # Dropped, so re-enabling torque re-seeds from the measured pose rather
    # than snapping back to where the arm was before it went limp.
    assert moving_gripper.interp is None
