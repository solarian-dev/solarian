"""ArmModule dual-space waypoints and exclusive params-driven moves."""

from types import SimpleNamespace

import numpy as np
import pytest

from solarian.core.action import ActionType, ModuleAction
from solarian.core.arm_module import ArmModule
from solarian.core.pose import compose_ee_delta


class _Field:
    def __init__(self, value, timestamp=1.0):
        self.value = value
        self.timestamp = timestamp


class _Action:
    def __init__(
        self,
        data,
        params=None,
        drop_timestamps=True,
        action_type=ActionType.SCHEDULE_WAYPOINT,
    ):
        self._data = data
        self.params = params or {}
        self.drop_timestamps = drop_timestamps
        self.action_type = action_type

    def values(self):
        return self._data


_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
]

def _arm():
    arm = ArmModule.__new__(ArmModule)
    arm.name = "right_arm"
    arm.default_joint_speed = 1.05
    arm.frequency = 90.0
    arm.interp = None
    arm._joint_interp = None
    arm.last_waypoint_time = 0.0
    arm.cfg = SimpleNamespace(
        drive_waypoints=False,
        interpolator_type="pose",
    )
    captured = {}
    arm.servoL = lambda pose, duration=0.1, t_now=None: captured.update(
        servo="L", pose=np.asarray(pose), duration=duration, t_now=t_now
    )
    arm.servoJ = lambda q, duration=0.1, t_now=None: captured.update(
        servo="J", q=np.asarray(q), duration=duration, t_now=t_now
    )
    arm.moveJ = lambda q, speed=1.05, t_now=None, **kw: captured.update(
        move="J", q=np.asarray(q), speed=speed, t_now=t_now
    )
    arm.moveL = lambda pose, speed=0.25, t_now=None, rot_speed=None, **kw: captured.update(
        move="L", pose=np.asarray(pose), speed=speed, rot_speed=rot_speed, t_now=t_now
    )
    arm._adaptive_drive_horizon = lambda t_now: 0.1
    arm._norm_to_rad = lambda name, value: float(value)
    arm._cartesian_pose_names = lambda: ["x", "y", "z", "rx", "ry", "rz"]
    arm._ik_joint_names = lambda: list(_JOINT_NAMES)
    return arm, captured


POSE = {n: _Field(float(i), 10.0) for i, n in enumerate(["x", "y", "z", "rx", "ry", "rz"])}
JOINTS = {
    n: _Field(float(i + 1), 10.0)
    for i, n in enumerate(
        ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]
    )
}


def test_schedule_waypoint_pose_calls_servo_l():
    arm, captured = _arm()
    arm._handle_schedule_waypoint(_Action(POSE), t_now=1.0)
    assert captured["servo"] == "L"
    np.testing.assert_allclose(captured["pose"], [0, 1, 2, 3, 4, 5])


def test_schedule_waypoint_joints_calls_servo_j():
    arm, captured = _arm()
    arm._handle_schedule_waypoint(_Action(JOINTS), t_now=1.0)
    assert captured["servo"] == "J"
    np.testing.assert_allclose(captured["q"], [1, 2, 3, 4, 5, 6])


def test_schedule_waypoint_prefers_pose_when_both_present():
    arm, captured = _arm()
    both = dict(POSE)
    both.update(JOINTS)
    arm._handle_schedule_waypoint(_Action(both), t_now=1.0)
    assert captured["servo"] == "L"


def test_the_module_answers_to_no_pose_name():
    """A pose name is a skill's vocabulary; the module only takes values."""
    arm, captured = _arm()
    arm._handle_schedule_waypoint(_Action({}, params={"named": "home"}), t_now=2.0)
    assert captured == {}


def test_an_exclusive_move_with_no_speed_uses_the_module_default():
    arm, captured = _arm()
    arm._handle_schedule_waypoint(_Action(JOINTS, params={"speed": None}), t_now=2.0)
    assert captured["speed"] == pytest.approx(arm.default_joint_speed)


def test_speed_params_joints_call_move_j():
    arm, captured = _arm()
    arm._handle_schedule_waypoint(_Action(JOINTS, params={"speed": None}), t_now=2.0)
    assert captured["move"] == "J"
    np.testing.assert_allclose(captured["q"], [1, 2, 3, 4, 5, 6])


def test_speed_params_pose_world_call_move_l():
    arm, captured = _arm()
    arm._handle_schedule_waypoint(
        _Action(POSE, params={"frame": "world", "speed": None}), t_now=2.0
    )
    assert captured["move"] == "L"
    np.testing.assert_allclose(captured["pose"], [0, 1, 2, 3, 4, 5])


def test_speed_params_pose_ee_composes_then_move_l():
    arm, captured = _arm()
    current = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    arm._read_current_pose = lambda: current
    delta = {
        "x": _Field(0.1),
        "y": _Field(0.0),
        "z": _Field(0.0),
        "rx": _Field(0.0),
        "ry": _Field(0.0),
        "rz": _Field(0.0),
    }
    arm._handle_schedule_waypoint(
        _Action(delta, params={"frame": "ee", "speed": None}), t_now=2.0
    )
    assert captured["move"] == "L"
    expected = compose_ee_delta(current, np.array([0.1, 0, 0, 0, 0, 0]))
    np.testing.assert_allclose(captured["pose"], expected)


def test_bare_params_still_stream():
    """``frame`` alone is not a timing hand-off; it stays on the servo path."""
    arm, captured = _arm()
    arm._handle_schedule_waypoint(_Action(POSE, params={"frame": "world"}), t_now=1.0)
    assert captured["servo"] == "L"
    assert "move" not in captured


def test_a_policy_chunk_survives_the_drain():
    """Timestamped waypoints each name their own moment, so all are kept."""
    arm, _ = _arm()
    chunk = [_Action(POSE, drop_timestamps=False) for _ in range(8)]
    assert arm._coalesce_actions(chunk) == chunk


def test_a_burst_of_drive_waypoints_collapses_to_the_last():
    arm, _ = _arm()
    burst = [_Action(POSE, drop_timestamps=True) for _ in range(8)]
    assert arm._coalesce_actions(burst) == [burst[-1]]


def test_a_chunk_outlives_the_drive_waypoints_drained_with_it():
    arm, _ = _arm()
    chunk = [_Action(POSE, drop_timestamps=False) for _ in range(3)]
    stale = [_Action(POSE, drop_timestamps=True) for _ in range(2)]
    drained = [stale[0], chunk[0], chunk[1], stale[1], chunk[2]]
    assert arm._coalesce_actions(drained) == [chunk[0], chunk[1], stale[1], chunk[2]]


def test_the_config_decides_when_the_waypoint_does_not():
    """``drop_timestamps=None`` defers to ``drive_waypoints``, coalescing too."""
    arm, _ = _arm()
    arm.cfg.drive_waypoints = True
    burst = [_Action(POSE, drop_timestamps=None) for _ in range(4)]
    assert arm._coalesce_actions(burst) == [burst[-1]]


def test_module_action_params_roundtrip():
    action = ModuleAction(
        action_type=ActionType.SCHEDULE_WAYPOINT,
        data={},
        params={"frame": "ee", "speed": 0.5},
    )
    dumped = action.to_dict()
    restored = ModuleAction(
        action_type=dumped["action_type"],
        data=dumped["data"],
        drop_timestamps=dumped["drop_timestamps"],
        params=dumped["params"],
    )
    assert restored.params["frame"] == "ee"
    assert restored.params["speed"] == pytest.approx(0.5)
