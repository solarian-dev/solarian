"""ArmModule dual-space waypoints, exclusive params-driven moves, and the pose
API the calibrate-arm-* commands drive."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from solarian.core.action import ActionType, ModuleAction
from solarian.core.arm_module import ArmModule
from solarian.core.configs import InterpolatorType
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


def test_a_missing_drop_timestamps_honours_the_stamp():
    """``drop_timestamps=None`` is not a drive waypoint, so a burst is kept."""
    arm, _ = _arm()
    burst = [_Action(POSE, drop_timestamps=None) for _ in range(4)]
    assert arm._coalesce_actions(burst) == burst


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


def _bare_arm() -> ArmModule:
    """An ArmModule with just enough state for the pose interpolator path."""
    module = ArmModule.__new__(ArmModule)
    module.name = "test_arm"
    module.frequency = 90.0
    module.cfg = SimpleNamespace(
        interpolator_type=InterpolatorType.POSE,
        max_speed=[0.25, 0.6],
    )
    module.interp = None
    module.last_waypoint_time = 0.0
    module._joint_interp = None
    module._read_current_pose = lambda: np.zeros(6, dtype=np.float64)
    return module


def test_read_pose_delegates_to_current():
    module = ArmModule.__new__(ArmModule)
    module._read_current_pose = lambda: np.arange(6, dtype=np.float64)
    assert module.read_pose().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_schedule_pose_never_drops_timestamps():
    module = ArmModule.__new__(ArmModule)
    captured: dict = {}

    module._cancel_joint_motion = lambda: captured.setdefault("cancelled", True)
    module._schedule_pose_waypoint = (
        lambda pose, target_time_wall, t_now, drop_timestamps=False: captured.update(
            pose=pose, wall=target_time_wall, drop=drop_timestamps
        )
    )

    module.schedule_pose([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], 123.0)
    assert captured["cancelled"] is True
    assert captured["wall"] == 123.0
    assert captured["drop"] is False


class _StubBackend:
    """One joint driving +z at 0.1 m/rad, clamped like mink's configuration limit."""

    def __init__(self, limit_rad: float):
        self._limit = limit_rad
        self._q = np.zeros(1)

    def set_q(self, q) -> None:
        self._q = np.clip(np.asarray(q, dtype=float), -self._limit, self._limit)

    def solve(self, target_pose, dt: float):
        return np.array([0.5 * (target_pose[2] - 0.1 * self._q[0]) / 0.1 / dt])

    def integrate(self, v, dt: float):
        return self._q + np.asarray(v, dtype=float) * dt

    def get_pose(self) -> np.ndarray:
        pose = np.zeros(6)
        pose[2] = 0.1 * self._q[0]
        return pose


def _ik_arm(limit_rad: float) -> ArmModule:
    backend = _StubBackend(limit_rad)
    module = ArmModule.__new__(ArmModule)
    module.frequency = 90.0
    module._resolve_ik_backend = lambda: backend
    module._read_joints_rad = lambda: np.zeros(1)
    module._ik_joint_names = lambda: ["j"]
    module._joint_limits = lambda: {"j": (-limit_rad, limit_rad)}
    return module


def test_reachable_travel_reports_the_full_amplitude_when_it_fits():
    module = _ik_arm(limit_rad=1.0)
    travel, blocked = module.reachable_travel(np.zeros(6), 2, 0.02)
    assert travel == pytest.approx(0.02, abs=1e-4)
    assert blocked == []


def test_reachable_travel_converges_short_against_a_joint_limit():
    """Differential IK clamps rather than failing, so a too-large sweep is only
    visible as travel that falls short of what was asked."""
    module = _ik_arm(limit_rad=0.15)
    travel, blocked = module.reachable_travel(np.zeros(6), 2, 0.02)
    assert travel == pytest.approx(0.015, abs=1e-4)
    assert blocked == ["j"]


def test_reachable_travel_handles_the_negative_direction():
    module = _ik_arm(limit_rad=0.15)
    travel, blocked = module.reachable_travel(np.zeros(6), 2, -0.02)
    assert travel == pytest.approx(-0.015, abs=1e-4)
    assert blocked == ["j"]


def test_hold_measured_pose_writes_back_what_it_read():
    """Goal position survives torque-off, so re-energising snaps the arm unless
    the measured configuration is latched first."""
    module = ArmModule.__new__(ArmModule)
    written: dict = {}
    module._read_joints_rad = lambda: np.array([0.1, -0.2, 0.3])
    module._write_joints_rad = lambda q: written.update(q=np.asarray(q))

    module.hold_measured_pose()
    assert written["q"].tolist() == [0.1, -0.2, 0.3]


def test_schedule_pose_honours_its_timestamp():
    """A calibration sweep lands where it asked, not on the adaptive drive horizon."""
    module = _bare_arm()
    lead = 0.5

    t_before = time.monotonic()
    module.schedule_pose(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]), time.time() + lead)
    t_after = time.monotonic()

    assert module.last_waypoint_time - t_before == pytest.approx(lead, abs=0.05)
    assert module.last_waypoint_time > t_after


def test_ik_tcp_survives_bind():
    from solarian.core.configs import ModuleConfig
    from solarian.core.ik import FrameOffset, MinkIK

    offset = FrameOffset(pos=(0.0, 0.0, 0.2), quat=(1.0, 0.0, 0.0, 0.0))

    class _Arm(ArmModule):
        ik_backend = MinkIK(mjcf="unused.xml", joints={"j": "j"}, frame="flange")

    placeholder = _Arm(ik_tcp=offset)
    assert placeholder.ik_backend.tcp_offset == offset
    bound = placeholder.bind(ModuleConfig(frequency=10), "right_arm")
    assert bound.ik_tcp == offset
    assert bound.ik_backend.tcp_offset == offset
    assert bound.ik_backend.frame == "flange"
