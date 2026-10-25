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
    def __init__(self, data, params=None, action_type=ActionType.SERVO_L):
        self._data = data
        self.params = params or {}
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
    arm.moveL = lambda pose, speed=None, t_now=None, rot_speed=None, **kw: captured.update(
        move="L", pose=np.asarray(pose), speed=speed, rot_speed=rot_speed, t_now=t_now
    )
    arm.scheduleL = lambda pose, target_time_wall, t_now=None: captured.update(
        schedule="L", pose=np.asarray(pose), wall=target_time_wall, t_now=t_now
    )
    arm.scheduleJ = lambda q, target_time_wall, t_now=None: captured.update(
        schedule="J", q=np.asarray(q), wall=target_time_wall, t_now=t_now
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


def test_servo_l_calls_servo_l():
    arm, captured = _arm()
    arm._handle_servo_l(_Action(POSE, action_type=ActionType.SERVO_L), t_now=1.0)
    assert captured["servo"] == "L"
    np.testing.assert_allclose(captured["pose"], [0, 1, 2, 3, 4, 5])


def test_servo_j_calls_servo_j():
    arm, captured = _arm()
    arm._handle_servo_j(_Action(JOINTS, action_type=ActionType.SERVO_J), t_now=1.0)
    assert captured["servo"] == "J"
    np.testing.assert_allclose(captured["q"], [1, 2, 3, 4, 5, 6])


def test_schedule_l_calls_schedule_l():
    arm, captured = _arm()
    arm._handle_schedule_l(_Action(POSE, action_type=ActionType.SCHEDULE_L), t_now=1.0)
    assert captured["schedule"] == "L"
    assert captured["wall"] == pytest.approx(10.0)


def test_schedule_j_calls_schedule_j():
    arm, captured = _arm()
    arm._handle_schedule_j(_Action(JOINTS, action_type=ActionType.SCHEDULE_J), t_now=1.0)
    assert captured["schedule"] == "J"
    np.testing.assert_allclose(captured["q"], [1, 2, 3, 4, 5, 6])


def test_the_module_answers_to_no_pose_name():
    """A pose name is a skill's vocabulary; the module only takes values."""
    arm, captured = _arm()
    arm._handle_move_l(_Action({}, params={"named": "home"}), t_now=2.0)
    assert captured == {}


def test_move_j_with_no_speed_uses_the_module_default():
    arm, captured = _arm()
    arm._handle_move_j(_Action(JOINTS, action_type=ActionType.MOVE_J), t_now=2.0)
    assert captured["speed"] == pytest.approx(arm.default_joint_speed)
    assert captured["move"] == "J"
    np.testing.assert_allclose(captured["q"], [1, 2, 3, 4, 5, 6])


def test_move_l_world_calls_move_l():
    arm, captured = _arm()
    arm._handle_move_l(
        _Action(POSE, params={"frame": "world"}, action_type=ActionType.MOVE_L),
        t_now=2.0,
    )
    assert captured["move"] == "L"
    np.testing.assert_allclose(captured["pose"], [0, 1, 2, 3, 4, 5])
    assert captured["speed"] is None


def test_move_l_ee_composes_then_move_l():
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
    arm._handle_move_l(
        _Action(delta, params={"frame": "ee"}, action_type=ActionType.MOVE_L),
        t_now=2.0,
    )
    assert captured["move"] == "L"
    expected = compose_ee_delta(current, np.array([0.1, 0, 0, 0, 0, 0]))
    np.testing.assert_allclose(captured["pose"], expected)


def test_a_policy_chunk_survives_the_drain():
    """Timestamped waypoints each name their own moment, so all are kept."""
    arm, _ = _arm()
    chunk = [_Action(POSE, action_type=ActionType.SCHEDULE_L) for _ in range(8)]
    assert arm._coalesce_actions(chunk) == chunk


def test_a_burst_of_drive_waypoints_collapses_to_the_last():
    arm, _ = _arm()
    burst = [_Action(POSE, action_type=ActionType.SERVO_L) for _ in range(8)]
    assert arm._coalesce_actions(burst) == [burst[-1]]


def test_a_chunk_outlives_the_drive_waypoints_drained_with_it():
    arm, _ = _arm()
    chunk = [_Action(POSE, action_type=ActionType.SCHEDULE_L) for _ in range(3)]
    stale = [_Action(POSE, action_type=ActionType.SERVO_L) for _ in range(2)]
    drained = [stale[0], chunk[0], chunk[1], stale[1], chunk[2]]
    assert arm._coalesce_actions(drained) == [chunk[0], chunk[1], stale[1], chunk[2]]


def test_module_action_params_roundtrip():
    action = ModuleAction(
        action_type=ActionType.MOVE_L,
        data={},
        params={"frame": "ee", "speed": 0.5},
    )
    dumped = action.to_dict()
    restored = ModuleAction(
        action_type=dumped["action_type"],
        data=dumped["data"],
        params=dumped["params"],
    )
    assert restored.params["frame"] == "ee"
    assert restored.params["speed"] == pytest.approx(0.5)
    assert "drop_timestamps" not in dumped


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
    module._exclusive_move = False
    module._cancel_joint_motion = lambda: None
    module._late_waypoint_warn_t = 0.0
    module._late_waypoint_suppressed = 0
    module._read_current_pose = lambda: np.zeros(6, dtype=np.float64)
    return module


def test_read_pose_delegates_to_current():
    module = ArmModule.__new__(ArmModule)
    module._read_current_pose = lambda: np.arange(6, dtype=np.float64)
    assert module.read_pose().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_schedule_l_never_drops_timestamps():
    module = ArmModule.__new__(ArmModule)
    captured: dict = {}

    module._cancel_joint_motion = lambda: captured.setdefault("cancelled", True)
    module._schedule_pose_waypoint = (
        lambda pose, target_time_wall, t_now: captured.update(
            pose=pose, wall=target_time_wall
        )
    )

    module.scheduleL([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], 123.0)
    assert captured["cancelled"] is True
    assert captured["wall"] == 123.0


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


def _hold_arm(reads):
    module = ArmModule.__new__(ArmModule)
    written: dict = {}
    module._ik_joint_names = lambda: ["a", "b", "c"]
    module._joint_raw_wrap_spans = lambda: np.zeros(3)
    module._read_joints_raw = lambda: next(reads)
    module._write_joints_raw = lambda q: written.update(q=np.asarray(q))
    return module, written


def test_hold_measured_pose_writes_back_what_it_read():
    """Goal position survives torque-off, so re-energising snaps the arm unless
    the measured configuration is latched first."""
    module, written = _hold_arm(iter([np.array([0.1, -0.2, 0.3])] * 2))

    assert module.hold_measured_pose().tolist() == [0.1, -0.2, 0.3]
    assert written["q"].tolist() == [0.1, -0.2, 0.3]


def test_hold_measured_pose_confirms_the_reading_before_writing():
    """Hand tremor between the two reads is fine; it is still the same pose."""
    module, written = _hold_arm(
        iter(
            [
                np.array([0.1, -0.2, 0.3]),
                np.array([0.1004, -0.2003, 0.2998]),
            ]
        )
    )

    module.hold_measured_pose()
    # The confirming read is the one written: it is the more recent truth.
    assert written["q"] == pytest.approx([0.1004, -0.2003, 0.2998])


def test_hold_measured_pose_refuses_to_latch_a_reading_it_cannot_confirm():
    """A mis-parsed sync read must not become a goal the arm is driven to.

    This is the only write with no check after it, because torque comes on
    next. wrist_roll is geared for a full turn, so a goal on the far side of
    the encoder wrap sends it all the way round rather than nowhere.
    """
    module, written = _hold_arm(
        iter([np.array([0.1, 0.0, float(i * 200)]) for i in range(10)])
    )

    with pytest.raises(RuntimeError, match="Refusing to latch"):
        module.hold_measured_pose()
    assert written == {}


def test_hold_measured_pose_retries_through_a_single_bad_read():
    """One corrupt sample should cost a retry, not the whole operation."""
    module, written = _hold_arm(
        iter(
            [
                np.array([0.1, -0.2, 0.3]),
                np.array([0.1, -0.2, 80.0]),  # garbage
                np.array([0.1, -0.2, 0.3]),
                np.array([0.1, -0.2, 0.3]),
            ]
        )
    )

    module.hold_measured_pose()
    assert written["q"] == pytest.approx([0.1, -0.2, 0.3])


def test_hold_measured_pose_treats_the_continuous_wrap_as_the_same_pose():
    """Tick 0 and tick 4095 on wrist_roll are neighbours, not a full turn."""
    from solarian.core.arm_module import _max_wrapped_abs

    assert _max_wrapped_abs(np.array([4095.0]), np.array([4096.0])) == pytest.approx(1.0)
    assert _max_wrapped_abs(np.array([2.0]), np.array([0.0])) == pytest.approx(2.0)

    module, written = _hold_arm(iter([np.array([4094.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]))
    module._joint_raw_wrap_spans = lambda: np.array([4096.0, 0.0, 0.0])

    module.hold_measured_pose()
    assert written["q"] == pytest.approx([1.0, 0.0, 0.0])


def test_enable_torque_holding_rewrites_the_latch_after_energising():
    """Some Feetech firmware ignores Goal_Position while torque is off."""
    module, written = _hold_arm(iter([np.array([100.0, 200.0, 300.0])] * 4))
    order: list[str] = []
    module.enable_torque = lambda: order.append("on")
    module._write_joints_raw = lambda q: (
        written.update(q=np.asarray(q)),
        order.append("write"),
    )

    module.enable_torque_holding()
    assert order == ["write", "on", "write"]
    assert written["q"] == pytest.approx([100.0, 200.0, 300.0])


def test_hold_measured_pose_writes_raw_ticks_not_a_converted_command():
    """drive_mode / offset / radians are how wrist_roll got a full-turn goal."""
    module = ArmModule.__new__(ArmModule)
    written: dict = {}
    module._ik_joint_names = lambda: ["wrist_roll"]
    module._joint_raw_wrap_spans = lambda: np.array([4096.0])
    module._read_joints_raw = lambda: np.array([12.0])
    module._write_joints_raw = lambda q: written.update(q=np.asarray(q))
    module._read_joints_norm = lambda: (_ for _ in ()).throw(AssertionError("no conversion"))
    module._write_joints_norm = lambda q: (_ for _ in ()).throw(AssertionError("no conversion"))
    module._read_joints_rad = lambda: (_ for _ in ()).throw(AssertionError("no conversion"))
    module._write_joints_rad = lambda q: (_ for _ in ()).throw(AssertionError("no conversion"))

    module.hold_measured_pose()
    assert written["q"].tolist() == [12.0]


def test_schedule_l_honours_its_timestamp():
    """A calibration sweep lands where it asked, not on the adaptive drive horizon."""
    module = _bare_arm()
    lead = 0.5

    t_before = time.monotonic()
    module.scheduleL(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]), time.time() + lead)
    t_after = time.monotonic()

    assert module.last_waypoint_time - t_before == pytest.approx(lead, abs=0.05)
    assert module.last_waypoint_time > t_after


def test_resolve_waypoint_time_discards_late():
    from solarian.core.actuator_module import ActuatorModule

    module = ActuatorModule.__new__(ActuatorModule)
    module.name = "right_arm"
    module._late_waypoint_warn_t = 0.0
    module._late_waypoint_suppressed = 0

    assert module._resolve_waypoint_time(2.0, 1.0) == 2.0
    assert module._resolve_waypoint_time(1.0, 1.0) is None
    assert module._resolve_waypoint_time(0.9, 1.0) is None


def test_schedule_l_discards_a_late_waypoint():
    """A past SCHEDULE_L must not be clamped into the next control period."""
    module = _bare_arm()
    t_now = time.monotonic()
    module._ensure_pose_interpolator(t_now)
    seed_t = module.last_waypoint_time
    seed_times = np.array(module.interp.times)

    module.scheduleL(
        np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]), time.time() - 0.2
    )

    assert module.last_waypoint_time == seed_t
    np.testing.assert_array_equal(module.interp.times, seed_times)


def test_replacing_unused_tail_splices_from_measured_pose():
    """Overlapping replan must not hop from an untracked commanded lookahead."""
    module = _bare_arm()
    measured = np.array([0.04, 0.0, 0.0, 0.0, 0.0, 0.0])
    module._read_current_pose = lambda: measured.copy()

    module.scheduleL(np.array([0.20, 0.0, 0.0, 0.0, 0.0, 0.0]), time.time() + 1.0)
    far = module.last_waypoint_time

    module.scheduleL(np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]), time.time() + 0.15)

    assert module.last_waypoint_time < far
    start = module.interp(module.interp.times[0])
    np.testing.assert_allclose(start[:3], measured[:3], atol=1e-5)


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


def _movej_arm(q_now: np.ndarray | None = None) -> ArmModule:
    """Arm with a real joint interpolator so exclusive-move cleanup can be tested."""
    module = ArmModule.__new__(ArmModule)
    module.name = "right_arm"
    module.default_joint_speed = 1.05
    module.interp = None
    module._joint_interp = None
    module.last_waypoint_time = 0.0
    module.cfg = SimpleNamespace(max_joint_accel=None)
    start = (
        np.zeros(6, dtype=np.float64)
        if q_now is None
        else np.asarray(q_now, dtype=np.float64)
    )
    module._read_joints_rad = lambda: start.copy()
    module._clear_pose_interpolator = lambda: None
    return module


def test_cancel_does_not_freeze_an_exclusive_move_j():
    """Reaping a policy after MOVE_J must not stop the go-to the new skill sent."""
    module = _movej_arm()
    target = np.array([0.0, -1.4, 1.2, 0.4, 0.0, 0.0])
    module.moveJ(target, speed=1.05, t_now=0.0)
    end_t = float(module._joint_interp.times[-1])
    assert end_t > 1.0

    module._handle_cancel_trajectory(
        ModuleAction(action_type=ActionType.CANCEL_TRAJECTORY), t_now=0.2
    )

    np.testing.assert_allclose(module._joint_interp(end_t), target)


def test_a_late_schedule_l_does_not_preempt_an_exclusive_move_j():
    """The dying policy's last chunk arrives after MOVE_J on the same queue."""
    module = _movej_arm()
    target = np.array([0.0, -1.4, 1.2, 0.4, 0.0, 0.0])
    module.moveJ(target, speed=1.05, t_now=0.0)
    scheduled = []
    module._cancel_joint_motion = lambda: scheduled.append("cancelled")
    module._schedule_pose_waypoint = lambda *a, **k: scheduled.append("scheduled")

    module.scheduleL(np.zeros(6), time.time() + 0.5, t_now=0.1)

    assert scheduled == []
    np.testing.assert_allclose(module._joint_interp(module._joint_interp.times[-1]), target)


def test_schedule_l_takes_over_after_exclusive_move_arrives():
    """A finished go-home must not block the next policy's SCHEDULE_L."""
    module = _movej_arm()
    target = np.array([0.0, -1.4, 1.2, 0.4, 0.0, 0.0])
    module.moveJ(target, speed=1.05, t_now=0.0)
    end_t = float(module._joint_interp.times[-1])
    scheduled = []
    module._cancel_joint_motion = lambda: scheduled.append("cancelled")
    module._schedule_pose_waypoint = lambda *a, **k: scheduled.append("scheduled")

    module.scheduleL(np.zeros(6), time.time() + 0.5, t_now=end_t + 0.01)

    assert scheduled == ["cancelled", "scheduled"]
    assert module._exclusive_move is False


def test_servo_l_does_preempt_an_exclusive_move_j():
    """Teleop is a new author and must be able to take the arm back."""
    module = _movej_arm()
    module.moveJ(np.array([0.0, -1.4, 1.2, 0.4, 0.0, 0.0]), speed=1.05, t_now=0.0)
    cancelled = []
    module._cancel_joint_motion = lambda: cancelled.append(True)
    module._ensure_pose_interpolator = lambda t_now=None: None
    module.interp = SimpleNamespace(
        drive_to_waypoint=lambda **kw: SimpleNamespace(times=np.array([kw["time"]]))
    )
    module._pose_speed_kwargs = lambda: {}
    module._min_servo_duration = lambda duration: duration

    module.servoL(np.zeros(6), duration=0.1, t_now=1.0)

    assert cancelled == [True]
    assert module._exclusive_move is False
