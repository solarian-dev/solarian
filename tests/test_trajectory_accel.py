"""Acceleration-limited go-to moves: the UR-style moveL / moveJ / stopL profile."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from solarian.core.action import ActionType
from solarian.core.arm_module import ArmModule
from solarian.core.configs import InterpolatorType
from solarian.core.gripper_module import GripperModule
from solarian.utils.trajectory_interpolator import (
    JointTrajectoryInterpolator,
    PoseTrajectoryInterpolator,
    ScalarTrajectoryInterpolator,
    _trapezoid,
)

_T0 = 100.0
_TICK = 1.0 / 90.0


def _at_rest(pose=None) -> PoseTrajectoryInterpolator:
    pose = np.zeros(6) if pose is None else np.asarray(pose, dtype=float)
    return PoseTrajectoryInterpolator(np.array([_T0]), pose[None])


def _sampled_x(interp, t_end):
    t = np.arange(_T0, t_end + _TICK, _TICK)
    return t, interp(t)[:, 0]


@pytest.mark.parametrize("dist", [0.15, 0.002])
def test_trapezoid_is_monotone_and_ends_on_the_target(dist):
    taus, s = _trapezoid(dist, v_max=0.1, a_max=0.5)
    assert taus[0] == 0.0
    assert np.all(np.diff(taus) > 0)
    assert np.all(np.diff(s) >= 0)
    assert s[-1] == 1.0


def test_trapezoid_short_move_peaks_below_v_max():
    """Too short to reach cruise: triangular, so it takes 2*sqrt(d/a)."""
    taus, _ = _trapezoid(0.002, v_max=0.1, a_max=0.5)
    assert taus[-1] == pytest.approx(2.0 * np.sqrt(0.002 / 0.5))


def test_move_ramps_up_cruises_and_ramps_down():
    out = _at_rest().drive_to_waypoint(
        np.array([0.15, 0, 0, 0, 0, 0]), _T0, _T0,
        max_pos_speed=0.1, max_rot_speed=0.6, max_pos_acc=0.5, max_rot_acc=3.0,
    )
    t, x = _sampled_x(out, out.times[-1])
    v = np.diff(x) / _TICK
    assert v[0] < 0.02
    assert v.max() == pytest.approx(0.1, rel=1e-3)
    assert v[-1] < 0.02
    assert np.abs(np.diff(v) / _TICK).max() < 0.5 * 1.2
    assert out(out.times[-1])[0] == pytest.approx(0.15)
    # 0.2 s up, 1.3 s cruise, 0.2 s down.
    assert out.times[-1] - _T0 == pytest.approx(1.7)


def test_without_acceleration_the_move_is_one_constant_speed_segment():
    out = _at_rest().drive_to_waypoint(
        np.array([0.15, 0, 0, 0, 0, 0]), _T0, _T0,
        max_pos_speed=0.1, max_rot_speed=0.6,
    )
    assert len(out.times) == 2
    assert out.times[-1] - _T0 == pytest.approx(1.5)


def test_retarget_mid_motion_keeps_the_current_speed():
    moving = _at_rest().drive_to_waypoint(
        np.array([0.15, 0, 0, 0, 0, 0]), _T0, _T0,
        max_pos_speed=0.1, max_rot_speed=0.6, max_pos_acc=0.5, max_rot_acc=3.0,
    )
    t = _T0 + 0.8
    further = moving.drive_to_waypoint(
        np.array([0.3, 0, 0, 0, 0, 0]), t, t,
        max_pos_speed=0.1, max_rot_speed=0.6, max_pos_acc=0.5, max_rot_acc=3.0,
    )
    assert further.velocity(t + 0.02)[0] == pytest.approx(0.1, rel=1e-3)


def test_reversing_brakes_before_heading_back():
    moving = _at_rest().drive_to_waypoint(
        np.array([0.15, 0, 0, 0, 0, 0]), _T0, _T0,
        max_pos_speed=0.1, max_rot_speed=0.6, max_pos_acc=0.5, max_rot_acc=3.0,
    )
    t = _T0 + 0.8
    back = moving.drive_to_waypoint(
        np.array([0.0, 0, 0, 0, 0, 0]), t, t,
        max_pos_speed=0.1, max_rot_speed=0.6, max_pos_acc=0.5, max_rot_acc=3.0,
    )
    samples, x = _sampled_x(back, back.times[-1])
    samples, x = samples[samples >= t], x[samples >= t]
    v = np.diff(x) / _TICK
    assert np.abs(np.diff(v) / _TICK).max() < 0.5 * 1.2
    assert back(back.times[-1])[0] == pytest.approx(0.0)


def test_stop_brakes_over_v_squared_over_2a():
    moving = _at_rest().drive_to_waypoint(
        np.array([0.15, 0, 0, 0, 0, 0]), _T0, _T0,
        max_pos_speed=0.1, max_rot_speed=0.6, max_pos_acc=0.5, max_rot_acc=3.0,
    )
    t = _T0 + 0.8
    at_stop = moving(t)[0]
    stopped = moving.stop(t, 0.5, 3.0)
    assert stopped.times[-1] - t == pytest.approx(0.2)
    assert stopped(stopped.times[-1])[0] - at_stop == pytest.approx(0.01)
    assert stopped.velocity(stopped.times[-1] + 0.05)[0] == 0.0


@pytest.mark.parametrize("t0", np.linspace(515000.0, 515001.0, 50))
def test_profiles_stay_strictly_increasing_days_into_uptime(t0):
    """On the Pi, monotonic() is ~5e5 s: a last knot a hair before the end of
    the profile collapsed onto it and Slerp refused the trajectory."""
    moving = PoseTrajectoryInterpolator(np.array([t0]), np.zeros((1, 6))).drive_to_waypoint(
        np.array([0.15, 0, 0, 0, 0, 0.3]), t0, t0,
        max_pos_speed=0.1, max_rot_speed=0.6, max_pos_acc=0.5, max_rot_acc=3.0,
    )
    for elapsed in (0.05, 0.13, 0.8, 1.61, 1.69, 5.0):
        stopped = moving.stop(t0 + elapsed, 0.5, 3.0)
        assert np.all(np.diff(stopped.times) > 0)
        back = moving.drive_to_waypoint(
            np.zeros(6), t0 + elapsed, t0 + elapsed,
            max_pos_speed=0.1, max_rot_speed=0.6, max_pos_acc=0.5, max_rot_acc=3.0,
        )
        assert np.all(np.diff(back.times) > 0)


def test_stop_at_rest_holds():
    held = _at_rest([0.1, 0, 0, 0, 0, 0]).stop(_T0 + 1.0, 0.5, 3.0)
    assert held(_T0 + 2.0)[0] == pytest.approx(0.1)


def test_rotation_follows_the_same_profile():
    out = _at_rest().drive_to_waypoint(
        np.array([0, 0, 0, 0, 0, 0.6]), _T0, _T0,
        max_pos_speed=0.1, max_rot_speed=0.6, max_pos_acc=0.5, max_rot_acc=3.0,
    )
    assert out(out.times[-1])[5] == pytest.approx(0.6)
    t = np.arange(_T0, out.times[-1], _TICK)
    rz = out(t)[:, 5]
    assert (np.diff(rz) / _TICK).max() == pytest.approx(0.6, rel=1e-3)


def test_joint_move_ramps_with_the_slowest_joint():
    start = JointTrajectoryInterpolator(np.array([_T0]), np.zeros((1, 3)))
    out = start.drive_to_waypoint(
        np.array([1.0, 0.5, 0.0]), _T0, _T0, max_joint_speed=1.0, max_joint_acc=2.0
    )
    assert out.times[-1] - _T0 == pytest.approx(1.5)
    np.testing.assert_allclose(out(out.times[-1]), [1.0, 0.5, 0.0])
    assert out.velocity(_T0 + 0.05)[0] < 0.2


def _pose_arm(max_accel=(0.5, 3.0)) -> ArmModule:
    module = ArmModule.__new__(ArmModule)
    module.name = "test_arm"
    module.frequency = 90.0
    module.cfg = SimpleNamespace(
        interpolator_type=InterpolatorType.POSE,
        max_speed=[0.1, 0.6],
        max_accel=None if max_accel is None else list(max_accel),
        max_joint_accel=None,
        max_joint_speed=None,
    )
    module.interp = None
    module.last_waypoint_time = 0.0
    module._joint_interp = None
    module._exclusive_move = False
    module._read_current_pose = lambda: np.zeros(6, dtype=np.float64)
    return module


def test_move_l_uses_the_configured_acceleration():
    module = _pose_arm()
    module.moveL(np.array([0.15, 0, 0, 0, 0, 0]), t_now=_T0)
    assert module.last_waypoint_time - _T0 == pytest.approx(1.7, abs=1e-3)
    assert module._exclusive_move is True


def test_infinite_acceleration_keeps_the_constant_speed_profile():
    """calibrate-arm measures speed off the middle of a move: no ramp allowed."""
    module = _pose_arm()
    module.moveL(
        np.array([0.15, 0, 0, 0, 0, 0]),
        acceleration=np.inf, rot_acceleration=np.inf, t_now=_T0,
    )
    assert module.last_waypoint_time - _T0 == pytest.approx(1.5, abs=1e-3)


def test_move_l_params_carry_speed_and_accel():
    module = _pose_arm(max_accel=None)
    captured = {}
    module.moveL = lambda pose, **kw: captured.update(kw)
    module._cartesian_pose_names = lambda: ["x", "y", "z", "rx", "ry", "rz"]
    action = SimpleNamespace(
        values=lambda: {
            n: SimpleNamespace(value=0.0, timestamp=0.0)
            for n in ("x", "y", "z", "rx", "ry", "rz")
        },
        params={"speed": 0.08, "accel": 0.4, "rot_speed": 0.5, "rot_accel": 2.0},
    )
    module._handle_move_l(action, t_now=_T0)
    assert captured["speed"] == 0.08
    assert captured["acceleration"] == 0.4
    assert captured["rot_speed"] == 0.5
    assert captured["rot_acceleration"] == 2.0


def test_stop_l_decelerates_an_arm_in_motion():
    module = _pose_arm()
    module.moveL(np.array([0.15, 0, 0, 0, 0, 0]), t_now=_T0)
    t = _T0 + 0.8
    before = module.interp(t)[0]
    module._handle_stop_l(SimpleNamespace(params={}), t_now=t)
    assert module.last_waypoint_time - t == pytest.approx(0.2)
    assert module.interp(module.last_waypoint_time)[0] - before == pytest.approx(0.01)


def test_stop_l_without_acceleration_holds_at_once():
    module = _pose_arm(max_accel=None)
    module.moveL(np.array([0.15, 0, 0, 0, 0, 0]), t_now=_T0)
    t = _T0 + 0.5
    before = module.interp(t)[0]
    module.stopL(t_now=t)
    assert module.interp(t + 1.0)[0] == pytest.approx(before)


def test_stop_j_on_the_arm_holds_the_joint_target():
    module = _pose_arm()
    module._joint_interp = JointTrajectoryInterpolator(
        np.array([_T0, _T0 + 1.0]), np.array([[0.0], [1.0]])
    )
    module._handle_stop_j(SimpleNamespace(params={}), t_now=_T0 + 0.5)
    assert module._joint_interp(_T0 + 1.0)[0] == pytest.approx(0.5)


def test_stop_j_on_the_gripper_holds_the_aperture():
    module = GripperModule.__new__(GripperModule)
    module.interp = ScalarTrajectoryInterpolator(
        np.array([_T0, _T0 + 1.0]), np.array([0.0, 80.0])
    )
    module.last_waypoint_time = _T0 + 1.0
    module._handle_stop_j(SimpleNamespace(params={}), t_now=_T0 + 0.25)
    assert float(module.interp(_T0 + 1.0)) == pytest.approx(20.0)
    assert module.last_waypoint_time == _T0 + 0.25


def test_stop_actions_coalesce_with_moves():
    """A key-up STOP_L drained behind its MOVE_L is what the operator meant."""
    module = _pose_arm()
    move = SimpleNamespace(action_type=ActionType.MOVE_L)
    stop = SimpleNamespace(action_type=ActionType.STOP_L)
    assert module._coalesce_actions([move, stop]) == [stop]
