"""Waypoints that land on an instant already in the trajectory.

A move to where the module already is produces a duration that rounds to zero at
float64 resolution. `Slerp` rejects the duplicated instant outright; `interp1d`
accepts it and returns nan. Both are wrong, and both are reachable from a plain
`moveL(current_pose)`.
"""

from __future__ import annotations

import numpy as np
import pytest

from solarian.utils.trajectory_interpolator import (
    JointTrajectoryInterpolator,
    PoseTrajectoryInterpolator,
    ScalarTrajectoryInterpolator,
)

_T = 12345.678
_HOME = np.array([0.05, 0.10, 0.166, 0.1, 0.2, 0.3])


def _pose_at_home() -> PoseTrajectoryInterpolator:
    return PoseTrajectoryInterpolator(np.array([_T]), _HOME[None])


def test_calibration_sweep_ends_exactly_on_its_start():
    """Why this is not a hypothetical: a 10 s sweep at 1.25 Hz closes a whole
    number of half-cycles, so the arm finishes where it began and the return
    move has nothing to travel."""
    offset = 0.02 * np.sin(2 * np.pi * 1.25 * 10.0)
    assert abs(offset) < 1e-15


def test_pose_drive_to_the_current_pose_holds_it():
    out = _pose_at_home().drive_to_waypoint(
        _HOME, time=_T, curr_time=_T, max_pos_speed=0.2, max_rot_speed=0.6
    )
    assert out(_T) == pytest.approx(_HOME)
    assert out.times.tolist() == [_T]


def test_pose_schedule_at_the_current_instant_holds_it():
    out = _pose_at_home().schedule_waypoint(
        _HOME, time=_T, curr_time=_T, max_pos_speed=0.2, max_rot_speed=0.6
    )
    assert out(_T) == pytest.approx(_HOME)


def test_scalar_drive_to_the_current_value_holds_it():
    interp = ScalarTrajectoryInterpolator(np.array([_T]), np.array([1.5]))
    out = interp.drive_to_waypoint(1.5, time=_T, curr_time=_T, max_speed=15.0)
    assert float(out(_T)) == pytest.approx(1.5)


def test_joint_drive_to_the_current_configuration_holds_it():
    interp = JointTrajectoryInterpolator(np.array([_T]), np.zeros((1, 3)))
    out = interp.drive_to_waypoint(
        np.zeros(3), time=_T, curr_time=_T, max_joint_speed=1.0
    )
    assert out(_T) == pytest.approx(np.zeros(3))


def test_a_real_move_is_still_speed_limited():
    """The guard must not swallow moves that genuinely have somewhere to go."""
    far = _HOME + np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    out = _pose_at_home().drive_to_waypoint(
        far, time=_T, curr_time=_T, max_pos_speed=0.2, max_rot_speed=0.6
    )
    assert out.times[-1] - _T == pytest.approx(0.5)
    assert out(out.times[-1]) == pytest.approx(far)


def test_a_tiny_but_resolvable_move_still_gets_its_own_segment():
    far = _HOME + np.array([0.0, 0.0, 0.001, 0.0, 0.0, 0.0])
    out = _pose_at_home().drive_to_waypoint(
        far, time=_T, curr_time=_T, max_pos_speed=0.2, max_rot_speed=0.6
    )
    assert out.times.size == 2
    assert out(out.times[-1]) == pytest.approx(far)
