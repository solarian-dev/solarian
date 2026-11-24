"""Unit tests for the speed-ramp helpers behind calibrate-{arm,gripper}-speed."""

from __future__ import annotations

import numpy as np
import pytest

from solarian.cli.commands._calibration import achieved_speed, ramp_to_plateau


def _move(speed: float, limit: float, distance: float, hz: float = 200.0):
    """Sample a move that tracks *speed* until it saturates at *limit*."""
    rate = min(speed, limit)
    t = np.arange(0.0, distance / rate + 0.5, 1.0 / hz)
    return t, np.clip(rate * t, 0.0, distance)


def test_achieved_speed_recovers_the_rate_of_a_clean_move():
    t, x = _move(speed=0.2, limit=1.0, distance=0.06)
    assert achieved_speed(t, x, 0.0, 0.06) == pytest.approx(0.2, rel=0.02)


def test_achieved_speed_reports_saturation_not_the_command():
    """Past the hardware limit the move takes longer than planned, which is the
    whole signal the ramp is looking for."""
    t, x = _move(speed=0.5, limit=0.2, distance=0.06)
    assert achieved_speed(t, x, 0.0, 0.06) == pytest.approx(0.2, rel=0.02)


def test_achieved_speed_resolves_a_move_only_a_few_ticks_long():
    """At 30 Hz a 53 mm window at 1000 mm/s is 1.6 ticks; rounding the
    crossings to a tick would read anything from 800 to 1600."""
    t, x = _move(speed=1.0, limit=10.0, distance=0.066, hz=30.0)
    assert achieved_speed(t + 0.013, x, 0.0, 0.066) == pytest.approx(1.0, rel=0.01)


def test_achieved_speed_handles_a_negative_direction():
    t, x = _move(speed=0.2, limit=1.0, distance=0.06)
    assert achieved_speed(t, -x, 0.0, -0.06) == pytest.approx(0.2, rel=0.02)


def test_achieved_speed_is_zero_when_the_recording_ends_mid_move():
    """The control loop runs for the *planned* duration, so an arm far slower
    than commanded is still moving when sampling stops. Reporting 0 ends the
    ramp and leaves the previous rung as the plateau."""
    t, x = _move(speed=0.2, limit=0.02, distance=0.06)
    cut = t <= 0.35
    assert achieved_speed(t[cut], x[cut], 0.0, 0.06) == 0.0


def test_ramp_stops_at_the_hardware_plateau():
    def run(speed: float) -> float:
        t, x = _move(speed, limit=0.31, distance=0.06)
        return achieved_speed(t, x, 0.0, 0.06)

    plateau, ladder, hit_ceiling = ramp_to_plateau(run, start=0.125, ceiling=1.0)
    assert plateau == pytest.approx(0.31, rel=0.05)
    assert hit_ceiling is False
    # The ladder must actually have pushed past the limit to find it.
    assert max(c for c, _ in ladder) > 0.31


def test_ramp_reports_when_the_ceiling_was_never_reached():
    """An arm that tracks everything asked of it has a limit above the test, so
    the returned figure is a floor, not the answer."""

    def run(speed: float) -> float:
        t, x = _move(speed, limit=99.0, distance=0.06)
        return achieved_speed(t, x, 0.0, 0.06)

    plateau, _, hit_ceiling = ramp_to_plateau(run, start=0.125, ceiling=1.0)
    assert hit_ceiling is True
    assert plateau == pytest.approx(1.0, rel=0.25)
