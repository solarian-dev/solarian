"""Unit tests for command-vs-measurement latency via cross-correlation."""

from __future__ import annotations

import numpy as np
import pytest

from solarian.utils.latency import get_latency


def test_get_latency_recovers_known_delay():
    delay = 0.13
    t = np.linspace(0.0, 10.0, 2000)
    x = np.sin(t * 2.0 * np.pi * 0.8)
    latency, info = get_latency(x, t, x, t + delay)
    assert latency == pytest.approx(delay, abs=0.005)
    assert "correlation" in info
    assert "lags" in info


def test_get_latency_recovers_the_same_delay_on_every_pose_axis():
    """`calibrate-arm-latency` sweeps one axis of a 6-D pose at a time; each
    axis must report the arm's delay whatever its amplitude or phase.

    Sweep parameters match that command's defaults: 1.25 Hz over 10 s. A lower
    frequency widens the correlation peak enough to bias the estimate (~18 ms
    at 0.5 Hz), which is why the command does not use one.
    """
    delay = 0.08
    t = np.linspace(0.0, 10.0, 2000)
    # Metres on x/y/z, radians on rx/ry/rz.
    amplitudes = [0.02, 0.02, 0.02, 0.2, 0.2, 0.2]
    phases = np.linspace(0.0, np.pi, 6)

    for axis, (amplitude, phase) in enumerate(zip(amplitudes, phases)):
        commanded = amplitude * np.sin(2.0 * np.pi * 1.25 * t + phase)
        latency, _ = get_latency(commanded, t, commanded, t + delay)
        assert latency == pytest.approx(delay, abs=0.005), f"axis {axis}"


def test_get_latency_survives_measurement_noise():
    """The measured pose comes off a servo bus, not a simulator."""
    delay = 0.12
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 10.0, 2000)
    commanded = 0.02 * np.sin(2.0 * np.pi * 1.25 * t)
    measured = commanded + rng.normal(scale=0.001, size=t.size)
    latency, _ = get_latency(commanded, t, measured, t + delay)
    assert latency == pytest.approx(delay, abs=0.01)
