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
