"""Cross-correlation latency between a commanded trajectory and a measured one.

Ported from UMI ``umi.common.latency_util`` so calibration commands (and later
camera / arm scripts) can recover a single delay without depending on that
stack.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
import scipy.signal as ss


def regular_sample(x, t, t_samples):
    """Linearly resample ``x(t)`` onto ``t_samples`` (hold endpoints outside)."""
    spline = interp1d(x=t, y=x, bounds_error=False, fill_value=(x[0], x[-1]))
    return spline(t_samples)


def get_latency(
    x_target,
    t_target,
    x_actual,
    t_actual,
    t_start=None,
    t_end=None,
    resample_dt=1 / 1000,
    force_positive=False,
):
    """Return the lag (seconds) that best aligns ``x_actual`` with ``x_target``.

    Both series are resampled onto a common grid, z-scored, then
    cross-correlated. A positive result means the measured signal lags the
    command (end-to-end hardware delay).
    """
    assert len(x_target) == len(t_target)
    assert len(x_actual) == len(t_actual)
    if t_start is None:
        t_start = max(t_target[0], t_actual[0])
    if t_end is None:
        t_end = min(t_target[-1], t_actual[-1])
    n_samples = int((t_end - t_start) / resample_dt)
    t_samples = np.arange(n_samples) * resample_dt + t_start
    target_samples = regular_sample(x_target, t_target, t_samples)
    actual_samples = regular_sample(x_actual, t_actual, t_samples)

    mean = np.mean(np.concatenate([target_samples, actual_samples]))
    std = np.std(np.concatenate([target_samples, actual_samples]))
    target_samples = (target_samples - mean) / std
    actual_samples = (actual_samples - mean) / std

    correlation = ss.correlate(actual_samples, target_samples)
    lags = ss.correlation_lags(len(actual_samples), len(target_samples))
    t_lags = lags * resample_dt

    if force_positive:
        latency = t_lags[np.argmax(correlation[t_lags >= 0])]
    else:
        latency = t_lags[np.argmax(correlation)]

    info = {
        "t_samples": t_samples,
        "x_target": target_samples,
        "x_actual": actual_samples,
        "correlation": correlation,
        "lags": t_lags,
    }
    return latency, info
