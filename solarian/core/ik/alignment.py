"""Servo-agnostic math for aligning a real arm to its IK (MuJoCo) model.

IK reasons in the model's joint frame (radians). On real hardware a joint value
passes through two conversions:

* the servo bus maps raw encoder units -> normalized ``[-100, 100]`` using the
  :class:`~solarian.core.calibration.MotorCalibration`
  (``range_min``/``range_max``/``drive_mode``);
* the :class:`~solarian.core.ik.bus.IKActuatorBus` maps normalized -> radians
  using the *model* joint limits ``(lo, hi)``.

Composed, for a raw (homing-adjusted) position ``P`` in ``range_m100_100`` mode::

    rad(P) = ((P - range_min) / (range_max - range_min)) * (hi - lo) + lo

(with ``drive_mode`` reversing the direction once). So Cartesian IK tracks
correctly iff **``range_min`` corresponds to the model's ``lo`` and
``range_max`` to ``hi``** -- i.e. the model's ``jnt_range`` equals the real arm's
mechanical hard-stop angles.

This module is pure (no I/O, no servo-specific imports). The only servo-specific
input it needs is a *counts-per-revolution* value used to convert raw encoder
span to radians; callers supply it via the optional
:meth:`~solarian.core.actuator_bus.ActuatorBus.counts_per_revolution` hook, and
every function degrades gracefully when it is unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence, Union

from solarian.core.calibration import MotorCalibration

TAU = 2.0 * math.pi

#: How to resolve counts-per-revolution for a joint. Either a mapping
#: ``{joint: counts}``, a callable ``joint -> counts | None``, or ``None`` (the
#: quantitative checks then report "unknown" and callers fall back to the live
#: mirror / Cartesian verification).
CountsPerRev = Union[Mapping[str, Optional[float]], Callable[[str], Optional[float]], None]


# -- calibration field access (works for MotorCalibration or a plain dict) ----
def _field(entry, name: str, default=None):
    if entry is None:
        return default
    if isinstance(entry, Mapping):
        return entry.get(name, default)
    return getattr(entry, name, default)


def _resolve_counts(counts_per_rev: CountsPerRev, joint: str) -> Optional[float]:
    if counts_per_rev is None:
        return None
    try:
        if callable(counts_per_rev):
            value = counts_per_rev(joint)
        else:
            value = counts_per_rev.get(joint)
    except Exception:
        return None
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


# -- core conversions ---------------------------------------------------------
def raw_span_to_rad(range_min: float, range_max: float, counts_per_rev: float) -> float:
    """Angular span (radians) covered by a raw encoder range.

    ``counts_per_rev`` is the encoder resolution (raw units per full turn).
    """
    if counts_per_rev is None or counts_per_rev <= 0:
        raise ValueError("counts_per_rev must be a positive number")
    return abs(float(range_max) - float(range_min)) * TAU / float(counts_per_rev)


def composed_rad(
    raw: float,
    range_min: float,
    range_max: float,
    model_lo: float,
    model_hi: float,
    drive_mode: int = 0,
) -> float:
    """The composed real mapping ``raw -> radians`` for ``range_m100_100``.

    Mirrors ``ActuatorBus._normalize`` (raw -> normalized, applying
    ``drive_mode`` once) followed by ``IKActuatorBus._norm_to_rad`` (normalized
    -> radians, using the model limits). Used by tests to assert that a produced
    calibration lands the model limits exactly on the range endpoints.
    """
    if range_max == range_min:
        return model_lo
    base_norm = ((float(raw) - range_min) / (range_max - range_min)) * 200.0 - 100.0
    norm = -base_norm if drive_mode else base_norm
    norm = min(100.0, max(-100.0, norm))
    return ((norm + 100.0) / 200.0) * (model_hi - model_lo) + model_lo


# -- continuous joints: one home pose -> encoder window -----------------------
@dataclass(frozen=True)
class ContinuousRangeWindow:
    """Encoder ``range_min``/``range_max`` that maps a home reading to model 0.

    ``needs_rehome`` means the window does not fit in ``[0, cpr-1]`` at the
    current present reading. Re-home so that pose reads mid-encoder, then
    compute again. A model range that puts 0 at an endpoint (e.g. ``[-2π, 0]``)
    can never fit a full-turn window; the fallback is the full encoder so
    travel stays bidirectional.
    """

    range_min: int
    range_max: int
    needs_rehome: bool


def _window_if_span_kept(lo_t: float, hi_t: float, max_tick: float) -> Optional[tuple[int, int]]:
    """Clamp to ``[0, max_tick]`` only when that keeps nearly the intended span.

    A full-turn span is ``cpr`` ticks but the valid encoder is ``[0, cpr-1]``;
    clipping that one extra tick is fine. Clipping half a turn is not — that
    is how a ``[-2π, 0]`` model used to save ``0..2047`` and fail verify at 0.50x.
    """
    intended = hi_t - lo_t
    if intended <= 0:
        return None
    clo = max(lo_t, 0.0)
    chi = min(hi_t, max_tick)
    if chi <= clo or (chi - clo) / intended < 0.98:
        return None
    return int(round(clo)), int(round(chi))


def continuous_range_window(
    present: float,
    model_lo: float,
    model_hi: float,
    cpr: float,
) -> ContinuousRangeWindow:
    """Linear encoder window for a continuous joint whose present pose is model 0."""
    span_rad = float(model_hi) - float(model_lo)
    if span_rad <= 0 or cpr <= 0:
        raise ValueError("model range and counts_per_rev must be positive")
    span_ticks = span_rad / TAU * float(cpr)
    frac = (0.0 - float(model_lo)) / span_rad
    max_tick = float(cpr) - 1.0

    fitted = _window_if_span_kept(
        present - span_ticks * frac,
        present + span_ticks * (1.0 - frac),
        max_tick,
    )
    if fitted is not None:
        return ContinuousRangeWindow(fitted[0], fitted[1], False)

    mid = float(cpr) / 2.0
    mid_fitted = _window_if_span_kept(
        mid - span_ticks * frac,
        mid + span_ticks * (1.0 - frac),
        max_tick,
    )
    if mid_fitted is not None:
        return ContinuousRangeWindow(mid_fitted[0], mid_fitted[1], True)

    return ContinuousRangeWindow(0, int(max_tick), True)


# -- verify: does the calibration span match the model's joint range? ---------
@dataclass
class JointLimitReport:
    """Per-joint result of :func:`verify_model_limits`."""

    name: str
    status: str  # "ok" | "mismatch" | "continuous" | "unknown" | "no_calibration"
    model_lo: float
    model_hi: float
    measured_span_rad: Optional[float] = None
    detail: str = ""

    @property
    def model_span_rad(self) -> float:
        return self.model_hi - self.model_lo

    @property
    def ratio(self) -> Optional[float]:
        if self.measured_span_rad is None or self.model_span_rad == 0:
            return None
        return self.measured_span_rad / self.model_span_rad


@dataclass
class VerifyReport:
    """Aggregate result of :func:`verify_model_limits`."""

    joints: list[JointLimitReport] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if every joint is consistent (``ok``) or a known-continuous joint.

        ``mismatch`` / ``unknown`` / ``no_calibration`` are treated as failures
        so the operator is told to re-check (a ``continuous`` joint is expected
        to be handled by the reference-pose fallback, so it does not fail).
        """
        return all(j.status in ("ok", "continuous") for j in self.joints)

    @property
    def warnings(self) -> list[JointLimitReport]:
        return [j for j in self.joints if j.status != "ok"]

    def format_table(self) -> str:
        header = f"{'JOINT':<16} {'STATUS':<12} {'MODEL(rad)':>12} {'REAL(rad)':>12} {'RATIO':>7}  DETAIL"
        lines = [header, "-" * len(header)]
        for j in self.joints:
            model = f"{j.model_span_rad:.3f}"
            real = "n/a" if j.measured_span_rad is None else f"{j.measured_span_rad:.3f}"
            ratio = "n/a" if j.ratio is None else f"{j.ratio:.2f}x"
            lines.append(
                f"{j.name:<16} {j.status:<12} {model:>12} {real:>12} {ratio:>7}  {j.detail}"
            )
        lines.append("")
        lines.append("RESULT: " + ("PASS" if self.passed else "FAIL"))
        return "\n".join(lines)


def verify_model_limits(
    calibration: Mapping[str, object],
    model_limits: Mapping[str, tuple[float, float]],
    counts_per_rev: CountsPerRev = None,
    *,
    tol: float = 0.05,
    continuous_frac: float = 0.9,
) -> VerifyReport:
    """Compare each joint's calibrated raw span (in radians) to the model range.

    Args:
        calibration: ``{joint: MotorCalibration|dict}`` with ``range_min``/``range_max``.
        model_limits: ``{joint: (lo, hi)}`` radians from the IK backend.
        counts_per_rev: resolver for encoder counts-per-revolution (see
            :data:`CountsPerRev`); when it returns ``None`` for a joint that
            joint's status is ``unknown``.
        tol: allowed fractional deviation of ``measured/model`` from 1.0.
        continuous_frac: a calibrated span >= this fraction of a full turn while
            the model range is smaller flags a near-continuous (no hard-stop)
            joint.
    """
    report = VerifyReport()
    for joint, (lo, hi) in model_limits.items():
        entry = calibration.get(joint) if hasattr(calibration, "get") else None
        if entry is None:
            report.joints.append(
                JointLimitReport(joint, "no_calibration", lo, hi, None, "not in calibration file")
            )
            continue

        cpr = _resolve_counts(counts_per_rev, joint)
        if cpr is None:
            report.joints.append(
                JointLimitReport(joint, "unknown", lo, hi, None, "no counts/rev; use live mirror check")
            )
            continue

        span = raw_span_to_rad(_field(entry, "range_min", 0.0), _field(entry, "range_max", 0.0), cpr)
        model_span = hi - lo
        if span >= continuous_frac * TAU and model_span < continuous_frac * TAU:
            report.joints.append(
                JointLimitReport(joint, "continuous", lo, hi, span, "near-continuous; use reference-pose fallback")
            )
            continue

        ratio = span / model_span if model_span else None
        if ratio is not None and abs(ratio - 1.0) <= tol:
            report.joints.append(JointLimitReport(joint, "ok", lo, hi, span, "model matches real"))
        else:
            pct = "n/a" if ratio is None else f"{(ratio - 1.0) * 100:+.0f}%"
            report.joints.append(
                JointLimitReport(joint, "mismatch", lo, hi, span, f"real vs model off by {pct}")
            )
    return report


# -- derive: what should the model's jnt_range be, from a real sweep? ---------
def derive_model_limits(
    calibration: Mapping[str, object],
    counts_per_rev: CountsPerRev,
    *,
    symmetric: bool = True,
) -> dict[str, tuple[float, float]]:
    """Turn a real limit sweep (calibration ranges) into model ``jnt_range`` radians.

    With ``symmetric`` (default) the model range is centered at 0 -- appropriate
    when homing centered the raw range at mid-travel -- so the author can paste
    ``[-span/2, +span/2]`` into the MJCF. Joints whose counts/rev is unknown are
    skipped.
    """
    limits: dict[str, tuple[float, float]] = {}
    for joint, entry in calibration.items():
        cpr = _resolve_counts(counts_per_rev, joint)
        if cpr is None:
            continue
        span = raw_span_to_rad(_field(entry, "range_min", 0.0), _field(entry, "range_max", 0.0), cpr)
        if symmetric:
            limits[joint] = (-span / 2.0, span / 2.0)
        else:
            limits[joint] = (0.0, span)
    return limits


# -- two-pose fallback (for continuous / no-hard-stop joints) -----------------
@dataclass
class Pose:
    """A captured reference pose: raw encoder ticks at known model angles."""

    raw_ticks: Mapping[str, float]  # {joint: raw present position}
    model_angles: Mapping[str, float]  # {joint: model angle (radians)}


@dataclass
class TwoPoseResult:
    calibration: dict[str, MotorCalibration]
    warnings: list[str] = field(default_factory=list)


def calibration_from_two_poses(
    pose_a: Pose,
    pose_b: Pose,
    model_limits: Mapping[str, tuple[float, float]],
    counts_per_rev: CountsPerRev = None,
    *,
    ids: Optional[Mapping[str, int]] = None,
    homing_offsets: Optional[Mapping[str, int]] = None,
    joints: Optional[Sequence[str]] = None,
    slope_tol: float = 0.1,
) -> TwoPoseResult:
    """Solve per-joint ``MotorCalibration`` from two captured reference poses.

    For each joint the affine raw<->radian map is fixed by the two captures
    ``(P1, q1)`` and ``(P2, q2)``; ``range_min``/``range_max`` are then the raw
    ticks at the model limits and ``drive_mode`` encodes the sign. When a
    counts/rev is available the physical slope is cross-checked against it and a
    warning is emitted on a large discrepancy (backlash, a wrong pose, or the
    wrong joint moved).
    """
    result = TwoPoseResult(calibration={})
    names = list(joints) if joints is not None else list(model_limits.keys())
    for joint in names:
        if joint not in model_limits:
            continue
        try:
            p1 = float(pose_a.raw_ticks[joint])
            q1 = float(pose_a.model_angles[joint])
            p2 = float(pose_b.raw_ticks[joint])
            q2 = float(pose_b.model_angles[joint])
        except KeyError:
            result.warnings.append(f"{joint}: missing capture in one of the poses; skipped")
            continue

        if q2 == q1:
            result.warnings.append(f"{joint}: the two reference angles are equal; skipped")
            continue
        if p2 == p1:
            result.warnings.append(f"{joint}: joint did not move between poses; skipped")
            continue

        slope_r = (q2 - q1) / (p2 - p1)  # radians per raw tick (signed)
        lo, hi = model_limits[joint]
        raw_lo = p1 + (lo - q1) / slope_r
        raw_hi = p1 + (hi - q1) / slope_r
        range_min = int(round(min(raw_lo, raw_hi)))
        range_max = int(round(max(raw_lo, raw_hi)))
        drive_mode = 0 if slope_r > 0 else 1

        cpr = _resolve_counts(counts_per_rev, joint)
        if cpr is not None:
            expected = TAU / cpr  # radians per tick magnitude
            if expected > 0 and abs(abs(slope_r) - expected) / expected > slope_tol:
                result.warnings.append(
                    f"{joint}: measured {abs(slope_r):.2e} rad/tick vs expected "
                    f"{expected:.2e} (>{slope_tol:.0%}); check the poses / joint"
                )

        result.calibration[joint] = MotorCalibration(
            id=int(ids[joint]) if ids and joint in ids else 0,
            drive_mode=drive_mode,
            homing_offset=int(homing_offsets[joint]) if homing_offsets and joint in homing_offsets else 0,
            range_min=range_min,
            range_max=range_max,
        )
    return result
