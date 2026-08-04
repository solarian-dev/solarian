"""Pure helpers for generic joint (servo) calibration.

Used by :meth:`Module.calibrate` so hard-stop joints only need a range sweep:
``drive_mode`` comes from a low-command-end pose, and wrap-unsafe / glitch-
expanded ranges are refused before writing hardware calibration.

Homing offset stays **0** (raw encoder = Present). Recentering to mid-encoder
is unnecessary for limited-travel joints and is what forced awkward horn
remounts when travel sat near 0/CPR. Vendor-specific register limits remain on
the actuator bus when a bus *does* choose to apply a non-zero offset.
"""

from __future__ import annotations

from dataclasses import dataclass

# Raw sweep spans at or above this fraction of CPR usually mean the operator
# crossed the encoder wrap (or a glitch hit the opposite side of the circle).
MAX_SPAN_FRACTION_OF_CPR = 0.45


class CalibrationSafetyError(ValueError):
    """Raised when a recorded range / derived homing would be unsafe to write."""


@dataclass(frozen=True)
class DerivedHoming:
    """Calibration window in Present frame (with ``homing_offset`` applied)."""

    homing_offset: int
    range_min: int
    range_max: int
    mid_raw: int


@dataclass
class EncoderRangeTracker:
    """Online min/max tracker that ignores wrap-glitch samples.

    A single bad read near the opposite side of the circle (e.g. 0 while the
    joint sits at ~3900) would otherwise expand min/max to the full CPR and
    abort calibration. Samples that would push the span over
    ``max_span_fraction * cpr`` are skipped; true mechanical travel that
    crosses the wrap still fails at :meth:`result`.
    """

    cpr: int
    max_span_fraction: float = MAX_SPAN_FRACTION_OF_CPR
    raw_min: int | None = None
    raw_max: int | None = None
    accepted: int = 0
    rejected: int = 0
    last: int | None = None

    def update(self, pos: int) -> bool:
        pos = int(pos)
        if self.raw_min is None:
            self.raw_min = self.raw_max = pos
            self.last = pos
            self.accepted = 1
            return True

        trial_min = min(self.raw_min, pos)
        trial_max = max(self.raw_max, pos)
        if (trial_max - trial_min) >= self.max_span_fraction * self.cpr:
            self.rejected += 1
            return False

        self.raw_min = trial_min
        self.raw_max = trial_max
        self.last = pos
        self.accepted += 1
        return True

    def result(self) -> tuple[int, int]:
        if self.raw_min is None or self.raw_max is None:
            raise CalibrationSafetyError("No encoder samples recorded.")
        validate_contiguous_raw_range(
            "joint",
            self.raw_min,
            self.raw_max,
            self.cpr,
            max_span_fraction=self.max_span_fraction,
        )
        return self.raw_min, self.raw_max


def half_turn_count(cpr: int) -> int:
    """Encoder count treated as mid-turn for half-turn-style homing."""
    if cpr < 2:
        raise CalibrationSafetyError(f"Invalid counts_per_revolution={cpr}")
    return (cpr - 1) // 2


def validate_contiguous_raw_range(
    name: str,
    raw_min: int,
    raw_max: int,
    cpr: int,
    *,
    max_span_fraction: float = MAX_SPAN_FRACTION_OF_CPR,
) -> None:
    """Refuse raw sweeps that look wrap-straddling or out of encoder bounds."""
    if raw_max <= raw_min:
        raise CalibrationSafetyError(
            f"{name}: invalid raw range {raw_min}..{raw_max} (max must be > min)."
        )
    if raw_min < 0 or raw_max > cpr - 1:
        raise CalibrationSafetyError(
            f"{name}: raw range {raw_min}..{raw_max} outside encoder window "
            f"[0, {cpr - 1}]."
        )
    span = raw_max - raw_min
    limit = max_span_fraction * cpr
    if span >= limit:
        raise CalibrationSafetyError(
            f"{name}: raw span {span} (>= {limit:.0f}, {max_span_fraction:.0%} of "
            f"CPR={cpr}) looks like the sweep crossed the encoder wrap (or a "
            "glitch hit the far side of the circle). Move only through the "
            "mechanical travel; do not force past hard stops. "
            "Hardware calibration was not modified."
        )


def calibration_from_raw_range(
    name: str,
    raw_min: int,
    raw_max: int,
    cpr: int,
    *,
    max_span_fraction: float = MAX_SPAN_FRACTION_OF_CPR,
) -> DerivedHoming:
    """Build calibration for a contiguous raw sweep with ``homing_offset=0``.

    Limited-travel joints keep the raw encoder as Present so travel may sit
    anywhere in the turn (including near 0 or CPR) without remounting. The
    wrap risk is handled by refusing non-contiguous / over-wide spans.
    """
    validate_contiguous_raw_range(
        name, raw_min, raw_max, cpr, max_span_fraction=max_span_fraction
    )
    mid_raw = int(round((raw_min + raw_max) / 2.0))
    return DerivedHoming(
        homing_offset=0,
        range_min=int(raw_min),
        range_max=int(raw_max),
        mid_raw=mid_raw,
    )


# Back-compat alias used by older call sites / tests.
def derive_homing_from_raw_range(
    name: str,
    raw_min: int,
    raw_max: int,
    cpr: int,
    *,
    max_span_fraction: float = MAX_SPAN_FRACTION_OF_CPR,
    max_homing_abs: int | None = None,
) -> DerivedHoming:
    """Deprecated name for :func:`calibration_from_raw_range` (offset always 0).

    ``max_homing_abs`` is ignored; kept so callers that still pass it keep working.
    """
    del max_homing_abs  # no longer used when homing_offset stays 0
    return calibration_from_raw_range(
        name, raw_min, raw_max, cpr, max_span_fraction=max_span_fraction
    )


def drive_mode_from_low_end_pose(pose: float, range_min: int, range_max: int) -> int:
    """Map a low-command-end pose to ``drive_mode``.

    If the pose is closer to ``range_min``, low command = low encoder → 0.
    If closer to ``range_max``, low command = high encoder → 1 (flip).
    """
    if abs(pose - range_min) <= abs(pose - range_max):
        return 0
    return 1


def low_command_label(mode: str) -> str:
    """Human label for the low end of the actuator's normalized command range."""
    if mode == "range_0_100":
        return "0 (e.g. fully closed for a gripper)"
    if mode == "range_m100_100":
        return "-100"
    return "the low end of the command range"
