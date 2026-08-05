"""Pure helpers for generic joint (servo) calibration.

Used by :meth:`Module.calibrate` so hard-stop joints only need a range sweep:
``drive_mode`` comes from a low-command-end pose. The recorded min/max *are*
the limits — there is no artificial cap on how large a fraction of a turn the
travel may be.

Homing offset stays **0** (raw encoder = Present). Recentering to mid-encoder
is unnecessary for limited-travel joints and is what forced awkward horn
remounts when travel sat near 0/CPR. Vendor-specific register limits remain on
the actuator bus when a bus *does* choose to apply a non-zero offset.

Glitch / wrap handling during the sweep:
* Reject a sample whose shortest path from the previous reading is a sudden
  teleport (larger than ``max_step_fraction`` of CPR).
* Reject a sample whose shortest path *crosses* the encoder wrap (0/CPR).
  Those need a horn remount so mechanical travel is contiguous in raw space;
  linear ``range_min``/``range_max`` cannot represent a wrap-straddling arc.
"""

from __future__ import annotations

from dataclasses import dataclass

# Max circular step between successive reads treated as continuous motion.
# Larger jumps are dropped as encoder glitches / teleports.
MAX_STEP_FRACTION_OF_CPR = 0.10

# Back-compat alias — span caps are no longer enforced. Kept so older imports
# / tests that reference the name keep resolving.
MAX_SPAN_FRACTION_OF_CPR = 1.0


class CalibrationSafetyError(ValueError):
    """Raised when a recorded range / derived homing would be unsafe to write."""


@dataclass(frozen=True)
class DerivedHoming:
    """Calibration window in Present frame (with ``homing_offset`` applied)."""

    homing_offset: int
    range_min: int
    range_max: int
    mid_raw: int


def circular_distance(a: int, b: int, cpr: int) -> int:
    """Shortest distance on the encoder circle between ``a`` and ``b``."""
    fwd = (int(b) - int(a)) % cpr
    bak = (int(a) - int(b)) % cpr
    return min(fwd, bak)


def shortest_path_crosses_wrap(a: int, b: int, cpr: int) -> bool:
    """True if the shortest arc from ``a`` to ``b`` crosses the 0/CPR boundary."""
    a, b = int(a), int(b)
    fwd = (b - a) % cpr
    bak = (a - b) % cpr
    if fwd <= bak:
        # Forward along increasing encoder counts (mod CPR).
        return b < a
    # Backward along decreasing counts.
    return b > a


@dataclass
class EncoderRangeTracker:
    """Online min/max tracker that ignores glitches and wrap-crossing steps.

    The operator's sweep defines the limits: any contiguous travel is accepted,
    no matter how large a fraction of a turn it is. Samples that teleport or
    whose shortest path crosses the encoder wrap are skipped (and counted in
    ``rejected`` / ``wrap_rejected``) so a single bad read cannot expand the
    window across the whole circle.
    """

    cpr: int
    max_step_fraction: float = MAX_STEP_FRACTION_OF_CPR
    # Deprecated kw-only accept for older callers; ignored.
    max_span_fraction: float = MAX_SPAN_FRACTION_OF_CPR
    raw_min: int | None = None
    raw_max: int | None = None
    accepted: int = 0
    rejected: int = 0
    wrap_rejected: int = 0
    last: int | None = None

    def update(self, pos: int) -> bool:
        pos = int(pos)
        if self.raw_min is None:
            self.raw_min = self.raw_max = pos
            self.last = pos
            self.accepted = 1
            return True

        assert self.last is not None
        step = circular_distance(self.last, pos, self.cpr)
        max_step = self.max_step_fraction * self.cpr
        if step > max_step:
            self.rejected += 1
            return False

        if shortest_path_crosses_wrap(self.last, pos, self.cpr):
            self.rejected += 1
            self.wrap_rejected += 1
            return False

        self.raw_min = min(self.raw_min, pos)
        self.raw_max = max(self.raw_max, pos)
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
    max_span_fraction: float | None = None,
) -> None:
    """Refuse inverted / out-of-bounds raw ranges.

    ``max_span_fraction`` is accepted for back-compat and ignored — the sweep
    defines the limits; there is no span-vs-CPR cap.
    """
    del max_span_fraction  # no longer enforced
    if raw_max <= raw_min:
        raise CalibrationSafetyError(
            f"{name}: invalid raw range {raw_min}..{raw_max} (max must be > min)."
        )
    if raw_min < 0 or raw_max > cpr - 1:
        raise CalibrationSafetyError(
            f"{name}: raw range {raw_min}..{raw_max} outside encoder window "
            f"[0, {cpr - 1}]."
        )


def calibration_from_raw_range(
    name: str,
    raw_min: int,
    raw_max: int,
    cpr: int,
    *,
    max_span_fraction: float | None = None,
) -> DerivedHoming:
    """Build calibration for a contiguous raw sweep with ``homing_offset=0``.

    Limited-travel joints keep the raw encoder as Present so travel may sit
    anywhere in the turn (including near 0 or CPR) without remounting.
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
    max_span_fraction: float | None = None,
    max_homing_abs: int | None = None,
) -> DerivedHoming:
    """Deprecated name for :func:`calibration_from_raw_range` (offset always 0).

    ``max_homing_abs`` / ``max_span_fraction`` are ignored; kept so callers that
    still pass them keep working.
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
