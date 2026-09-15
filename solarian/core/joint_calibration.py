"""Pure helpers for generic joint (servo) calibration.

Used by :meth:`Module.calibrate` so hard-stop joints only need a range sweep.
The recorded min/max *are* the limits — there is no artificial cap on how
large a fraction of a turn the travel may be.

A contiguous arc, including one that sits right next to 0/CPR without crossing,
keeps ``center_offset=0`` (raw = Present). An arc that *crosses* the seam is
rejected with :class:`SeamCrossingError`: the servo cannot be driven across the
seam, so the caller has to recover by remounting the horn at mid-travel.

Glitch handling during the sweep:
* Reject a sample whose shortest path from the previous reading is a sudden
  teleport (larger than ``max_step_fraction`` of CPR).
* A single wrap-looking sample is ignored at result time (glitch). Several
  samples on the far side of the seam commit the unwrap.
"""

from __future__ import annotations

from dataclasses import dataclass

# Max circular step between successive reads treated as continuous motion.
# Larger jumps are dropped as encoder glitches / teleports.
MAX_STEP_FRACTION_OF_CPR = 0.10

# Samples accepted on the far side of a wrap before the unwrap is committed.
# One isolated 4095→0 read stays a glitch.
MIN_WRAP_CONFIRM_SAMPLES = 3

# Back-compat alias — span caps are no longer enforced. Kept so older imports
# / tests that reference the name keep resolving.
MAX_SPAN_FRACTION_OF_CPR = 1.0


class CalibrationSafetyError(ValueError):
    """Raised when a recorded range / derived offset would be unsafe to write."""


class SeamCrossingError(CalibrationSafetyError):
    """Travel straddles the 0/CPR encoder seam, which the servo cannot be driven across.

    No offset fixes this. The servo's position loop computes ``Goal - Present`` in
    raw counts with no modulo, so a goal on the far side of the seam drives the
    wrong way into a hard stop. Offsets (software ``center_offset`` or hardware
    ``Homing_Offset``) are translations, which move the discontinuity without
    removing it. Only rotating the horn takes the seam out of the travel.
    """

    def __init__(self, name: str, span: int, mid_raw: int, cpr: int):
        self.name, self.span, self.mid_raw, self.cpr = name, span, mid_raw, cpr
        super().__init__(
            f"{name}: travel spans {span} counts across the 0/{cpr - 1} encoder seam "
            f"(mid-travel reads {mid_raw}, wanted near {(cpr - 1) // 2})."
        )


@dataclass(frozen=True)
class DerivedHoming:
    """Calibration window in Present frame (with ``center_offset`` applied)."""

    center_offset: int
    range_min: int
    range_max: int
    mid_raw: int

    @property
    def homing_offset(self) -> int:
        """Alias for :attr:`center_offset` (MotorCalibration field name)."""
        return self.center_offset


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
        return b < a
    return b > a


def _unwrapped_delta(last: int, pos: int, cpr: int) -> int:
    """Signed shortest-path step from ``last`` to ``pos`` on the encoder circle."""
    fwd = (int(pos) - int(last)) % cpr
    bak = (int(last) - int(pos)) % cpr
    return fwd if fwd <= bak else -bak


@dataclass
class EncoderRangeTracker:
    """Online tracker that unwraps a wrapping sweep and ignores teleports.

    Linear travel (including sitting next to 0/CPR without crossing) is stored
    as raw min/max. Small wrap-crossing steps update an unwrapped interval;
    :meth:`derived` uses that interval only after
    :data:`MIN_WRAP_CONFIRM_SAMPLES` land on the far side of the seam.
    """

    cpr: int
    max_step_fraction: float = MAX_STEP_FRACTION_OF_CPR
    max_span_fraction: float = MAX_SPAN_FRACTION_OF_CPR
    raw_min: int | None = None
    raw_max: int | None = None
    accepted: int = 0
    rejected: int = 0
    wrap_rejected: int = 0
    wrap_crossings: int = 0
    samples_since_wrap: int = 0
    last: int | None = None
    last_unwrapped: int | None = None
    u_lo: int | None = None
    u_hi: int | None = None

    @property
    def wrap_committed(self) -> bool:
        return (
            self.wrap_crossings >= 1
            and self.samples_since_wrap >= MIN_WRAP_CONFIRM_SAMPLES
        )

    def update(self, pos: int) -> bool:
        pos = int(pos)
        if self.raw_min is None:
            self.raw_min = self.raw_max = pos
            self.last = pos
            self.last_unwrapped = pos
            self.u_lo = self.u_hi = pos
            self.accepted = 1
            return True

        assert self.last is not None and self.last_unwrapped is not None
        assert self.u_lo is not None and self.u_hi is not None
        step = circular_distance(self.last, pos, self.cpr)
        max_step = self.max_step_fraction * self.cpr
        if step > max_step:
            self.rejected += 1
            return False

        delta = _unwrapped_delta(self.last, pos, self.cpr)
        new_unwrapped = self.last_unwrapped + delta
        crosses = shortest_path_crosses_wrap(self.last, pos, self.cpr)

        self.last = pos
        self.last_unwrapped = new_unwrapped
        self.u_lo = min(self.u_lo, new_unwrapped)
        self.u_hi = max(self.u_hi, new_unwrapped)
        self.accepted += 1

        if crosses:
            self.wrap_crossings += 1
            self.samples_since_wrap = 1
            return True

        # After any wrap, freeze linear min/max so an unconfirmed glitch
        # cannot stretch the range across the circle.
        if self.wrap_crossings == 0:
            self.raw_min = min(self.raw_min, pos)
            self.raw_max = max(self.raw_max, pos)
        else:
            self.samples_since_wrap += 1
        return True

    def result(self) -> tuple[int, int]:
        """Linear raw min/max from non-wrap samples (unconfirmed wraps ignored)."""
        if self.raw_min is None or self.raw_max is None:
            raise CalibrationSafetyError("No encoder samples recorded.")
        validate_contiguous_raw_range(
            "joint",
            self.raw_min,
            self.raw_max,
            self.cpr,
        )
        return self.raw_min, self.raw_max

    def live_window(self) -> tuple[int, int, int] | None:
        """MIN, POS, MAX in one encoder frame for the sweep UI.

        Raw POS wraps at 0/CPR; unwrapped MIN/MAX do not. Once a wrap is seen the
        unwrapped frame is used instead, so POS stays between MIN and MAX even
        though that sweep will be rejected as seam-crossing.
        """
        if self.last is None or self.raw_min is None or self.raw_max is None:
            return None
        if self.wrap_crossings and self.u_lo is not None and self.last_unwrapped is not None:
            return int(self.u_lo), int(self.last_unwrapped), int(self.u_hi)
        return int(self.raw_min), int(self.last), int(self.raw_max)

    def derived(self, name: str) -> DerivedHoming:
        """Calibration for this sweep, or :class:`SeamCrossingError` if it wrapped."""
        if self.raw_min is None or self.u_lo is None or self.u_hi is None:
            raise CalibrationSafetyError("No encoder samples recorded.")
        if self.wrap_committed:
            return calibration_from_circular_arc(name, self.u_lo, self.u_hi, self.cpr)
        if self.wrap_crossings:
            self.wrap_rejected += self.wrap_crossings
        return calibration_from_raw_range(name, self.raw_min, self.raw_max, self.cpr)


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
    """Build calibration for a contiguous raw sweep with ``center_offset=0``."""
    validate_contiguous_raw_range(
        name, raw_min, raw_max, cpr, max_span_fraction=max_span_fraction
    )
    mid_raw = int(round((raw_min + raw_max) / 2.0))
    return DerivedHoming(
        center_offset=0,
        range_min=int(raw_min),
        range_max=int(raw_max),
        mid_raw=mid_raw,
    )


def calibration_from_circular_arc(name: str, u_lo: int, u_hi: int, cpr: int) -> DerivedHoming:
    """Validate an unwrapped hard-stop arc, or reject it as seam-crossing.

    ``u_lo``/``u_hi`` are unwrapped encoder ticks and may lie outside ``[0, cpr)``.
    An arc that stayed inside the window is a normal range; one that left it
    straddles the seam and is only fixable by remounting the horn, so it raises
    :class:`SeamCrossingError`.
    """
    span = int(u_hi) - int(u_lo)
    if span <= 0:
        raise CalibrationSafetyError(
            f"{name}: empty unwrapped arc {u_lo}..{u_hi}."
        )
    if span >= cpr:
        raise CalibrationSafetyError(
            f"{name}: unwrapped span {span} covers a full turn; "
            "hard-stop joints cannot be continuous. Declare is_continuous=True "
            "or remount the horn."
        )
    if u_lo >= 0 and u_hi <= cpr - 1:
        return calibration_from_raw_range(name, u_lo, u_hi, cpr)

    mid_raw = int(round((int(u_lo) + int(u_hi)) / 2.0)) % cpr
    raise SeamCrossingError(name, span, mid_raw, cpr)


def derive_homing_from_raw_range(
    name: str,
    raw_min: int,
    raw_max: int,
    cpr: int,
    *,
    max_span_fraction: float | None = None,
    max_homing_abs: int | None = None,
) -> DerivedHoming:
    """Deprecated name for :func:`calibration_from_raw_range` (offset always 0)."""
    del max_homing_abs
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
