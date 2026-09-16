"""Ruler-measured jaw opening against normalized servo command.

A parallel gripper's opening is rarely linear in servo travel: the linkage
compresses motion at one end of the stroke. Two calibration points therefore
describe the ends correctly and everything between them wrong. ``solarian
calibrate-gripper-ranges`` sweeps the servo and records what a ruler reads at
each stop; this turns those pairs into a conversion both ways.
"""

from __future__ import annotations

import numpy as np
import scipy.interpolate as si
from pydantic import BaseModel, PrivateAttr, model_validator


#: Resolution of the resampled grid the mm -> norm direction is inverted from.
#: 2048 points over a ~100 mm travel put the inversion error below a micron,
#: far under what a ruler can resolve.
_INVERSE_SAMPLES = 2048


class ApertureTable(BaseModel):
    """Measured aperture (mm) at each normalized servo command [-100, 100].

    Interpolation is PCHIP, not a natural cubic spline: a cubic overshoots
    between knots, and an overshoot that reverses the slope would move the jaws
    the wrong way for a command falling between two measurements. PCHIP is
    shape-preserving, so a monotone table stays monotone everywhere.
    """

    #: Normalized servo command at each measured stop.
    norm: list[float]
    #: Jaw opening in mm read off the ruler at the matching :attr:`norm`.
    aperture_mm: list[float]

    _to_norm: si.PchipInterpolator = PrivateAttr()
    _to_aperture: si.PchipInterpolator = PrivateAttr()

    @model_validator(mode="after")
    def _build_interpolators(self) -> "ApertureTable":
        norm = np.asarray(self.norm, dtype=np.float64)
        mm = np.asarray(self.aperture_mm, dtype=np.float64)
        if norm.size != mm.size:
            raise ValueError(
                f"aperture_table needs one aperture per servo command, got "
                f"{norm.size} norm and {mm.size} aperture_mm"
            )
        if norm.size < 2:
            raise ValueError("aperture_table needs at least 2 measurements")
        _check_strictly_monotone(norm, "norm")
        _check_strictly_monotone(mm, "aperture_mm")

        # The sweep sets norm and measures mm, so mm as a function of norm is
        # the direction the data was gathered in and the one that gets fitted.
        self._to_aperture = _ascending_interpolator(norm, mm)
        # Fitting mm -> norm as a second, independent PCHIP would agree at the
        # knots and drift between them, so a commanded aperture would not read
        # back as itself. Resampling the fit densely and inverting that keeps
        # both directions exact inverses at O(1) cost.
        grid_norm = np.linspace(norm.min(), norm.max(), _INVERSE_SAMPLES)
        self._to_norm = _ascending_interpolator(
            self._to_aperture(grid_norm), grid_norm
        )
        return self

    @property
    def limits(self) -> tuple[float, float]:
        """Closed / open ends of the measured travel, in mm."""
        return float(min(self.aperture_mm)), float(max(self.aperture_mm))

    def to_norm(self, aperture_mm: float) -> float:
        """Normalized servo command that opens the jaws to *aperture_mm*."""
        return _evaluate(self._to_norm, aperture_mm)

    def to_aperture(self, norm: float) -> float:
        """Jaw opening in mm at normalized servo command *norm*."""
        return _evaluate(self._to_aperture, norm)


def _check_strictly_monotone(values: np.ndarray, field: str) -> None:
    steps = np.diff(values)
    if np.all(steps > 0) or np.all(steps < 0):
        return
    rising = steps[0] > 0
    bad = int(np.argmax(steps <= 0 if rising else steps >= 0))
    raise ValueError(
        f"aperture_table {field} must be strictly monotone to be invertible; "
        f"{values[bad]:g} -> {values[bad + 1]:g} at index {bad} turns back"
    )


def _ascending_interpolator(x: np.ndarray, y: np.ndarray) -> si.PchipInterpolator:
    if x[0] > x[-1]:
        x, y = x[::-1], y[::-1]
    return si.PchipInterpolator(x, y)


def _evaluate(interp: si.PchipInterpolator, value: float) -> float:
    """Evaluate *interp*, saturating instead of extrapolating past the ends.

    Extrapolating a cubic past the last knot diverges fast, so a command just
    outside the measured travel would resolve to a wildly wrong servo value.
    """
    knots = interp.x
    return float(interp(np.clip(float(value), knots[0], knots[-1])))
