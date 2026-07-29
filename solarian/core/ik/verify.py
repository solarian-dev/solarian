"""Cartesian verification for IK sim<->real alignment.

After the joint mapping is calibrated, the final question is behavioural: does a
commanded pure +X / +Y / +Z end-effector move actually translate the tool that
way, by the right amount? This module provides a pluggable
:class:`Verifier` (how ground truth is obtained) and :func:`run_cartesian_check`
(the axis-by-axis routine that commands the moves and collects results).

v1 ships :class:`HumanVerifier` (operator confirms direction, optionally measures
distance with a ruler). :class:`FiducialVerifier` is a documented seam for a
future camera/AprilTag automated check.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Protocol, Sequence

#: Human-friendly cue per axis, tied to the world triad the sim viewer draws
#: (x=red, y=green, z=blue).
AXIS_CUE = {
    "x": "along the RED (X) axis shown in the sim",
    "y": "along the GREEN (Y) axis shown in the sim",
    "z": "along the BLUE (Z) axis shown in the sim",
}


@dataclass
class AxisCheckResult:
    axis: str
    commanded_delta: float
    direction_ok: Optional[bool] = None  # None => not answered / skipped
    observed_distance: Optional[float] = None  # metres, if measured
    detail: str = ""

    @property
    def scale_ok(self) -> Optional[bool]:
        if self.observed_distance is None or self.commanded_delta == 0:
            return None
        ratio = self.observed_distance / abs(self.commanded_delta)
        return 0.8 <= ratio <= 1.2


@dataclass
class CartesianCheckReport:
    results: list[AxisCheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if not self.results:
            return False
        for r in self.results:
            if r.direction_ok is False:
                return False
            if r.scale_ok is False:
                return False
        return True

    def format_summary(self) -> str:
        lines = ["Cartesian axis check:"]
        for r in self.results:
            direction = {True: "ok", False: "WRONG", None: "?"}[r.direction_ok]
            scale = {True: "ok", False: "WRONG", None: "?"}[r.scale_ok]
            lines.append(
                f"  +{r.axis}: direction={direction}, scale={scale}"
                + (f"  ({r.detail})" if r.detail else "")
            )
        lines.append("RESULT: " + ("PASS" if self.passed else "FAIL / INCOMPLETE"))
        return "\n".join(lines)

    def hints(self) -> list[str]:
        out: list[str] = []
        if any(r.direction_ok is False for r in self.results):
            out.append(
                "A wrong direction usually means a joint's sign (drive_mode) is off -- "
                "re-run the sign step in the live mirror."
            )
        if any(r.scale_ok is False for r in self.results):
            out.append(
                "A wrong distance usually means the model jnt_range does not match the "
                "real mechanical range -- run `derive-model-limits` and update the MJCF."
            )
        return out


class Verifier(Protocol):
    """How the truth of an end-effector move is obtained."""

    def confirm_axis_move(self, axis: str, delta: float) -> AxisCheckResult:
        """Return the operator/sensor's assessment of a just-commanded +``delta`` move."""
        ...


class HumanVerifier:
    """Prompts the operator to confirm direction and (optionally) measure distance."""

    def confirm_axis_move(self, axis: str, delta: float) -> AxisCheckResult:
        cue = AXIS_CUE.get(axis, f"along +{axis}")
        cm = delta * 100.0
        print(f"\nThe tool should have moved ~{cm:.1f} cm {cue}.")
        # Re-prompt on empty/invalid input so a stray ENTER is never silently
        # recorded as a wrong direction.
        while True:
            ans = input("Did it move in the correct direction? [y/n/skip]: ").strip().lower()
            if ans and ans[0] in ("y", "n", "s"):
                break
            print("Please answer 'y', 'n', or 'skip'.")
        if ans.startswith("s"):
            return AxisCheckResult(axis, delta, None, None, "skipped")
        direction_ok = ans.startswith("y")
        measured = input("Optional: measured distance in cm (ENTER to skip): ").strip()
        observed = None
        if measured:
            try:
                observed = float(measured) / 100.0
            except ValueError:
                observed = None
        return AxisCheckResult(axis, delta, direction_ok, observed, "" if direction_ok else "reported wrong direction")


class FiducialVerifier:
    """Seam for a future camera/AprilTag automated verifier (not implemented in v1)."""

    def confirm_axis_move(self, axis: str, delta: float) -> AxisCheckResult:  # pragma: no cover
        raise NotImplementedError(
            "Camera/fiducial verification is not implemented yet; use HumanVerifier."
        )


def run_cartesian_check(
    read_pose: Callable[[], Dict[str, float]],
    command_pose: Callable[[Dict[str, float]], None],
    verifier: Verifier,
    *,
    axes: Sequence[str] = ("x", "y", "z"),
    delta: float = 0.05,
    settle_ticks: int = 60,
    tick_period: float = 1.0 / 60.0,
    return_to_start: bool = True,
) -> CartesianCheckReport:
    """Command a pure +``delta`` move on each axis and collect the verifier's result.

    ``read_pose`` returns the current EE pose ``{'x','y','z',...}`` (metres);
    ``command_pose`` writes one control tick toward the given pose target. The
    routine ramps to ``target`` over ``settle_ticks``, asks the verifier, then
    optionally returns to the start pose.
    """
    report = CartesianCheckReport()
    for axis in axes:
        start = dict(read_pose())
        target = dict(start)
        target[axis] = start.get(axis, 0.0) + delta
        for _ in range(settle_ticks):
            command_pose(target)
            time.sleep(tick_period)
        report.results.append(verifier.confirm_axis_move(axis, delta))
        if return_to_start:
            for _ in range(settle_ticks):
                command_pose(start)
                time.sleep(tick_period)
    return report
