"""Measure the aperture speed the gripper can actually sustain.

Fills ``max_speed`` in the gripper's module yaml, which
:class:`~solarian.utils.trajectory_interpolator.ScalarTrajectoryInterpolator`
enforces. That limit does not slow the jaws down — it moves a waypoint's arrival
time later — so a value below what the hardware can do manufactures lag, and one
above it turns into tracking error nobody measures. Neither is safe to guess.
"""

from __future__ import annotations

import time

from solarian.cli import BaseCommand, CommandRegistry
from solarian.cli.commands._calibration import (
    SETTLE_S,
    achieved_speed,
    module_running,
    pick_module,
    ramp_to_plateau,
    run_control_loop,
)
from solarian.core.gripper_module import APERTURE_UNIT, GripperModule

# Stay off the hard stops: the last tenth of travel at speed is a collision with
# the end of the rail, and its deceleration is not a speed measurement.
_STROKE_MARGIN = 0.1
# Ladder bounds as multiples of the configured max_speed. Starting below it
# proves the rig measures a known-good speed before anything is pushed.
_RAMP_START = 0.5
_RAMP_CEILING = 4.0


def _configured_speed(module: GripperModule) -> float:
    max_speed = module.cfg.max_speed
    if isinstance(max_speed, (int, float)):
        return float(max_speed)
    return float(max_speed[0])


def _drive(module: GripperModule, target: float, speed: float):
    """One full move at *speed*, returning ``(start, t, x)``."""
    start = module.read_aperture()
    module.moveJ(target, speed=speed)
    # The plan is built at the commanded speed even when the jaws cannot manage
    # it, so stopping at the planned end would cut the move short.
    duration = module.last_waypoint_time - time.monotonic() + SETTLE_S
    t, x = run_control_loop(module, duration, record=module.read_aperture)
    return start, t, x


def _run(module: GripperModule) -> None:
    lo, hi = module.scalar_limits()
    stroke = hi - lo
    near = lo + _STROKE_MARGIN * stroke
    far = hi - _STROKE_MARGIN * stroke
    configured = _configured_speed(module)
    units = APERTURE_UNIT

    print(
        f"\nSweeping {near:.2f} to {far:.2f} {units} at "
        f"rising speed. Configured max_speed is {configured:.2f} {units}/s."
    )

    def attempt(speed: float) -> float:
        # Return to the near end at the configured speed: only the outbound
        # stroke is being measured.
        _drive(module, near, min(speed, configured))
        start, t, x = _drive(module, far, speed)
        return achieved_speed(t, x, start, far)

    plateau, _, hit_ceiling = ramp_to_plateau(
        attempt, _RAMP_START * configured, _RAMP_CEILING * configured
    )

    print(f"\n=== gripper max_speed ({units}/s) ===")
    print(f"  configured:  {configured:.2f}")
    if hit_ceiling:
        print(
            f"  measured:    kept up to {plateau:.2f}, the ceiling this command "
            f"will command ({_RAMP_CEILING:g}x configured)."
        )
        print(
            "\nThe jaws tracked everything asked of them, so their real limit is "
            "above this. Raise max_speed toward it only as far as a move you "
            "actually need — the number's job is to describe the hardware, and "
            "there is nothing to gain from headroom you never use."
        )
        return

    print(f"  measured:    {plateau:.2f} sustained before falling behind")
    print(f"\n  max_speed: {plateau:.2f}")
    ratio = plateau / configured
    if ratio < 0.9:
        print(
            f"\nThe configured {configured:.2f} is optimistic: the jaws only hold "
            f"{plateau:.2f}. Every move planned at the configured speed arrives "
            "late by the difference, and nothing in the pipeline reports it."
        )
    elif ratio > 1.5:
        print(
            f"\nThe configured {configured:.2f} is leaving {plateau - configured:.2f} "
            f"{units}/s on the table. Grip and release are being stretched to fit "
            "a limit the hardware does not have."
        )
    else:
        print(f"\nThe configured {configured:.2f} is a fair description of this gripper.")


@CommandRegistry.register("calibrate-gripper-speed", "robot")
class CalibrateGripperSpeedCommand(BaseCommand):
    help = "Measure the aperture speed the gripper sustains, for max_speed."

    def handle(self, cfg) -> None:
        del cfg
        module = pick_module("gripper", base_cls=GripperModule)
        if module is None:
            return

        with module_running(
            module,
            "The jaws will stroke open and closed repeatedly, at speeds above "
            "the configured limit. Keep clear and keep a hand on the power.",
        ):
            _run(module)
