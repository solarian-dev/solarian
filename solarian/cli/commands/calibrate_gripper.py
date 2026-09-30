"""Measure how the gripper behaves in time, and write it to the module yaml.

Three numbers, in the order they depend on each other:

``max_speed``
    The aperture speed the jaws sustain, which
    :class:`~solarian.utils.trajectory_interpolator.ScalarTrajectoryInterpolator`
    enforces by pushing a waypoint's arrival time later rather than by slowing
    anything down. Under the truth it manufactures lag; over it, tracking error
    nobody measures.
``action_latency``
    Command to motion, by cross-correlating a commanded sinusoid against the
    measured aperture. Measured second because the sweep has to stay inside
    ``max_speed`` — past it the interpolator stretches every waypoint and the
    result is the speed cap, not the pipeline.
``receive_latency``
    How long a read takes, which is how far before the stamp the value was
    physically true. ``Module.update_observations`` subtracts it at the point
    of stamping, so every consumer gets the corrected instant.

Ports UMI ``calibrate_gripper_latency.py`` onto any
:class:`~solarian.core.gripper_module.GripperModule`, with the speed ramp UMI
had no equivalent for. See /docs/latency.
"""

from __future__ import annotations

import time

import numpy as np

from solarian.cli import BaseCommand, CommandRegistry
from solarian.cli.commands._calibration import (
    SETTLE_S,
    achieved_speed,
    cfg_float,
    configured_speed,
    measure_read_latency,
    module_running,
    pick_module,
    plot_latency,
    prompt_choice,
    ramp_to_plateau,
    report_read_latency,
    run_control_loop,
    save_module_field,
)
from solarian.core.gripper_module import APERTURE_UNIT, GripperModule
from solarian.utils.latency import get_latency

_DURATION_S = 10.0
_SAMPLE_HZ = 100.0
_OBS_ITERATIONS = 200
# Hertz. Cross-correlation of a narrowband signal has a broad peak that the
# finite window biases toward zero lag, and the bias falls as 1/f^2; same
# reasoning and same value as calibrate-arm.
_FREQ_HZ = 1.25
# Half-stroke as a fraction of travel, keeping the sweep off both ends where
# the jaws would clip and the sinusoid would come back flattened.
_AMPLITUDE_FRACTION = 0.25
# Refuse above this fraction of max_speed: a stretched waypoint arrives late
# and the measurement would report the speed limit, not the pipeline delay.
# UMI swept the full stroke at ~1 Hz, which peaks at 300 mm/s -- fine for the
# WSG50 they ran, twice what a 150 mm/s gripper can follow.
_SPEED_MARGIN = 0.8
# What the default sweep aims for, left under the refusal threshold so a run
# with no arguments is never borderline.
_SPEED_TARGET = 0.6
# Stay off the hard stops: the last tenth of travel at speed is a collision
# with the end of the rail, and its deceleration is not a speed measurement.
_STROKE_MARGIN = 0.1
# Ladder bounds as multiples of the configured max_speed. Starting below it
# proves the rig measures a known-good speed before anything is pushed.
_RAMP_START = 0.5
_RAMP_CEILING = 4.0


def _default_amplitude(module: GripperModule, freq: float) -> float:
    """Half-stroke that fits the travel and the speed cap, whichever binds."""
    lo, hi = module.scalar_limits()
    return min(
        _AMPLITUDE_FRACTION * (hi - lo),
        _SPEED_TARGET * configured_speed(module) / (2.0 * np.pi * freq),
    )


def _within_speed_limit(module: GripperModule, amplitude: float, freq: float) -> bool:
    limit = configured_speed(module)
    peak_speed = 2.0 * np.pi * freq * amplitude
    if peak_speed <= _SPEED_MARGIN * limit:
        return True
    print(
        f"\nSweep peaks at {peak_speed:.1f} {APERTURE_UNIT}/s but this gripper's "
        f"max_speed is {limit:.1f} {APERTURE_UNIT}/s. The interpolator would "
        "stretch every waypoint, so the jaws would crawl and the measurement "
        "would report the speed limit rather than the command pipeline.\n"
        f"Measure the speed first (mode 2), or re-run with amplitude="
        f"{_SPEED_MARGIN * limit / (2.0 * np.pi * freq):.1f} or freq="
        f"{_SPEED_MARGIN * limit / (2.0 * np.pi * amplitude):.2f}."
    )
    return False


def _drive(module: GripperModule, target: float, speed: float):
    """One full move at *speed*, returning ``(start, t, x)``."""
    start = module.read_aperture()
    module.moveJ(target, speed=speed)
    # The plan is built at the commanded speed even when the jaws cannot manage
    # it, so stopping at the planned end would cut the move short.
    duration = module.last_waypoint_time - time.monotonic() + SETTLE_S
    t, x = run_control_loop(module, duration, record=module.read_aperture)
    return start, t, x


def _run_speed(module: GripperModule) -> None:
    lo, hi = module.scalar_limits()
    stroke = hi - lo
    near = lo + _STROKE_MARGIN * stroke
    far = hi - _STROKE_MARGIN * stroke
    configured = configured_speed(module)

    print(
        f"\nSweeping {near:.2f} to {far:.2f} {APERTURE_UNIT} at rising speed. "
        f"Configured max_speed is {configured:.2f} {APERTURE_UNIT}/s."
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

    print(f"\n=== gripper max_speed ({APERTURE_UNIT}/s) ===")
    print(f"  configured:  {configured:.2f}")
    if hit_ceiling:
        print(
            f"  measured:    kept up to {plateau:.2f}, the ceiling this command "
            f"will command ({_RAMP_CEILING:g}x configured)."
        )
        print(
            "\nThe jaws tracked everything asked of them, so their real limit is "
            "above this and the figure is a floor, not a measurement. Not "
            "written. Raise max_speed by hand toward a move you actually need — "
            "the number's job is to describe the hardware, and there is nothing "
            "to gain from headroom you never use."
        )
        return

    print(f"  measured:    {plateau:.2f} sustained before falling behind")
    ratio = plateau / configured
    if ratio < 0.9:
        print(
            f"\nThe configured {configured:.2f} was optimistic: the jaws only "
            f"hold {plateau:.2f}. Every move planned at the old speed arrived "
            "late by the difference, and nothing in the pipeline reported it."
        )
    elif ratio > 1.5:
        print(
            f"\nThe configured {configured:.2f} was leaving "
            f"{plateau - configured:.2f} {APERTURE_UNIT}/s on the table. Grip "
            "and release were being stretched to fit a limit the hardware does "
            "not have."
        )
    save_module_field(module, "max_speed", round(plateau, 2))


def _run_action_latency(
    module: GripperModule, amplitude: float, freq: float
) -> None:
    lo, hi = module.scalar_limits()
    sample_dt = 1.0 / _SAMPLE_HZ
    k = int(_DURATION_S / sample_dt)
    sample_t = np.linspace(0.0, _DURATION_S, k)
    # Centred rather than spanning the stroke: an end stop clips the sinusoid,
    # and a clipped peak is a flat that cross-correlation reads as lag.
    width = 0.5 * (lo + hi) + amplitude * np.sin(2.0 * np.pi * freq * sample_t)

    module.scheduleJ(width[0], time.time() + 0.3)
    run_control_loop(module, 1.0)

    # Match UMI: settle, then stamp the sinusoid 1 s into the future.
    timestamps = time.time() + sample_t + 1.0
    for i in range(k):
        module.scheduleJ(width[i], timestamps[i])

    actual_t, actual_x = run_control_loop(
        module, _DURATION_S + 1.0, record=module.read_aperture
    )
    if actual_t.size < 2:
        print("Not enough samples to estimate latency.")
        return

    latency, info = get_latency(
        x_target=width,
        t_target=timestamps,
        x_actual=actual_x,
        t_actual=actual_t,
    )
    print(f"\n=== gripper action_latency (s) ===")
    print(f"  command-to-motion: {latency:.4f}")
    if latency < 0.0:
        # The jaws cannot move before the command that moved them, so this is
        # not a small number to round off: the sweep did not track, and the
        # correlation locked onto whatever else the window contained.
        print(
            "\nNegative, which is not physical, so nothing was written. The "
            "jaws did not follow the sweep — check that nothing is obstructing "
            "them and that max_speed matches the hardware (mode 2 measures "
            "it). The plot shows what was commanded against what moved."
        )
    else:
        print(
            "\nAction latency moves a target time earlier: send the command "
            "sooner so the jaws arrive on time. It is the mirror of "
            "receive_latency, not the same number."
        )
        save_module_field(module, "action_latency", round(latency, 4))

    plot_latency(
        t_target=timestamps,
        x_target=width,
        t_actual=actual_t,
        x_actual=actual_x,
        latency=latency,
        info=info,
        ylabel=f"aperture ({APERTURE_UNIT})",
    )


def _run_obs_latency(module: GripperModule) -> None:
    durations = measure_read_latency(module.read_aperture, _OBS_ITERATIONS)
    print("\n=== gripper receive_latency (s) ===")
    save_module_field(module, "receive_latency", round(report_read_latency(durations), 6))


@CommandRegistry.register("calibrate-gripper", "robot")
class CalibrateGripperCommand(BaseCommand):
    category = "Robot"
    help = "Measure the gripper's max_speed and its two latencies, into the module yaml."
    description = """
    Pick a gripper module, then a mode:

      all     speed, then action latency, then observation latency
      speed   strokes the jaws at rising speeds until they fall behind
              -> max_speed
      action  the jaws follow an aperture sinusoid and the measured aperture is
              cross-correlated against the command -> action_latency
      obs     repeated aperture reads are timed -> receive_latency

    Each result is written to conf/<module>.yaml as it is measured. A figure
    that is not a measurement is not written: a speed the jaws never fell
    behind is only a lower bound, and a negative latency means the sweep did
    not track.

    Run `all` on new hardware, because the order matters. The sinusoid has to
    stay inside max_speed or the interpolator stretches every waypoint and the
    latency comes back as the speed cap, so the speed is measured first and the
    sweep is sized from it.

    The jaws move in every mode but obs, at speeds above the configured limit
    during the ramp: keep clear and keep a hand on the power.
    """
    params = {
        "amplitude": "Half-stroke of the action sweep in mm (default: fits max_speed).",
        "freq": "Sweep frequency in Hz (default: 1.25).",
    }
    examples = (
        "solarian calibrate-gripper",
        "solarian calibrate-gripper amplitude=8 freq=1.0",
    )

    def handle(self, cfg) -> None:
        module = pick_module("gripper", base_cls=GripperModule)
        if module is None:
            return

        print("\n=== Calibrate gripper ===")
        print("1. all     — speed, then action latency, then obs latency")
        print("2. speed   — aperture speed ramp        → max_speed")
        print("3. action  — command-to-motion delay    → action_latency")
        print("4. obs     — sensor read duration       → receive_latency")
        mode_idx = prompt_choice("Select mode", 4)

        freq = cfg_float(cfg, "freq", _FREQ_HZ)
        explicit_amplitude = cfg_float(cfg, "amplitude", 0.0)

        if mode_idx == 4:
            with module_running(
                module, "The gripper will be read repeatedly but not commanded."
            ):
                _run_obs_latency(module)
            return

        if mode_idx in (1, 2):
            warning = (
                "The jaws will stroke open and closed repeatedly, at speeds "
                "above the configured limit.\nKeep clear and keep a hand on "
                "the power."
            )
        else:
            # Sizeable up front here, unlike `all`, because no ramp is about
            # to change the max_speed it is derived from.
            sweep = explicit_amplitude or _default_amplitude(module, freq)
            warning = (
                f"The jaws will sweep +/-{sweep:.1f} {APERTURE_UNIT} at "
                f"{freq:.2f} Hz around mid-stroke. Keep clear of the workspace."
            )
        with module_running(module, warning):
            if mode_idx in (1, 2):
                _run_speed(module)
            if mode_idx == 2:
                return
            # Sized after the ramp, so `all` uses the speed just measured
            # rather than the one that was in the yaml when the run started.
            amplitude = explicit_amplitude or _default_amplitude(module, freq)
            if _within_speed_limit(module, amplitude, freq):
                _run_action_latency(module, amplitude, freq)
            if mode_idx == 1:
                _run_obs_latency(module)
