"""Measure the arm's action and observation latency.

The counterpart of ``calibrate-gripper-latency`` for a Cartesian arm, and the
piece UMI never shipped: their ``robot_action_latency`` is a UR5 RTDE figure
quoted in the paper, not something their scripts measure. Here the arm follows a
Cartesian sinusoid on one axis and the measured pose is cross-correlated against
the command.

Overrides: ``axis=z amplitude=0.02 freq=1.25``.
"""

from __future__ import annotations

import time

import numpy as np
from omegaconf import OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.cli.commands._calibration import (
    free_drive_pose,
    measure_read_latency,
    module_running,
    pick_module,
    plot_latency,
    prompt_choice,
    report_read_latency,
    returns_to_start,
    run_control_loop,
)
from solarian.core.arm_module import ArmModule
from solarian.utils.latency import get_latency

_AXES = ["x", "y", "z"]
_DURATION_S = 10.0
_SAMPLE_HZ = 100.0
_OBS_ITERATIONS = 200
# Metres and hertz. Cross-correlation of a narrowband signal has a broad peak
# that the finite window biases toward zero lag; the bias falls as 1/f^2, and is
# ~18 ms at 0.5 Hz but ~2 ms at 1.25 Hz over a 10 s sweep. 2 cm at 1.25 Hz peaks
# at 0.16 m/s, inside the 0.25 m/s so106 arms run.
_AMPLITUDE_M = 0.02
_FREQ_HZ = 1.25
# Refuse above this fraction of cfg.max_speed: a stretched waypoint arrives late
# and the measurement would report the speed limit, not the pipeline delay.
_SPEED_MARGIN = 0.8
# Accept a start pose only if the model can still travel this fraction of the
# amplitude both ways; below it the sinusoid comes back clipped.
_REACH_MARGIN = 0.9


def _cfg_float(cfg, key: str, default: float) -> float:
    val = OmegaConf.select(cfg, key)
    return default if val is None else float(val)


def _max_pos_speed(module: ArmModule) -> float:
    """The translation speed limit `PoseTrajectoryInterpolator` enforces (m/s)."""
    max_speed = module.cfg.max_speed
    if isinstance(max_speed, (int, float)):
        return float(max_speed)
    return float(max_speed[0])


def _within_speed_limit(module: ArmModule, amplitude: float, freq: float) -> bool:
    limit = _max_pos_speed(module)
    peak_speed = 2.0 * np.pi * freq * amplitude
    if peak_speed <= _SPEED_MARGIN * limit:
        return True
    print(
        f"\nSinusoid peaks at {peak_speed:.3f} m/s but this arm's "
        f"max_speed is {limit:.3f} m/s. The interpolator would stretch every "
        "waypoint and the measured lag would be the speed limit, not the "
        "command pipeline.\n"
        f"Re-run with a smaller amplitude or freq, e.g. "
        f"amplitude={_SPEED_MARGIN * limit / (2.0 * np.pi * freq):.3f} "
        f"or freq={_SPEED_MARGIN * limit / (2.0 * np.pi * amplitude):.3f}."
    )
    return False


def _position_by_hand(module: ArmModule, axis: int, amplitude: float) -> np.ndarray | None:
    """Free-drive the arm, energise it where the user holds it, and vet the envelope.

    Returns the pose to centre the sweep on, or ``None`` when the arm cannot
    travel the full amplitude from there.
    """
    axis_name = _AXES[axis]
    home = free_drive_pose(
        module,
        f"Put the end-effector where the sweep should be centred, with "
        f"{amplitude * 100:.1f} cm clear either side along {axis_name}.",
    )
    print(
        f"\nCentred at {axis_name}={home[axis] * 100.0:.1f} cm, sweeping "
        f"{(home[axis] - amplitude) * 100.0:.1f} to "
        f"{(home[axis] + amplitude) * 100.0:.1f} cm."
    )

    for direction in (1.0, -1.0):
        travel, blocked = module.reachable_travel(home, axis, direction * amplitude)
        if abs(travel) >= _REACH_MARGIN * amplitude:
            continue
        sign = "+" if direction > 0 else "-"
        print(
            f"\nOnly {abs(travel) * 100.0:.1f} cm of the requested "
            f"{amplitude * 100.0:.1f} cm is reachable on {sign}{axis_name}"
            + (f", blocked by: {', '.join(blocked)}." if blocked else ".")
            + "\nThe sweep would come back clipped and the latency would measure "
            "the clip. Re-run and hold the arm further from that limit, or pass "
            f"amplitude={abs(travel) * _REACH_MARGIN:.3f}."
        )
        return None

    input("Press Enter to sweep, or Ctrl+C to abort...")
    return home


def _run_action_latency(
    module: ArmModule, axis: int, amplitude: float, freq: float, home: np.ndarray
) -> None:
    limit = _max_pos_speed(module)
    sample_dt = 1.0 / _SAMPLE_HZ
    k = int(_DURATION_S / sample_dt)
    sample_t = np.linspace(0.0, _DURATION_S, k)
    offsets = amplitude * np.sin(2.0 * np.pi * freq * sample_t)
    poses = np.tile(home, (k, 1))
    poses[:, axis] += offsets

    module.schedule_pose(poses[0], time.time() + 0.3)
    run_control_loop(module, 1.0)

    # Match UMI's gripper sweep: settle, then stamp the sinusoid 1 s ahead so
    # every waypoint is scheduled before its target time.
    timestamps = time.time() + sample_t + 1.0
    for i in range(k):
        module.schedule_pose(poses[i], timestamps[i])

    actual_t, actual_x = run_control_loop(
        module,
        _DURATION_S + 1.0,
        record=lambda: float(module.read_pose()[axis]),
    )
    if actual_t.size < 2:
        print("Not enough samples to estimate latency.")
        return

    latency, info = get_latency(
        x_target=poses[:, axis],
        t_target=timestamps,
        x_actual=actual_x,
        t_actual=actual_t,
    )
    print(f"\nCommand-to-motion latency on {_AXES[axis]}: {latency:.4f} s")
    print(f"\n  arm_action_latency: {latency:.4f}")
    print(
        "Action latency moves a target time earlier: send the command sooner so "
        "the end-effector arrives on time. It is NOT the same number as "
        "arm_obs_latency — run mode 2 for that."
    )

    module.moveL(home, speed=_SPEED_MARGIN * limit)
    run_control_loop(module, module.last_waypoint_time - time.monotonic())

    plot_latency(
        t_target=timestamps,
        x_target=poses[:, axis],
        t_actual=actual_t,
        x_actual=actual_x,
        latency=latency,
        info=info,
        ylabel=f"{_AXES[axis]} (m)",
    )


def _run_obs_latency(module: ArmModule) -> None:
    durations = measure_read_latency(module.read_pose, _OBS_ITERATIONS)
    report_read_latency(durations, "arm_obs_latency")
    print(
        "This is a serial bus round trip plus forward kinematics, so expect "
        "single-digit milliseconds — not UMI's 0.000124, which is a UR5 RTDE "
        "poll off a 500 Hz stream."
    )


@CommandRegistry.register("calibrate-arm-latency", "robot")
class CalibrateArmLatencyCommand(BaseCommand):
    help = "Measure arm action latency (Cartesian sinusoid) or observation latency (read timing)."
    params = ("axis", "amplitude", "freq")

    def handle(self, cfg) -> None:
        module = pick_module("arm", base_cls=ArmModule)
        if module is None:
            return

        print("\n=== Calibrate arm latency ===")
        print("1. action  — command-to-motion delay  → arm_action_latency")
        print("2. obs     — sensor read duration     → arm_obs_latency")
        if prompt_choice("Select mode", 2) == 2:
            with module_running(
                module, "The arm will be read repeatedly but not commanded."
            ):
                _run_obs_latency(module)
            return

        axis_name = str(OmegaConf.select(cfg, "axis") or "z").lower()
        if axis_name not in _AXES:
            print(f"axis must be one of {_AXES}, got {axis_name!r}.")
            return
        amplitude = _cfg_float(cfg, "amplitude", _AMPLITUDE_M)
        freq = _cfg_float(cfg, "freq", _FREQ_HZ)
        # Before the hardware is touched: no point free-driving the arm into
        # position only to reject the sweep it was positioned for.
        if not _within_speed_limit(module, amplitude, freq):
            return

        axis = _AXES.index(axis_name)
        with module_running(
            module,
            "Torque will be released so you can position the arm by hand, then "
            f"the arm will sweep +/-{amplitude * 100:.1f} cm on {axis_name} at "
            f"{freq:.2f} Hz around where you left it, and finally return here.\n"
            "Keep a hand on the power.",
        ):
            with returns_to_start(module, _SPEED_MARGIN * _max_pos_speed(module)):
                home = _position_by_hand(module, axis, amplitude)
                if home is not None:
                    _run_action_latency(module, axis, amplitude, freq, home)
