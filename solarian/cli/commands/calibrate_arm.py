"""Measure how the arm behaves in time, and write it to the module yaml.

The same three numbers as ``calibrate-gripper`` — ``max_speed``,
``action_latency``, ``receive_latency`` — with two differences that come from
the arm being a kinematic chain rather than one axis.

The speed is not a property of the hardware alone: the joint speed limit
reaches the end-effector through the Jacobian, so what is achievable in the
middle of the workspace is not achievable near full extension or close to a
singularity. A single figure is only safe as the *minimum* over the region the
arm works in, which is why several poses are sampled and the worst is kept.

The action latency is the piece UMI never shipped: their ``robot_action_latency``
is a UR5 RTDE figure quoted in the paper, not something their scripts measure.
Here the arm follows a Cartesian sinusoid on one axis and the measured pose is
cross-correlated against the command. See /docs/latency.
"""

from __future__ import annotations

import time

import numpy as np
from omegaconf import OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.cli.commands._calibration import (
    SETTLE_S,
    achieved_speed,
    cfg_float,
    configured_speed,
    free_drive_pose,
    measure_read_latency,
    module_running,
    pick_module,
    plot_latency,
    prompt_choice,
    ramp_to_plateau,
    report_read_latency,
    returns_to_start,
    run_control_loop,
    save_module_field,
)
from solarian.core.arm_module import ArmModule
from solarian.utils.latency import get_latency

_AXES = ["x", "y", "z"]
_DURATION_S = 10.0
_SAMPLE_HZ = 100.0
_OBS_ITERATIONS = 200
_DISTANCE_M = 0.06
_POSES = 3
# Metres and hertz. Cross-correlation of a narrowband signal has a broad peak
# that the finite window biases toward zero lag; the bias falls as 1/f^2, and is
# ~18 ms at 0.5 Hz but ~2 ms at 1.25 Hz over a 10 s sweep. 2 cm at 1.25 Hz peaks
# at 0.16 m/s, inside the 0.25 m/s arm1 arms run.
_AMPLITUDE_M = 0.02
_FREQ_HZ = 1.25
# Refuse above this fraction of max_speed: a stretched waypoint arrives late
# and the measurement would report the speed limit, not the pipeline delay.
_SPEED_MARGIN = 0.8
# Accept a start pose only if the model can still travel this fraction of the
# amplitude both ways; below it the sinusoid comes back clipped.
_REACH_MARGIN = 0.9
# Ladder bounds as multiples of the configured max_speed.
_RAMP_START = 0.5
_RAMP_CEILING = 4.0
# A direction is only sampled when the model can travel the whole test distance
# there; short of that the move would be clipped and read as a low speed.
_SPEED_REACH_MARGIN = 0.95


def _rot_speed(module: ArmModule) -> float:
    """The configured rotation limit, untouched: only translation is measured."""
    max_speed = module.cfg.max_speed
    if isinstance(max_speed, (int, float)):
        return float(max_speed)
    return float(max_speed[1] if len(max_speed) > 1 else max_speed[0])


def _within_speed_limit(module: ArmModule, amplitude: float, freq: float) -> bool:
    limit = configured_speed(module)
    peak_speed = 2.0 * np.pi * freq * amplitude
    if peak_speed <= _SPEED_MARGIN * limit:
        return True
    print(
        f"\nSinusoid peaks at {peak_speed:.3f} m/s but this arm's max_speed is "
        f"{limit:.3f} m/s. The interpolator would stretch every waypoint and "
        "the measured lag would be the speed limit, not the command pipeline.\n"
        f"Measure the speed first (mode 2), or re-run with amplitude="
        f"{_SPEED_MARGIN * limit / (2.0 * np.pi * freq):.3f} "
        f"or freq={_SPEED_MARGIN * limit / (2.0 * np.pi * amplitude):.3f}."
    )
    return False


# ---------------------------------------------------------------- speed


def _run_to(module: ArmModule, target, speed: float, axis: int | None = None):
    """Drive to *target* at *speed*, recording *axis* when one is given."""
    # Unlimited acceleration: a ramp inside the 10-90% window would read as
    # the actuator being slower than it is.
    module.moveL(
        target, speed=speed, acceleration=np.inf, rot_acceleration=np.inf
    )
    # Planned at the commanded speed even when the arm cannot manage it, so
    # stopping at the planned end would cut the move short.
    duration = module.last_waypoint_time - time.monotonic() + SETTLE_S
    record = None if axis is None else (lambda: float(module.read_pose()[axis]))
    return run_control_loop(module, duration, record=record)


def _travel(
    module: ArmModule, home, axis: int, distance: float, speed: float, back_at: float
) -> float:
    """Move *distance* along *axis* at *speed*, then return to *home*.

    Returning to the pose the user chose, rather than reversing the same offset,
    is what keeps the samples comparable: a move that fell short would otherwise
    leave the arm somewhere the next direction starts from.
    """
    target = np.asarray(home, dtype=np.float64).copy()
    target[axis] += distance

    t, x = _run_to(module, target, speed, axis=axis)
    got = achieved_speed(t, x, float(home[axis]), float(target[axis]))
    _run_to(module, home, back_at)
    return got


def _reachable_directions(
    module: ArmModule, pose, distance: float
) -> list[tuple[int, float]]:
    """The ``(axis, sign)`` pairs with room for the full test distance."""
    directions = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            travel, blocked = module.reachable_travel(pose, axis, sign * distance)
            if abs(travel) >= _SPEED_REACH_MARGIN * distance:
                directions.append((axis, sign))
            else:
                label = "+" if sign > 0 else "-"
                print(
                    f"  skipping {label}{_AXES[axis]}: only "
                    f"{abs(travel) * 100:.1f} cm of room"
                    + (f" ({', '.join(blocked)} at its limit)" if blocked else "")
                )
    return directions


def _run_speed(module: ArmModule, n_poses: int, distance: float) -> None:
    configured = configured_speed(module)
    print(
        f"\nConfigured max_speed is {configured:.3f} m/s. Testing "
        f"{distance * 100:.0f} cm moves at {n_poses} pose(s), every axis "
        "direction with room for them."
    )

    limit = np.inf
    worst = ""
    for pose_idx in range(1, n_poses + 1):
        pose = free_drive_pose(
            module,
            f"Pose {pose_idx} of {n_poses}: put the end-effector somewhere the arm "
            f"works, with {distance * 100:.0f} cm clear around it in every direction.",
        )
        directions = _reachable_directions(module, pose, distance)
        if not directions:
            print("  no direction has room here; reposition and re-run.")
            continue

        for axis, sign in directions:
            label = ("+" if sign > 0 else "-") + _AXES[axis]
            if limit == np.inf:
                # First direction sampled: find the plateau, then only check
                # whether the rest can hold it. A full ladder everywhere would
                # multiply the run time by the number of samples for a number
                # that is taken as a minimum anyway.
                print(f"\nPose {pose_idx}, {label} — finding the plateau:")
                plateau, _, hit_ceiling = ramp_to_plateau(
                    lambda speed, a=axis, s=sign, p=pose: _travel(
                        module, p, a, s * distance, speed, configured
                    ),
                    _RAMP_START * configured,
                    _RAMP_CEILING * configured,
                )
                if hit_ceiling:
                    print(
                        f"\nThe arm tracked every speed up to {plateau:.3f} m/s, "
                        f"the ceiling this command will command "
                        f"({_RAMP_CEILING:g}x configured). Its real limit is "
                        "above that, so this is a floor and was not written."
                    )
                    return
                limit, worst = plateau, f"pose {pose_idx} {label}"
                continue

            got = _travel(module, pose, axis, sign * distance, limit, configured)
            print(f"  pose {pose_idx} {label}: holds {got:.3f} of {limit:.3f} m/s")
            if got < limit:
                limit, worst = got, f"pose {pose_idx} {label}"

    if limit == np.inf:
        print("\nNothing was sampled.")
        return

    print("\n=== arm max_speed (m/s, translation) ===")
    print(f"  configured:  {configured:.3f}")
    print(f"  measured:    {limit:.3f}, worst at {worst}")
    print(
        "\nThis is the slowest direction sampled, which is the only figure safe "
        "to use as one global limit — the Jacobian makes the rest of the "
        "workspace slower still. Re-run over the poses a skill actually visits "
        "if it works somewhere this did not reach."
    )
    save_module_field(module, "max_speed", [round(limit, 3), _rot_speed(module)])


# -------------------------------------------------------------- latency


def _position_by_hand(
    module: ArmModule, axis: int, amplitude: float
) -> np.ndarray | None:
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
    sample_dt = 1.0 / _SAMPLE_HZ
    k = int(_DURATION_S / sample_dt)
    sample_t = np.linspace(0.0, _DURATION_S, k)
    offsets = amplitude * np.sin(2.0 * np.pi * freq * sample_t)
    poses = np.tile(home, (k, 1))
    poses[:, axis] += offsets

    module.scheduleL(poses[0], time.time() + 0.3)
    run_control_loop(module, 1.0)

    # Match UMI's gripper sweep: settle, then stamp the sinusoid 1 s ahead so
    # every waypoint is scheduled before its target time.
    timestamps = time.time() + sample_t + 1.0
    for i in range(k):
        module.scheduleL(poses[i], timestamps[i])

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
    print("\n=== arm action_latency (s) ===")
    print(f"  command-to-motion on {_AXES[axis]}: {latency:.4f}")
    if latency < 0.0:
        # The arm cannot move before the command that moved it, so this is not
        # a small number to round off: the sweep did not track, and the
        # correlation locked onto whatever else the window contained.
        print(
            "\nNegative, which is not physical, so nothing was written. The arm "
            "did not follow the sweep — check that max_speed matches the "
            "hardware (mode 2 measures it). The plot shows what was commanded "
            "against what moved."
        )
    else:
        print(
            "\nAction latency moves a target time earlier: send the command "
            "sooner so the end-effector arrives on time. It is the mirror of "
            "receive_latency, not the same number."
        )
        save_module_field(module, "action_latency", round(latency, 4))

    module.moveL(home, speed=_SPEED_MARGIN * configured_speed(module))
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
    print("\n=== arm receive_latency (s) ===")
    print(
        "This is a serial bus round trip plus forward kinematics, so expect "
        "single-digit milliseconds — not UMI's 0.000124, which is a UR5 RTDE "
        "poll off a 500 Hz stream."
    )
    save_module_field(module, "receive_latency", round(report_read_latency(durations), 6))


@CommandRegistry.register("calibrate-arm", "robot")
class CalibrateArmCommand(BaseCommand):
    category = "Robot"
    help = "Measure the arm's max_speed and its two latencies, into the module yaml."
    description = """
    Pick an arm module, then a mode:

      all     speed, then action latency, then observation latency
      speed   short moves at rising speeds from several poses, worst kept
              -> max_speed[0]
      action  the arm follows a Cartesian sinusoid on one axis and the measured
              pose is cross-correlated against the command -> action_latency
      obs     repeated pose reads are timed -> receive_latency

    Each result is written to conf/<module>.yaml as it is measured. A figure
    that is not a measurement is not written: a speed the arm never fell behind
    is only a lower bound, and a negative latency means the sweep did not track.

    Run `all` on new hardware, because the order matters. The sinusoid has to
    stay inside max_speed or the interpolator stretches every waypoint and the
    latency comes back as the speed cap, so the speed is measured first and the
    sweep is sized from it. Expect to place the arm by hand several times: once
    per speed pose, and once more to centre the sweep.

    Torque is released for each placement and the arm returns to where it
    started. Speed mode deliberately drives faster than the configured limit:
    keep the volume clear and a hand on the power.
    """
    params = {
        "axis": "Axis the action sweep runs on: x, y or z (default: z).",
        "amplitude": "Half-stroke of the action sweep in metres (default: 0.02).",
        "freq": "Sweep frequency in Hz (default: 1.25).",
        "poses": "How many poses the speed ramp samples (default: 3).",
        "distance": "Length of each speed test move in metres (default: 0.06).",
    }
    examples = (
        "solarian calibrate-arm",
        "solarian calibrate-arm poses=1 axis=y amplitude=0.03",
    )

    def handle(self, cfg) -> None:
        module = pick_module("arm", base_cls=ArmModule)
        if module is None:
            return

        print("\n=== Calibrate arm ===")
        print("1. all     — speed, then action latency, then obs latency")
        print("2. speed   — Cartesian speed ramp       → max_speed")
        print("3. action  — command-to-motion delay    → action_latency")
        print("4. obs     — sensor read duration       → receive_latency")
        mode_idx = prompt_choice("Select mode", 4)

        if mode_idx == 4:
            with module_running(
                module, "The arm will be read repeatedly but not commanded."
            ):
                _run_obs_latency(module)
            return

        axis_name = str(OmegaConf.select(cfg, "axis") or "z").lower()
        if axis_name not in _AXES:
            print(f"axis must be one of {_AXES}, got {axis_name!r}.")
            return
        axis = _AXES.index(axis_name)
        amplitude = cfg_float(cfg, "amplitude", _AMPLITUDE_M)
        freq = cfg_float(cfg, "freq", _FREQ_HZ)
        n_poses = int(cfg_float(cfg, "poses", _POSES))
        distance = cfg_float(cfg, "distance", _DISTANCE_M)

        with module_running(
            module,
            "Torque will be released so you can place the arm by hand, the arm "
            "will move at speeds above the configured limit, and it will "
            "finally return here.\nKeep the volume clear and a hand on the power.",
        ):
            with returns_to_start(module, _SPEED_MARGIN * configured_speed(module)):
                if mode_idx in (1, 2):
                    _run_speed(module, n_poses, distance)
                if mode_idx == 2:
                    return
                # Checked after the ramp, so `all` vets the sweep against the
                # speed just measured rather than the one the run started with.
                if _within_speed_limit(module, amplitude, freq):
                    home = _position_by_hand(module, axis, amplitude)
                    if home is not None:
                        _run_action_latency(module, axis, amplitude, freq, home)
                if mode_idx == 1:
                    _run_obs_latency(module)
