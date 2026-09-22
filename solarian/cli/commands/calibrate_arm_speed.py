"""Measure the Cartesian speed the arm can actually sustain.

Fills ``max_speed[0]`` in the arm's module yaml, which
:class:`~solarian.utils.trajectory_interpolator.PoseTrajectoryInterpolator`
enforces by pushing a waypoint's arrival time later rather than by slowing
anything down. Configure it below the truth and every fast motion silently
arrives late; above it and the arm falls behind a plan that believes it is on
time.

Unlike the gripper's, this number is not a property of the hardware alone. The
joint speed limit reaches the end-effector through the Jacobian, so what is
achievable in the middle of the workspace is not achievable near full extension
or close to a singularity. A single figure is only safe if it is the *minimum*
over the region the arm works in, which is why this samples several poses and
every axis direction.

Overrides: ``poses=3 distance=0.06``.
"""

from __future__ import annotations

import time

import numpy as np
from omegaconf import OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.cli.commands._calibration import (
    SETTLE_S,
    achieved_speed,
    free_drive_pose,
    module_running,
    pick_module,
    ramp_to_plateau,
    returns_to_start,
    run_control_loop,
)
from solarian.core.arm_module import ArmModule

_AXES = ["x", "y", "z"]
_DISTANCE_M = 0.06
_POSES = 3
# Ladder bounds as multiples of the configured max_speed.
_RAMP_START = 0.5
_RAMP_CEILING = 4.0
# A direction is only sampled when the model can travel the whole test distance
# there; short of that the move would be clipped and read as a low speed.
_REACH_MARGIN = 0.95


def _configured_speed(module: ArmModule) -> float:
    max_speed = module.cfg.max_speed
    if isinstance(max_speed, (int, float)):
        return float(max_speed)
    return float(max_speed[0])


def _run_to(module: ArmModule, target, speed: float, axis: int | None = None):
    """Drive to *target* at *speed*, recording *axis* when one is given."""
    module.moveL(target, speed=speed)
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


def _reachable_directions(module: ArmModule, pose, distance: float) -> list[tuple[int, float]]:
    """The ``(axis, sign)`` pairs with room for the full test distance."""
    directions = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            travel, blocked = module.reachable_travel(pose, axis, sign * distance)
            if abs(travel) >= _REACH_MARGIN * distance:
                directions.append((axis, sign))
            else:
                label = "+" if sign > 0 else "-"
                print(
                    f"  skipping {label}{_AXES[axis]}: only "
                    f"{abs(travel) * 100:.1f} cm of room"
                    + (f" ({', '.join(blocked)} at its limit)" if blocked else "")
                )
    return directions


def _run(module: ArmModule, n_poses: int, distance: float) -> None:
    configured = _configured_speed(module)
    print(
        f"\nConfigured max_speed is {configured:.3f} m/s. Testing {distance * 100:.0f} cm "
        f"moves at {n_poses} pose(s), every axis direction with room for them."
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
                        f"\nThe arm tracked every speed up to {plateau:.3f} m/s, the "
                        f"ceiling this command will command ({_RAMP_CEILING:g}x "
                        "configured). Its real limit is above that."
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
    print(f"\n  max_speed: [{limit:.3f}, {_rot_speed(module)}]")
    print(
        "\nThis is the slowest direction sampled, which is the only figure safe "
        "to use as one global limit — the Jacobian makes the rest of the "
        "workspace slower still. Re-run over the poses a skill actually visits "
        "if it works somewhere this did not reach."
    )


def _rot_speed(module: ArmModule) -> str:
    """The configured rotation limit, untouched: only translation was measured."""
    max_speed = module.cfg.max_speed
    if isinstance(max_speed, (int, float)):
        return f"{float(max_speed):.3f}"
    return f"{float(max_speed[1] if len(max_speed) > 1 else max_speed[0]):.3f}"


@CommandRegistry.register("calibrate-arm-speed", "robot")
class CalibrateArmSpeedCommand(BaseCommand):
    help = "Measure the Cartesian speed the arm sustains across the workspace, for max_speed."
    description = """
    Commands short moves at rising speeds from several poses and reports the
    translation speed the arm actually keeps up with, which is the value for
    max_speed[0] in the arm's module yaml. Nothing is written.

    Measure it per pose because the joint speed limit reaches the end effector
    through the Jacobian: what is achievable near the middle of the workspace is
    not achievable stretched out. The command takes the slowest result.

    Torque is released first so you can place the arm, and it returns to where it
    started. It deliberately drives faster than the configured limit: keep the
    volume clear and a hand on the power.
    """
    params = {
        "poses": "How many poses to sample, each placed by hand (default: 3).",
        "distance": "Length of each test move in metres (default: 0.06).",
    }
    examples = (
        "solarian calibrate-arm-speed",
        "solarian calibrate-arm-speed poses=5 distance=0.1",
    )

    def handle(self, cfg) -> None:
        module = pick_module("arm", base_cls=ArmModule)
        if module is None:
            return

        n_poses = int(OmegaConf.select(cfg, "poses") or _POSES)
        distance = float(OmegaConf.select(cfg, "distance") or _DISTANCE_M)

        with module_running(
            module,
            "Torque will be released so you can place the arm, then it will make "
            f"{distance * 100:.0f} cm moves at speeds above the configured limit, "
            "and finally return here.\n"
            "Keep the volume clear and keep a hand on the power.",
        ):
            with returns_to_start(module, _configured_speed(module)):
                _run(module, n_poses, distance)
