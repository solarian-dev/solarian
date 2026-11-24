"""Measure how the arm behaves in time, and write it to the module yaml.

The same three numbers as ``calibrate-gripper`` — ``max_speed``,
``action_latency``, ``receive_latency`` — with two differences that come from
the arm being a kinematic chain rather than one axis. The speed is three
numbers here: translation and rotation (``max_speed``) and the joint speed
(``max_joint_speed``) that ``moveJ`` and Cartesian moves stay under.

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
from collections.abc import Callable

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

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
    speed_limits_lifted,
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
# Radians. Long enough that the 10-90% window is mostly cruise at the speeds a
# wrist reaches, short enough to fit inside most joints' travel.
_ROT_ANGLE_RAD = 0.5
_JOINT_SWING_RAD = 0.4
# A turn the model can only make by moving the tool this far (m) is not a turn
# in place, so that axis is skipped.
_ROT_POSITION_TOL_M = 0.005
# Same as ramp_to_plateau's: achieved under this fraction of commanded is
# falling behind.
_TRACK_MARGIN = 0.9


def _rot_speed(module: ArmModule) -> float:
    """The configured rotation limit (rad/s), ``max_speed[1]``."""
    max_speed = module.cfg.max_speed
    if isinstance(max_speed, (int, float)):
        return float(max_speed)
    return float(max_speed[1] if len(max_speed) > 1 else max_speed[0])


def _joint_speed(module: ArmModule) -> float:
    """The configured joint limit (rad/s), or the default ``moveJ`` speed."""
    if module.cfg.max_joint_speed is not None:
        return float(module.cfg.max_joint_speed)
    return float(module.default_joint_speed)


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


def _fit_amplitude(module: ArmModule, amplitude: float, freq: float) -> float:
    """*amplitude*, shrunk until the sinusoid's peak stays inside ``max_speed``."""
    fitted = min(amplitude, _SPEED_MARGIN * configured_speed(module) / (2.0 * np.pi * freq))
    if fitted < amplitude:
        print(
            f"\nShrinking the sweep to amplitude={fitted:.3f} m so its peak "
            "speed stays inside the max_speed just measured."
        )
    return fitted


# ---------------------------------------------------------------- speed


def _run_to(module: ArmModule, target, speed: float, rot_speed: float, record=None):
    """Drive to *target* at *speed* / *rot_speed*, sampling *record* when given."""
    # Unlimited acceleration: a ramp inside the 10-90% window would read as
    # the actuator being slower than it is.
    module.moveL(
        target,
        speed=speed,
        rot_speed=rot_speed,
        acceleration=np.inf,
        rot_acceleration=np.inf,
    )
    # Planned at the commanded speed even when the arm cannot manage it, so
    # stopping at the planned end would cut the move short.
    duration = module.last_waypoint_time - time.monotonic() + SETTLE_S
    return run_control_loop(module, duration, record=record)


def _angle_from(start: Rotation, pose) -> float:
    """Geodesic angle (rad) from *start* to the orientation of *pose*."""
    return float((start.inv() * Rotation.from_rotvec(np.asarray(pose)[3:6])).magnitude())


def _rotated(pose, axis: int, angle: float) -> np.ndarray:
    """*pose* turned by *angle* about base-frame *axis*, the position kept."""
    out = np.asarray(pose, dtype=np.float64).copy()
    turn = Rotation.from_rotvec(np.eye(3)[axis] * angle)
    out[3:6] = (turn * Rotation.from_rotvec(out[3:6])).as_rotvec()
    return out


class _Limit:
    """One speed, found by a ramp at its first sample and then held at the rest.

    A full ladder at every sample would multiply the run time by the number of
    samples, for a number that is taken as a minimum anyway. A later sample
    only lowers it when it falls behind by as much as the ramp itself counts as
    falling behind.
    """

    def __init__(self, label: str, unit: str, configured: float):
        self.label = label
        self.unit = unit
        self.configured = configured
        self.value = np.inf
        self.worst = ""
        self.at_ceiling = False

    def sample(self, where: str, run: Callable[[float], float]) -> None:
        if self.value == np.inf:
            print(f"\n{where} — finding the {self.label} plateau:")
            self.value, _, self.at_ceiling = ramp_to_plateau(
                run, _RAMP_START * self.configured, _RAMP_CEILING * self.configured
            )
            self.worst = where
            return
        got = run(self.value)
        print(f"  {where}: holds {got:.3f} of {self.value:.3f} {self.unit}")
        if got < _TRACK_MARGIN * self.value:
            self.value, self.worst, self.at_ceiling = got, where, False

    def result(self) -> float | None:
        """Print the outcome; the value to write, or ``None`` when there is none."""
        print(f"\n=== arm {self.label} ({self.unit}) ===")
        print(f"  configured:  {self.configured:.3f}")
        if self.value == np.inf:
            print("  measured:    nothing was sampled.")
            return None
        if self.value <= 0.0:
            print(
                f"  measured:    nothing — at {self.worst} the arm never got "
                "through the move. It did not move, or it was not read back. "
                "That is a fault to find, not a speed, so nothing was written."
            )
            return None
        if self.at_ceiling:
            print(
                f"  measured:    kept up to {self.value:.3f}, the ceiling this "
                f"command will command ({_RAMP_CEILING:g}x configured). The real "
                "limit is higher; this is written because the arm has shown it "
                f"holds it, and another run climbs up to {_RAMP_CEILING:g}x further."
            )
        else:
            print(f"  measured:    {self.value:.3f}, worst at {self.worst}")
        return round(self.value, 3)


def _sample_translation(module, limit: _Limit, pose, distance, back, where) -> None:
    for axis in range(3):
        for sign in (1.0, -1.0):
            label = ("+" if sign > 0 else "-") + _AXES[axis]
            travel, blocked = module.reachable_travel(pose, axis, sign * distance)
            if abs(travel) < _SPEED_REACH_MARGIN * distance:
                print(
                    f"  skipping {label}: only {abs(travel) * 100:.1f} cm of room"
                    + (f" ({', '.join(blocked)} at its limit)" if blocked else "")
                )
                continue

            def run(speed, a=axis, d=sign * distance):
                target = np.asarray(pose, dtype=np.float64).copy()
                target[a] += d
                t, x = _run_to(
                    module, target, speed, back[1],
                    record=lambda: float(module.read_pose()[a]),
                )
                _run_to(module, pose, *back)
                return achieved_speed(t, x, float(pose[a]), float(target[a]))

            limit.sample(f"{where} {label}", run)


def _sample_rotation(module, limit: _Limit, pose, back, where) -> None:
    start = Rotation.from_rotvec(np.asarray(pose)[3:6])
    for axis in range(3):
        for sign in (1.0, -1.0):
            label = ("+" if sign > 0 else "-") + "r" + _AXES[axis]
            target = _rotated(pose, axis, sign * _ROT_ANGLE_RAD)
            achieved, blocked, _q = module.reachable_pose(target)
            short = _angle_from(start, achieved)
            if (
                abs(short - _ROT_ANGLE_RAD) > (1.0 - _SPEED_REACH_MARGIN) * _ROT_ANGLE_RAD
                or np.linalg.norm(achieved[:3] - np.asarray(pose)[:3]) > _ROT_POSITION_TOL_M
            ):
                print(
                    f"  skipping {label}: the model cannot turn the full "
                    f"{_ROT_ANGLE_RAD:.2f} rad here"
                    + (f" ({', '.join(blocked)} at its limit)" if blocked else "")
                )
                continue

            def run(speed, tgt=target):
                t, x = _run_to(
                    module, tgt, back[0], speed,
                    record=lambda: _angle_from(start, module.read_pose()),
                )
                _run_to(module, pose, *back)
                return achieved_speed(t, x, 0.0, _ROT_ANGLE_RAD)

            limit.sample(f"{where} {label}", run)


def _sample_joints(module, limit: _Limit, back_joint: float, where) -> None:
    joints = module.read_joints()
    q_home = np.array(list(joints.values()))
    for index, name in enumerate(joints):
        delta = next(
            (
                d for d in (_JOINT_SWING_RAD, -_JOINT_SWING_RAD)
                if abs(module.joint_travel(index, d)) >= _SPEED_REACH_MARGIN * _JOINT_SWING_RAD
            ),
            None,
        )
        if delta is None:
            print(f"  skipping {name}: no room for {_JOINT_SWING_RAD:.2f} rad either way")
            continue

        def run(speed, i=index, d=delta, name=name):
            target = q_home.copy()
            target[i] += d
            module.moveJ(target, speed=speed, acceleration=np.inf)
            t, x = run_control_loop(
                module,
                module.last_waypoint_time - time.monotonic() + SETTLE_S,
                record=lambda: float(module.read_joints()[name]),
            )
            module.moveJ(q_home, speed=back_joint)
            run_control_loop(module, module.last_waypoint_time - time.monotonic() + SETTLE_S)
            return achieved_speed(t, x, float(q_home[i]), float(target[i]))

        limit.sample(f"{where} {name}", run)


def _run_speed(module: ArmModule, n_poses: int, distance: float) -> None:
    translation = _Limit("max_speed[0], translation", "m/s", configured_speed(module))
    rotation = _Limit("max_speed[1], rotation", "rad/s", _rot_speed(module))
    joint = _Limit("max_joint_speed", "rad/s", _joint_speed(module))
    # Every return move runs at the configured speeds, under the motor caps
    # that will be in force after the run.
    back = (_SPEED_MARGIN * translation.configured, _SPEED_MARGIN * rotation.configured)
    print(
        f"\nConfigured max_speed is [{translation.configured:.3f} m/s, "
        f"{rotation.configured:.3f} rad/s], max_joint_speed {joint.configured:.3f} rad/s. "
        f"At each of {n_poses} pose(s): {distance * 100:.0f} cm moves along every "
        f"axis with room, {_ROT_ANGLE_RAD:.2f} rad turns about every axis, and "
        f"{_JOINT_SWING_RAD:.2f} rad swings of every joint."
    )

    with speed_limits_lifted(
        module,
        max_speed=[_RAMP_CEILING * translation.configured, _RAMP_CEILING * rotation.configured],
        max_joint_speed=_RAMP_CEILING * joint.configured,
        # Ramps inside the 10-90% window would read as a slower arm; some arms
        # treat these as ceilings on every caller.
        max_accel=None,
        max_joint_accel=None,
    ):
        for pose_idx in range(1, n_poses + 1):
            pose = free_drive_pose(
                module,
                f"Pose {pose_idx} of {n_poses}: put the end-effector somewhere the "
                f"arm works, with {distance * 100:.0f} cm clear around it in every "
                "direction and room for the wrist to turn.",
            )
            where = f"pose {pose_idx}"
            _sample_translation(module, translation, pose, distance, back, where)
            _sample_rotation(module, rotation, pose, back, where)
            _sample_joints(module, joint, _SPEED_MARGIN * joint.configured, where)

    pos = translation.result()
    rot = rotation.result()
    joint_speed = joint.result()
    print(
        "\nEach is the slowest sample, which is the only figure safe to use as "
        "one global limit — the Jacobian makes the rest of the workspace slower "
        "still. Re-run over the poses a skill actually visits if it works "
        "somewhere this did not reach."
    )
    if pos is not None or rot is not None:
        save_module_field(
            module,
            "max_speed",
            [
                translation.configured if pos is None else pos,
                rotation.configured if rot is None else rot,
            ],
        )
    if joint_speed is not None:
        save_module_field(module, "max_joint_speed", joint_speed)


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
    print("\n=== arm receive_latency (s) ===")
    for bus in module.module_bus.component_bus_by_port.values():
        joints = getattr(bus, "inner_bus", bus)
        if hasattr(joints, "read_position_times"):
            _run_sample_age(module, joints.read_position_times)
            return
    durations = measure_read_latency(module.read_pose, _OBS_ITERATIONS)
    print(
        "This is a serial bus round trip plus forward kinematics, so expect "
        "single-digit milliseconds — not UMI's 0.000124, which is a UR5 RTDE "
        "poll off a 500 Hz stream."
    )
    save_module_field(module, "receive_latency", round(report_read_latency(durations), 6))


def _run_sample_age(
    module: ArmModule, read_position_times: Callable[[list[str]], dict[str, float]]
) -> None:
    """Measure how old a joint reading is when the module stamps it.

    For a bus that answers reads from replies to the commands it sends, the
    read itself is fast and its value up to a tick old, so this times the
    reading rather than the read. It ticks the control loop as the module does
    in service, reading right after each step as ``update_observations`` does.
    """
    names = list(module.read_joints())

    def sample_age() -> float:
        # Arrival times are time.monotonic() from the bus, possibly in its own
        # process, which shares this clock.
        arrived = read_position_times(names)
        return time.monotonic() - float(np.mean(list(arrived.values())))

    _, age = run_control_loop(
        module, _OBS_ITERATIONS / max(float(module.frequency), 1.0), record=sample_age
    )
    if age.size == 0:
        print("No readings taken.")
        return
    print(
        f"\nJoint reading age over {age.size} ticks: mean {age.mean() * 1e3:.3f} ms, "
        f"std {age.std() * 1e3:.3f} ms, p95 {np.percentile(age, 95) * 1e3:.3f} ms"
    )
    print(
        "The bus answers reads from the motors' replies to its commands, so a "
        "reading is as old as the last reply — up to a control tick — however "
        "fast the read returns."
    )
    save_module_field(module, "receive_latency", round(float(age.mean()), 6))


@CommandRegistry.register("calibrate-arm", "robot")
class CalibrateArmCommand(BaseCommand):
    category = "Robot"
    help = (
        "Measure the arm's max_speed, max_joint_speed and its two latencies, "
        "into the module yaml."
    )
    description = """
    Pick an arm module, then a mode:

      all     speed, then action latency, then observation latency
      speed   from several poses, at rising speeds and worst kept: short moves
              along x/y/z -> max_speed[0]; turns about x/y/z -> max_speed[1];
              a swing of each joint -> max_joint_speed
      action  the arm follows a Cartesian sinusoid on one axis and the measured
              pose is cross-correlated against the command -> action_latency
      obs     repeated pose reads are timed -> receive_latency; on a bus that
              answers reads from cached replies (Robstride), the age of the
              reading is measured instead, with the control loop ticking

    Each result is written to conf/<module>.yaml as it is measured. A ramp
    climbs to 4x the configured speed; if the arm keeps up all the way, that is
    written as a floor it has shown it holds, and another run climbs further.
    A negative latency means the sweep did not track, and is not written.

    While the ramps run, the configured speeds and accelerations are lifted and
    so is the motors' own speed cap; they are set back to follow the new values
    afterwards.

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
        "amplitude": (
            "Half-stroke of the action sweep in metres (default: 0.02). "
            "`all` shrinks it to stay inside the max_speed it measured."
        ),
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
        print("2. speed   — speed ramps                → max_speed, max_joint_speed")
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
                # Fitted after the ramp, so `all` sizes the sweep from the
                # speed just measured rather than the one the run started with.
                if mode_idx == 1:
                    amplitude = _fit_amplitude(module, amplitude, freq)
                if mode_idx == 1 or _within_speed_limit(module, amplitude, freq):
                    home = _position_by_hand(module, axis, amplitude)
                    if home is not None:
                        _run_action_latency(module, axis, amplitude, freq, home)
                if mode_idx == 1:
                    _run_obs_latency(module)
