"""Shared plumbing for the interactive `calibrate-*-latency` commands.

Each of those commands picks one module off the project's Robot/Controller
class, drives it in-process (no module subprocess, no observation schema, like
``solarian calibrate``) and reports a delay. The discovery, the fixed-rate control
loop and the cross-correlation plot are identical across arm and gripper, so
they live here.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from contextlib import contextmanager

import numpy as np
from omegaconf import OmegaConf

from solarian.core.module import Module
from solarian.utils import get_class
from solarian.utils.precise_sleep import precise_wait

# Servos need a moment to take up the slack after torque re-engages, before the
# pose a calibration is built around is read.
SETTLE_S = 0.5


def cfg_float(cfg, key: str, default: float) -> float:
    """A numeric command override, or *default* when it was not given."""
    val = OmegaConf.select(cfg, key)
    return default if val is None else float(val)


def configured_speed(module: Module) -> float:
    """The speed limit this module's interpolator enforces.

    ``max_speed`` is a list on an arm (translation, rotation) and a scalar on a
    gripper; only the first entry is a linear speed either way.
    """
    max_speed = module.cfg.max_speed
    if isinstance(max_speed, (int, float)):
        return float(max_speed)
    return float(max_speed[0])


def save_module_field(module: Module, key: str, value) -> None:
    """Write one measurement to the module's yaml, and to the live config.

    The live config matters within a single run: the latency sweep sizes its
    amplitude off ``max_speed``, so a speed measured a minute ago has to be the
    one in force by the time the sweep is planned.
    """
    from solarian.core.module_yaml import patch_top_level_value

    setattr(module.cfg, key, value)
    yaml_path = module._module_yaml_path()
    if yaml_path is None:
        print(f"  {key}: {value} — kept in memory, {module.name} has no yaml")
        return
    patch_top_level_value(yaml_path, key, value)
    print(f"  wrote {key}: {value} to {yaml_path}")


def _opencv_bgr(frame, color_mode: str) -> np.ndarray:
    """cv2.imshow / imwrite expect BGR. Wrist cameras publish RGB."""
    import cv2

    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if str(color_mode).lower() == "rgb":
        return np.ascontiguousarray(frame[:, :, ::-1])
    return np.ascontiguousarray(frame)


def prompt_choice(prompt: str, n: int) -> int:
    while True:
        user_input = input(f"{prompt} (1-{n}): ").strip()
        if user_input.isdigit():
            idx = int(user_input)
            if 1 <= idx <= n:
                return idx
            print(f"Invalid number. Please enter a number between 1 and {n}")
        else:
            print(f"Invalid input. Please enter a number between 1 and {n}")


def _collect_modules_from_class(cls) -> dict[str, type]:
    """Return {name: ModuleSubclass} for all Module descriptors on *cls*."""
    modules = {}
    for name in cls.__dict__:
        val = getattr(cls, name, None)
        if isinstance(val, Module):
            modules[name] = val.__class__
    return modules


def _declares_sensor(module_cls: type, sensor_cls: type) -> bool:
    """True when *module_cls* declares a *sensor_cls* descriptor anywhere in its MRO."""
    return any(
        isinstance(value, sensor_cls)
        for cls in module_cls.__mro__
        for value in cls.__dict__.values()
    )


def pick_module(
    label: str, *, base_cls: type = Module, sensor_cls: type | None = None
) -> Module | None:
    """Prompt for one matching module on this project and construct it.

    Searches the project's Robot and Controller classes for modules that
    subclass *base_cls* and, when *sensor_cls* is given, declare a sensor of
    that type (cameras share no module base class, only a ``Camera`` sensor).
    Prompts when more than one matches and loads ``conf/modules/<module_name>.yaml``.
    Returns ``None`` (after printing why) when nothing matches. The result is
    not connected — wrap it in :func:`module_running`.
    """
    from solarian.conf import settings
    from solarian.core.conf_paths import module_config_path
    from solarian.utils.configs import load_config_yaml

    found: dict[str, tuple[type, str]] = {}
    for class_attr, source_label in [
        ("ROBOT_CLASS", "Robot"),
        ("CONTROLLER_CLASS", "Controller"),
    ]:
        class_path = getattr(settings, class_attr, None)
        if not class_path:
            continue
        try:
            parent_cls = get_class(class_path)
        except ImportError as exc:
            # The robot machine is not expected to carry controller-side
            # hardware plugins (and vice versa) — that is what setup_modules
            # exists for. Only the half that imports can own the module we want.
            print(f"Skipping {class_attr} ({class_path}): {exc}")
            continue
        for name, module_cls in _collect_modules_from_class(parent_cls).items():
            if not issubclass(module_cls, base_cls):
                continue
            if sensor_cls is not None and not _declares_sensor(module_cls, sensor_cls):
                continue
            found[name] = (module_cls, source_label)

    if not found:
        what = sensor_cls.__name__ if sensor_cls is not None else base_cls.__name__
        print(
            f"No module with a {what} found on the project's Robot or "
            "Controller classes."
        )
        return None

    entries = list(found.items())
    if len(entries) == 1:
        idx = 1
        print(
            f"\nUsing {label}: {entries[0][0]} "
            f"({entries[0][1][0].__name__}, {entries[0][1][1]})"
        )
    else:
        print(f"\n=== {label.capitalize()} modules ===")
        for i, (name, (module_cls, source)) in enumerate(entries, 1):
            print(f"{i}. {name} ({module_cls.__name__}, {source})")
        print(f"\n=== Select {label} ===")
        idx = prompt_choice("Enter module number", len(entries))

    module_name, (module_class, _) = entries[idx - 1]
    module_cfg = load_config_yaml(str(module_config_path(module_name)), module_class.config_class)
    return module_class(module_cfg, module_name)


@contextmanager
def module_running(module: Module, warning: str, confirm: bool = True):
    """Connect, initialise, and always release the hardware afterwards.

    Set *confirm* false for the inner module of a nested pair, where the outer
    module's prompt already stood in for both and a second one just adds a
    keystroke between the user and the same decision.
    """
    print(f"\n{warning}")
    if confirm:
        input("Press Enter to start...")

    module.load_calibration()
    module.connect(module.get_calibration() or {})
    if module.needs_motor_calibration() and module.calibration:
        module.write_calibration()
    try:
        module.initialize()
        yield module
    finally:
        # Sensor-only modules (cameras) have no torque, and a module whose bus
        # already dropped cannot be relaxed — neither must block the disconnect.
        try:
            module.disable_torque()
        except Exception:  # noqa: BLE001
            pass
        module.disconnect()


def run_control_loop(
    module: Module,
    duration: float,
    record: Callable[[], float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tick ``module.control_step()`` at ``module.frequency`` for *duration*.

    When *record* is given, samples it after every step and returns the
    ``(wall_time, value)`` series. Sample times are ``time.time()`` because they
    are compared against wall-clock waypoint timestamps.
    """
    freq = max(float(module.frequency), 1.0)
    dt = 1.0 / freq
    t_start = time.monotonic()
    t_end = t_start + float(duration)
    iter_idx = 0
    actual_t: list[float] = []
    actual_x: list[float] = []
    while time.monotonic() < t_end:
        module.control_step()
        if record is not None:
            actual_t.append(time.time())
            actual_x.append(record())
        iter_idx += 1
        precise_wait(t_end=t_start + dt * iter_idx, time_func=time.monotonic)
    return np.asarray(actual_t), np.asarray(actual_x)


@contextmanager
def returns_to_start(module, speed: float):
    """Record where the arm is, hand that pose to the body, and put it back after.

    Runs on the way out of a failure too: a calibration that stopped early has
    still left the arm somewhere the user did not choose.
    """
    start = module.read_pose()
    try:
        yield start
    finally:
        print("\nReturning the arm to where it started...")
        try:
            module.moveL(start, speed=speed)
            run_control_loop(module, module.last_waypoint_time - time.monotonic())
        except Exception as exc:  # noqa: BLE001
            # A bus that already dropped must not mask why we are unwinding.
            print(f"Could not drive back to the start pose: {exc}")


def free_drive_pose(module, instruction: str):
    """Hand the arm to the user, then energise it wherever they leave it.

    Torque comes off only once the user confirms a hand is on the arm. A servo
    arm falls the instant it is released, so disengaging in the same breath as
    printing the warning drops it before the warning has been read.
    """
    print(f"\n{instruction}")
    input("Take hold of the arm, then press Enter to release torque...")

    module.start_calibration_mode()
    input("Torque is OFF. Move the arm, then press Enter while still holding it...")

    # Goal position survives the torque-off, so latch it to where the arm now is
    # before energising or it snaps back to the pre-freedrive target.
    module.end_calibration_mode()
    module.enable_torque_holding()
    time.sleep(SETTLE_S)

    # Read after the user lets go: held, the pose includes whatever they were
    # supporting; released, it is the pose the servos actually hold against
    # gravity, which is what the calibration will be centred on.
    input("Torque is ON. Let go of the arm, then press Enter...")
    time.sleep(SETTLE_S)
    return module.read_pose()


def achieved_speed(t: np.ndarray, x: np.ndarray, start: float, target: float) -> float:
    """Rate between the 10% and 90% crossings of a commanded move, in units/s.

    The middle of the move only: the ends are acceleration and settling, and
    averaging over them would report something lower than the rate the actuator
    actually sustains.
    """
    span = target - start
    if t.size < 2 or span == 0.0:
        return 0.0
    progress = (x - start) / span
    reached = np.flatnonzero(progress >= 0.9)
    begun = np.flatnonzero(progress >= 0.1)
    if reached.size == 0 or begun.size == 0:
        return 0.0
    elapsed = t[reached[0]] - t[begun[0]]
    if elapsed <= 0.0:
        return 0.0
    return abs(0.8 * span) / elapsed


def ramp_to_plateau(
    run: Callable[[float], float],
    start: float,
    ceiling: float,
    track_margin: float = 0.9,
    step: float = 1.25,
) -> tuple[float, list[tuple[float, float]], bool]:
    """Raise the commanded speed until the achieved speed stops following it.

    *run* performs one move at a commanded speed and returns the speed actually
    achieved. Past the hardware's limit the achieved speed plateaus while the
    command keeps rising, so the largest achieved value is the limit. Returns
    that plateau, the ``(commanded, achieved)`` ladder, and whether *ceiling*
    was reached without ever falling behind — in which case the limit is above
    everything that was tried, not the value returned.
    """
    ladder: list[tuple[float, float]] = []
    commanded = start
    while commanded <= ceiling:
        got = run(commanded)
        ladder.append((commanded, got))
        print(
            f"  commanded {commanded:.3f} → achieved {got:.3f} "
            f"({got / commanded:.0%})"
        )
        if got < track_margin * commanded:
            return max(a for _, a in ladder), ladder, False
        commanded *= step
    return max(a for _, a in ladder), ladder, True


def measure_read_latency(
    read: Callable[[], object], iterations: int = 200
) -> np.ndarray:
    """Time *read* over *iterations* calls, returning the durations in seconds.

    This is what an observation latency is on Solarian hardware:
    ``Module.update_observations`` stamps ``synced_time()`` *after* the read
    returns, so the value it stamps was physically true roughly one read
    duration earlier.
    """
    durations = np.zeros(iterations, dtype=np.float64)
    for i in range(iterations):
        t0 = time.perf_counter()
        read()
        durations[i] = time.perf_counter() - t0
    return durations


def report_read_latency(durations: np.ndarray) -> float:
    """Print the read-duration spread and return the mean, in seconds."""
    print(
        f"\nRead duration over {durations.size} samples: "
        f"mean {durations.mean() * 1e3:.3f} ms, "
        f"std {durations.std() * 1e3:.3f} ms, "
        f"p95 {np.percentile(durations, 95) * 1e3:.3f} ms"
    )
    print(
        "Observation latency moves a timestamp backwards: the reading was true "
        "before we stamped it."
    )
    return float(durations.mean())


def plot_latency(
    *,
    t_target: np.ndarray,
    x_target: np.ndarray,
    t_actual: np.ndarray,
    x_actual: np.ndarray,
    latency: float,
    info: dict,
    ylabel: str,
) -> None:
    """UMI's three-panel latency figure: correlation, raw series, aligned series."""
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(15, 5, forward=True)

    ax = axes[0]
    ax.plot(info["lags"], info["correlation"])
    ax.set_xlabel("lag")
    ax.set_ylabel("cross-correlation")
    ax.set_title("Cross Correlation")

    ax = axes[1]
    ax.plot(t_target, x_target, label="target")
    ax.plot(t_actual, x_actual, label="actual")
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title("Raw observation")

    ax = axes[2]
    t_samples = info["t_samples"] - info["t_samples"][0]
    ax.plot(t_samples, info["x_target"], label="target")
    ax.plot(t_samples - latency, info["x_actual"], label="actual-latency")
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title(f"Aligned with latency={latency}")
    plt.show()
