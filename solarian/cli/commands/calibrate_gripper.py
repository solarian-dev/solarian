"""Interactive gripper calibration: latency (cross-correlation) or ranges (ruler).

Ports UMI ``calibrate_gripper_latency.py`` / ``calibrate_gripper_ranges.py`` onto
any :class:`~solarian.core.gripper_module.GripperModule`. Runs in-process (no
module subprocess / observation schema) like ``calibrate-module``.
"""

from __future__ import annotations

import os
import time

import numpy as np

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.gripper_module import GripperModule
from solarian.core.module import Module
from solarian.utils import get_class
from solarian.utils.latency import get_latency
from solarian.utils.precise_sleep import precise_wait

_RANGE_STEP = 0.5
_LATENCY_DURATION_S = 10.0
_LATENCY_SAMPLE_HZ = 100.0


def _collect_modules_from_class(cls) -> dict[str, type]:
    """Return {name: ModuleSubclass} for all Module descriptors on *cls*."""
    modules = {}
    for name in cls.__dict__:
        val = getattr(cls, name, None)
        if isinstance(val, Module):
            modules[name] = val.__class__
    return modules


def _prompt_choice(prompt: str, n: int) -> int:
    while True:
        user_input = input(f"{prompt} (1-{n}): ").strip()
        if user_input.isdigit():
            idx = int(user_input)
            if 1 <= idx <= n:
                return idx
            print(f"Invalid number. Please enter a number between 1 and {n}")
        else:
            print(f"Invalid input. Please enter a number between 1 and {n}")


def _unit_suffix(module: GripperModule) -> str:
    if getattr(module.cfg, "aperture_open_cm", None) is not None:
        return " (cm)"
    return ""


def _run_control_loop(
    module: GripperModule,
    duration: float,
    *,
    record: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    freq = max(float(module.frequency), 1.0)
    dt = 1.0 / freq
    t_start = time.monotonic()
    t_end = t_start + float(duration)
    iter_idx = 0
    actual_t: list[float] = []
    actual_x: list[float] = []
    while time.monotonic() < t_end:
        module.control_step()
        if record:
            actual_t.append(time.time())
            actual_x.append(module.read_scalar())
        iter_idx += 1
        precise_wait(t_end=t_start + dt * iter_idx, time_func=time.monotonic)
    return np.asarray(actual_t), np.asarray(actual_x)


def _drive_to(module: GripperModule, value: float, settle_s: float = 0.2) -> None:
    module.schedule_scalar(value, time.time() + 0.3)
    deadline = float(module.last_waypoint_time)
    freq = max(float(module.frequency), 1.0)
    dt = 1.0 / freq
    t_start = time.monotonic()
    iter_idx = 0
    while time.monotonic() < deadline:
        module.control_step()
        iter_idx += 1
        precise_wait(t_end=t_start + dt * iter_idx, time_func=time.monotonic)
    if settle_s > 0:
        _run_control_loop(module, settle_s, record=False)


def _run_latency(module: GripperModule) -> None:
    from matplotlib import pyplot as plt

    lo, hi = module.scalar_limits()
    duration = _LATENCY_DURATION_S
    sample_dt = 1.0 / _LATENCY_SAMPLE_HZ
    k = int(duration / sample_dt)
    sample_t = np.linspace(0.0, duration, k)
    value = np.sin(sample_t * duration / 1.5) * 0.5 + 0.5
    width = lo + value * (hi - lo)

    module.schedule_scalar(width[0], time.time() + 0.3)
    _run_control_loop(module, 1.0)

    # Match UMI: settle, then stamp the sinusoid 1 s into the future.
    timestamps = time.time() + sample_t + 1.0
    for i in range(k):
        module.schedule_scalar(width[i], timestamps[i])

    actual_t, actual_x = _run_control_loop(module, duration + 1.0, record=True)
    if actual_t.size < 2:
        print("Not enough samples to estimate latency.")
        return

    latency, info = get_latency(
        x_target=width,
        t_target=timestamps,
        x_actual=actual_x,
        t_actual=actual_t,
    )
    print(f"End-to-end latency: {latency}sec")
    print(
        "Copy this into ModuleConfig.receive_latency or the skill's "
        "gripper_action_latency if you want compensation."
    )

    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(15, 5, forward=True)

    ax = axes[0]
    ax.plot(info["lags"], info["correlation"])
    ax.set_xlabel("lag")
    ax.set_ylabel("cross-correlation")
    ax.set_title("Cross Correlation")

    suffix = _unit_suffix(module).strip() or "gripper-width"
    ax = axes[1]
    ax.plot(timestamps, width, label="target")
    ax.plot(actual_t, actual_x, label="actual")
    ax.set_xlabel("time")
    ax.set_ylabel(suffix)
    ax.legend()
    ax.set_title("Raw observation")

    ax = axes[2]
    t_samples = info["t_samples"] - info["t_samples"][0]
    ax.plot(t_samples, info["x_target"], label="target")
    ax.plot(t_samples - latency, info["x_actual"], label="actual-latency")
    ax.set_xlabel("time")
    ax.set_ylabel(suffix)
    ax.legend()
    ax.set_title(f"Aligned with latency={latency}")
    plt.show()


def _run_ranges(module: GripperModule) -> None:
    lo, hi = module.scalar_limits()
    suffix = _unit_suffix(module)
    m = lo
    while m <= hi + 1e-9:
        print(f"{m}{suffix}")
        _drive_to(module, m)
        input("Press Enter for next measurement")
        m += _RANGE_STEP

    print(f"Max gripper width: {hi}{suffix}")
    _drive_to(module, hi)
    input("Press Enter for next measurement")


@CommandRegistry.register("calibrate-gripper", "robot")
class CalibrateGripperCommand(BaseCommand):
    help = "Calibrate gripper latency (cross-correlation) or ranges (ruler check)."

    def handle(self, cfg) -> None:
        del cfg
        from solarian.conf import settings
        from solarian.utils.configs import load_config_yaml

        conf_path = os.environ.get(
            "SOLARIAN_CONF_PATH", os.path.join(os.getcwd(), "conf")
        )

        grippers: dict[str, tuple[type, str]] = {}
        for class_attr, source_label in [
            ("ROBOT_CLASS", "Robot"),
            ("CONTROLLER_CLASS", "Controller"),
        ]:
            class_path = getattr(settings, class_attr, None)
            if not class_path:
                continue
            parent_cls = get_class(class_path)
            for name, module_cls in _collect_modules_from_class(parent_cls).items():
                if issubclass(module_cls, GripperModule):
                    grippers[name] = (module_cls, source_label)

        if not grippers:
            print(
                "No GripperModule subclasses found on the project's Robot or "
                "Controller classes."
            )
            return

        entries = list(grippers.items())
        if len(entries) == 1:
            idx = 1
            print(
                f"\nUsing gripper: {entries[0][0]} "
                f"({entries[0][1][0].__name__}, {entries[0][1][1]})"
            )
        else:
            print("\n=== Gripper modules ===")
            for i, (name, (module_cls, source)) in enumerate(entries, 1):
                print(f"{i}. {name} ({module_cls.__name__}, {source})")
            print("\n=== Select gripper ===")
            idx = _prompt_choice("Enter module number", len(entries))

        module_name, (module_class, _) = entries[idx - 1]

        print("\n=== Calibrate gripper ===")
        print("1. latency  — measure command-to-observation delay (sinusoid)")
        print("2. ranges   — step through aperture so you can check a ruler")
        mode_idx = _prompt_choice("Select mode", 2)
        mode = "latency" if mode_idx == 1 else "ranges"

        yaml_path = os.path.join(conf_path, f"{module_name}.yaml")
        module_cfg = load_config_yaml(yaml_path, module_class.config_class)
        module = module_class(module_cfg, module_name)

        print(
            "\nThe gripper jaws will move. Keep clear of the workspace."
        )
        input("Press Enter to start...")

        module.load_calibration()
        module.connect(module.get_calibration() or {})
        try:
            module.initialize()
            if mode == "latency":
                _run_latency(module)
            else:
                _run_ranges(module)
        finally:
            try:
                module.disable_torque()
            except Exception:  # noqa: BLE001
                pass
            module.disconnect()
