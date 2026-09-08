"""Measure the gripper's action and observation latency.

Ports UMI ``calibrate_gripper_latency.py`` onto any
:class:`~solarian.core.gripper_module.GripperModule`. The two modes fill the two
config keys that compensate in opposite directions — see /docs/latency.
"""

from __future__ import annotations

import time

import numpy as np

from solarian.cli import BaseCommand, CommandRegistry
from solarian.cli.commands._calibration import (
    measure_read_latency,
    module_running,
    pick_module,
    plot_latency,
    prompt_choice,
    report_read_latency,
    run_control_loop,
)
from solarian.core.gripper_module import GripperModule
from solarian.utils.latency import get_latency

_DURATION_S = 10.0
_SAMPLE_HZ = 100.0
_OBS_ITERATIONS = 200


def _run_action_latency(module: GripperModule) -> None:
    lo, hi = module.scalar_limits()
    sample_dt = 1.0 / _SAMPLE_HZ
    k = int(_DURATION_S / sample_dt)
    sample_t = np.linspace(0.0, _DURATION_S, k)
    value = np.sin(sample_t * _DURATION_S / 1.5) * 0.5 + 0.5
    width = lo + value * (hi - lo)

    module.schedule_scalar(width[0], time.time() + 0.3)
    run_control_loop(module, 1.0)

    # Match UMI: settle, then stamp the sinusoid 1 s into the future.
    timestamps = time.time() + sample_t + 1.0
    for i in range(k):
        module.schedule_scalar(width[i], timestamps[i])

    actual_t, actual_x = run_control_loop(
        module, _DURATION_S + 1.0, record=module.read_scalar
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
    print(f"\nCommand-to-motion latency: {latency:.4f} s")
    print(f"\n  gripper_action_latency: {latency:.4f}")
    print(
        "Action latency moves a target time earlier: send the command sooner so "
        "the jaws arrive on time. It is NOT the same number as "
        "gripper_obs_latency — run mode 2 for that."
    )

    plot_latency(
        t_target=timestamps,
        x_target=width,
        t_actual=actual_t,
        x_actual=actual_x,
        latency=latency,
        info=info,
        ylabel=module.aperture_unit_suffix().strip() or "gripper-width",
    )


def _run_obs_latency(module: GripperModule) -> None:
    durations = measure_read_latency(module.read_scalar, _OBS_ITERATIONS)
    report_read_latency(durations, "gripper_obs_latency")


@CommandRegistry.register("calibrate-gripper-latency", "robot")
class CalibrateGripperLatencyCommand(BaseCommand):
    help = "Measure gripper action latency (sinusoid) or observation latency (read timing)."

    def handle(self, cfg) -> None:
        del cfg
        module = pick_module("gripper", base_cls=GripperModule)
        if module is None:
            return

        print("\n=== Calibrate gripper latency ===")
        print("1. action  — command-to-motion delay  → gripper_action_latency")
        print("2. obs     — sensor read duration     → gripper_obs_latency")
        mode_idx = prompt_choice("Select mode", 2)

        warning = (
            "The gripper jaws will move. Keep clear of the workspace."
            if mode_idx == 1
            else "The gripper will be read repeatedly but not commanded."
        )
        with module_running(module, warning):
            if mode_idx == 1:
                _run_action_latency(module)
            else:
                _run_obs_latency(module)
