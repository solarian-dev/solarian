"""Step the gripper through its aperture range so you can check it with a ruler.

Ports UMI ``calibrate_gripper_ranges.py``. Nothing here measures latency — this
tells you whether ``aperture_closed_cm`` / ``aperture_open_cm`` in the module
YAML describe the real jaws.
"""

from __future__ import annotations

import time

from solarian.cli import BaseCommand, CommandRegistry
from solarian.cli.commands._calibration import (
    module_running,
    pick_module,
    run_control_loop,
)
from solarian.core.gripper_module import GripperModule
from solarian.utils.precise_sleep import precise_wait

_RANGE_STEP = 0.5


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
        run_control_loop(module, settle_s)


@CommandRegistry.register("calibrate-gripper-ranges", "robot")
class CalibrateGripperRangesCommand(BaseCommand):
    help = "Step the gripper through its aperture range for a ruler check."

    def handle(self, cfg) -> None:
        del cfg
        module = pick_module("gripper", base_cls=GripperModule)
        if module is None:
            return

        with module_running(
            module, "The gripper jaws will move. Keep clear of the workspace."
        ):
            lo, hi = module.scalar_limits()
            suffix = module.aperture_unit_suffix()
            m = 0
            while m <= hi + 1e-9:
                print(f"{m}{suffix}")
                _drive_to(module, m)
                input("Press Enter for next measurement")
                m += _RANGE_STEP

            print(f"Max gripper width: {hi}{suffix}")
            _drive_to(module, hi)
            input("Press Enter for next measurement")
