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
from solarian.core.data import PosData
from solarian.core.gripper_module import GripperModule
from solarian.utils.precise_sleep import precise_wait

_RANGE_STEP = 0.5


def _aperture_ruler_steps(lo: float, hi: float, step: float = _RANGE_STEP) -> list[float]:
    """Closed (if below 0), then 0, ``step``, … ``hi`` for a ruler check."""
    steps: list[float] = []
    if lo < 0.0:
        steps.append(lo)
    m = 0.0
    while m <= hi + 1e-9:
        steps.append(m)
        m += step
    if not steps or abs(steps[-1] - hi) > 1e-9:
        steps.append(hi)
    return steps


def _drive_to(module: GripperModule, value: float, settle_s: float = 0.3) -> None:
    module.moveJ(value)
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
            name = module._scalar_actuator_name()
            closed_norm = module._scalar_to_actuator(lo)
            open_norm = module._scalar_to_actuator(hi)
            print(
                f"Limits {lo:g}{suffix} → norm {closed_norm:.1f} (closed), "
                f"{hi:g}{suffix} → norm {open_norm:.1f} (open)."
            )
            for width in _aperture_ruler_steps(lo, hi):
                cmd_norm = module._scalar_to_actuator(width)
                extra = ""
                if name:
                    port = module.actuators[name].cfg.port
                    bus = module.module_bus.component_bus_by_port[port]
                    raw_goal = bus._unnormalize({name: cmd_norm})[name]
                    extra = f"  raw Goal {raw_goal:.0f}"
                print(f"{width:g}{suffix}  commanded norm {cmd_norm:.1f}{extra}")
                _drive_to(module, width)
                if name:
                    reading = module.read_sensors(PosData, [name])
                    meas_norm = float(reading[name]["value"])
                    meas = module._actuator_to_scalar(meas_norm)
                    print(
                        f"  measured {meas_norm:.1f} norm / {meas:.2f}{suffix}"
                    )
                input("Press Enter for next measurement")
