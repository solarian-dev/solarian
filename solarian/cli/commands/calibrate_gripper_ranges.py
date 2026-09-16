"""Measure what a gripper's jaws actually open to at each servo command.

Ports UMI ``calibrate_gripper_ranges.py``. The servo sweeps its normalized
range; you read a ruler at each stop. The pairs become
``aperture_table`` in the module yaml, which
:class:`~solarian.core.gripper_module.GripperModule` interpolates at runtime.

Two calibration points would describe the ends of travel correctly and
everything between them wrong, because a parallel gripper's linkage compresses
the opening at one end of the stroke. This is how that curve gets measured.
"""

from __future__ import annotations

import time

import numpy as np

from solarian.cli import BaseCommand, CommandRegistry
from solarian.cli.commands._calibration import module_running, pick_module
from solarian.core.aperture_table import ApertureTable
from solarian.core.data import PosData
from solarian.core.gripper_module import APERTURE_UNIT, GripperModule
from solarian.core.module_yaml import patch_top_level_mapping
from solarian.utils.precise_sleep import precise_wait

_DEFAULT_STEPS = 21
# Seconds to cross the full normalized range. The sweep steps open-loop rather
# than through moveJ, so this is the only thing keeping the jaws off their stops.
_FULL_TRAVEL_S = 3.0
_SETTLE_S = 0.4


def _prompt_steps() -> int:
    raw = input(f"Measurements across the range [{_DEFAULT_STEPS}]: ").strip()
    if not raw:
        return _DEFAULT_STEPS
    if raw.isdigit() and int(raw) >= 2:
        return int(raw)
    print(f"Need a whole number of at least 2; using {_DEFAULT_STEPS}.")
    return _DEFAULT_STEPS


def _prompt_aperture() -> float | None:
    """Read one ruler measurement, or ``None`` to leave this stop out."""
    while True:
        raw = input(
            f"  measured opening in {APERTURE_UNIT} (Enter to skip): "
        ).strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            print(f"  '{raw}' is not a number.")


def _ramp_to_norm(module: GripperModule, name: str, start: float, target: float) -> None:
    """Walk the servo from *start* to *target* in normalized space, and hold.

    Deliberately bypasses both ``moveJ`` and ``control_step``. ``moveJ`` would
    convert mm through the very table this command exists to measure, and
    ``control_step`` re-writes the servo from the scalar interpolator, which is
    still seeded wherever the gripper was when the sweep started — it would drag
    the jaws back there the moment we stopped writing, leaving nothing to
    measure. The servo holds the last goal we send it while the ruler comes out.
    """
    freq = max(float(module.frequency), 1.0)
    dt = 1.0 / freq
    span = abs(target - start)
    ticks = max(int(freq * _FULL_TRAVEL_S * span / 200.0), 1)
    t_start = time.monotonic()
    for i in range(1, ticks + 1):
        module.write_actuators(
            {name: PosData(value=start + (target - start) * i / ticks)}
        )
        precise_wait(t_end=t_start + dt * i, time_func=time.monotonic)
    time.sleep(_SETTLE_S)


def _sweep(module: GripperModule, name: str, steps: int) -> list[tuple[float, float]]:
    """Step through normalized commands, collecting ``(norm, aperture_mm)``."""
    port = module.actuators[name].cfg.port
    bus = module.module_bus.component_bus_by_port[port]
    measured: list[tuple[float, float]] = []
    current = float(module.read_sensors(PosData, [name])[name]["value"])

    for norm in np.linspace(-100.0, 100.0, steps):
        norm = float(norm)
        raw_goal = bus._unnormalize({name: norm})[name]
        print(f"\nnorm {norm:+7.1f}  raw Goal {raw_goal:.0f}")
        _ramp_to_norm(module, name, current, norm)
        current = norm
        reading = float(module.read_sensors(PosData, [name])[name]["value"])
        print(f"  servo reports {reading:+.1f} norm")
        aperture = _prompt_aperture()
        if aperture is not None:
            measured.append((norm, aperture))
    return measured


def _build_table(measured: list[tuple[float, float]]) -> ApertureTable | None:
    """Validate the measurements, explaining any rejection in ruler terms."""
    if len(measured) < 2:
        print(
            f"\nOnly {len(measured)} measurement(s) recorded; a table needs at "
            "least 2. Nothing written."
        )
        return None
    norms = [n for n, _ in measured]
    apertures = [a for _, a in measured]
    try:
        return ApertureTable(norm=norms, aperture_mm=apertures)
    except ValueError as exc:
        print(f"\nThese measurements cannot be inverted: {exc}")
        print(
            "Two stops read the same opening, or the opening turned back on "
            "itself. Check for a jaw hitting its stop before the servo does, "
            "then re-run. Nothing written."
        )
        return None


@CommandRegistry.register("calibrate-gripper-ranges", "robot")
class CalibrateGripperRangesCommand(BaseCommand):
    help = "Sweep the gripper and record the aperture curve against a ruler."

    def handle(self, cfg) -> None:
        del cfg
        module = pick_module("gripper", base_cls=GripperModule)
        if module is None:
            return

        yaml_path = module._module_yaml_path()
        if yaml_path is None:
            print(f"No conf/{module.name}.yaml to write the table to.")
            return

        steps = _prompt_steps()
        with module_running(
            module, "The gripper jaws will move. Keep clear of the workspace."
        ):
            name = module._scalar_actuator_name()
            if not name:
                print(f"Module {module.name} has no scalar actuator to sweep.")
                return
            print(
                f"\nMeasure fingertip separation in {APERTURE_UNIT} at each stop. "
                "Skip any stop the ruler cannot reach."
            )
            measured = _sweep(module, name, steps)

        table = _build_table(measured)
        if table is None:
            return

        patch_top_level_mapping(
            yaml_path,
            "aperture_table",
            {"norm": table.norm, "aperture_mm": table.aperture_mm},
        )
        lo, hi = table.limits
        print(
            f"\nWrote {len(table.norm)} measurements to {yaml_path}.\n"
            f"Travel is {lo:g}-{hi:g} {APERTURE_UNIT}; commands outside it saturate."
        )
