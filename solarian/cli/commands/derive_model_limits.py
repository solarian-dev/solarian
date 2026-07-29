"""``derive-model-limits`` -- generate MJCF ``jnt_range`` from the real arm.

The IK sim<->real contract is: the MuJoCo model's ``jnt_range`` for each joint
equals the real arm's mechanical hard-stop angles. This command reads the real
range (from the existing calibration JSON, or a fresh guided limit sweep),
converts encoder counts to radians via the servo bus resolution, and prints the
``range`` values to paste into the model -- so the model matches the real arm by
construction.
"""

from __future__ import annotations

from solarian.cli import BaseCommand, CommandRegistry


@CommandRegistry.register("derive-model-limits", "robot")
class DeriveModelLimitsCommand(BaseCommand):
    help = "Derive MJCF joint ranges (radians) from the real arm's mechanical limits."

    def handle(self, cfg) -> None:
        from solarian.core.ik import alignment
        from solarian.cli.commands import _ik_common as ik

        module_name = getattr(cfg, "module", None) if cfg is not None else None
        sweep = bool(getattr(cfg, "sweep", False)) if cfg is not None else False

        name, module_class = ik.select_ik_module(module_name)
        print(f"\nDeriving model limits for module: {name}")
        module = ik.load_module(name, module_class)

        if sweep:
            calibration = self._sweep(module)
        else:
            calibration = ik.load_calibration_dict(module)
            if not calibration:
                raise SystemExit(
                    f"No calibration for '{name}'. Either run `solarian calibrate-module` "
                    f"first, or re-run this with `sweep=true` to record ranges now."
                )

        cpr = ik.counts_per_rev_resolver(module)
        if cpr is None:
            raise SystemExit(
                "This servo does not report an encoder resolution, so model limits "
                "cannot be derived numerically. Set the MJCF jnt_range from the arm's "
                "datasheet/CAD instead."
            )

        limits = alignment.derive_model_limits(calibration, cpr, symmetric=True)
        if not limits:
            raise SystemExit("No joints could be derived (missing counts/rev).")

        print("\nSuggested MJCF joint ranges (radians), centered at 0:\n")
        for joint, (lo, hi) in limits.items():
            print(f'  <joint name="{joint}" ... range="{lo:.5f} {hi:.5f}" />')
        print(
            "\nPaste these into the arm's MJCF <joint range=...> attributes and keep the "
            "sim `range_min/range_max` in the module yaml in sync. Then run "
            "`solarian verify-ik-calibration` to confirm.\n"
        )

    @staticmethod
    def _sweep(module):
        """Connect the module and record ranges of motion (torque off)."""
        print("\nConnecting to hardware for a limit sweep...")
        module.connect()
        module.start_calibration_mode()
        input("Move each joint through its FULL mechanical range, then press ENTER...")
        calibration: dict = {}
        from solarian.core.calibration import MotorCalibration

        for port, names in module.actuators_by_port.items():
            bus = module.module_bus.component_bus_by_port[port]
            mins, maxes = bus.record_ranges_of_motion(names)
            for n in names:
                if n not in mins:
                    continue
                calibration[n] = MotorCalibration(
                    id=module.actuators[n].cfg.id,
                    drive_mode=0,
                    homing_offset=0,
                    range_min=int(mins[n]),
                    range_max=int(maxes[n]),
                )
        module.end_calibration_mode()
        return calibration
