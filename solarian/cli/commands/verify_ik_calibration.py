"""``verify-ik-calibration`` -- check that a real arm's calibration matches its IK model.

Converts each joint's calibrated encoder span to radians (via the servo bus's
counts-per-revolution hook) and compares it to the MuJoCo model's ``jnt_range``.
A mismatch means normalized ``[-100, 100]`` joint commands map to different
radians on the real arm than the IK solver assumes -- the root cause of "Cartesian
moves track in sim but skew on the real robot". Runs offline from the calibration
JSON; no hardware required.
"""

from __future__ import annotations

from solarian.cli import BaseCommand, CommandRegistry


@CommandRegistry.register("verify-ik-calibration", "robot")
class VerifyIKCalibrationCommand(BaseCommand):
    help = "Verify a module's calibration matches its IK model joint limits."

    def handle(self, cfg) -> None:
        from solarian.core.ik import alignment
        from solarian.cli.commands import _ik_common as ik

        module_name = getattr(cfg, "module", None) if cfg is not None else None
        name, module_class = ik.select_ik_module(module_name)
        print(f"\nVerifying IK calibration for module: {name}")

        module = ik.load_module(name, module_class)
        calibration = ik.load_calibration_dict(module)
        if not calibration:
            raise SystemExit(
                f"No calibration found for '{name}'. Run `solarian calibrate-module` "
                f"(limit sweep) first, then re-run this check."
            )

        try:
            backend = ik.build_backend(module_class)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"Could not build the IK backend (needs the [sim] extra): {exc}"
            )
        model_limits = dict(backend.joint_limits)

        cpr = ik.counts_per_rev_resolver(module)
        if cpr is None:
            print(
                "\nNote: this servo does not report an encoder resolution, so the "
                "quantitative span check is unavailable. Use the live mirror in "
                "`solarian calibrate-ik` to verify direction/scale instead.\n"
            )

        report = alignment.verify_model_limits(calibration, model_limits, cpr)
        print("\n" + report.format_table() + "\n")

        if not report.passed:
            print(
                "One or more joints do not match. Fixes:\n"
                "  * mismatch  -> the model jnt_range differs from the real mechanical\n"
                "                 range. Run `solarian derive-model-limits` and paste the\n"
                "                 values into the MJCF, OR re-run the limit sweep.\n"
                "  * continuous-> joint has no hard stop; use the reference-pose fallback\n"
                "                 in `solarian calibrate-ik`.\n"
                "  * unknown   -> no encoder resolution; verify with the live mirror.\n"
            )
