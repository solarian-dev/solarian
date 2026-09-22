"""``verify-calibration`` -- check spans against the IK model, then live-mirror an arm.

Converts each joint's calibrated encoder span to radians (via the servo bus's
counts-per-revolution hook) and compares it to the MuJoCo model's ``jnt_range``.
For an IK arm it then opens the same live sim mirror as ``solarian calibrate``
so you can confirm the real arm and the sim move together.
"""

from __future__ import annotations

from solarian.cli import BaseCommand, CommandRegistry


@CommandRegistry.register("verify-calibration", "robot")
class VerifyCalibrationCommand(BaseCommand):
    help = "Verify calibration spans against the IK model, then live-mirror the arm."
    description = """
    Converts each calibrated joint span to radians and compares it with the MuJoCo
    model's jnt_range, printing a per-joint pass/fail table. Then connects the real
    arm and opens the live sim mirror so you can watch sim and hardware move
    together.

    Run it after `solarian calibrate`. A mismatch means the model and the real arm
    disagree: fix the model with `solarian derive-model-limits`, or re-run the
    sweep. Joints whose servo reports no encoder resolution can only be checked in
    the mirror.
    """
    params = {
        "module": "Name of the IK module to verify; prompts when omitted.",
    }
    examples = (
        "solarian verify-calibration",
        "solarian verify-calibration module=right_arm",
    )

    def handle(self, cfg) -> None:
        from solarian.core.ik import alignment
        from solarian.cli.commands import _ik_common as ik

        module_name = getattr(cfg, "module", None) if cfg is not None else None
        name, module_class = ik.select_ik_module(module_name)
        print(f"\nVerifying calibration for module: {name}")

        module = ik.load_module(name, module_class)
        calibration = ik.load_calibration_dict(module)
        if not calibration:
            raise SystemExit(
                f"No calibration found for '{name}'. Run `solarian calibrate` first, "
                f"then re-run this check."
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
                "quantitative span check is unavailable. Use the live mirror below "
                "to verify direction/scale instead.\n"
            )

        report = alignment.verify_model_limits(calibration, model_limits, cpr)
        print("\n" + report.format_table() + "\n")

        if not report.passed:
            print(
                "One or more joints do not match. Fixes:\n"
                "  * mismatch  -> the model jnt_range differs from the real mechanical\n"
                "                 range. Run `solarian derive-model-limits` and paste the\n"
                "                 values into the MJCF, OR re-run the limit sweep.\n"
                "  * continuous-> joint has no hard stop; use the home step in\n"
                "                 `solarian calibrate`.\n"
                "  * unknown   -> no encoder resolution; verify with the live mirror.\n"
            )

        self._show_live_mirror(cfg, name)

    @staticmethod
    def _show_live_mirror(cfg, module_name: str) -> None:
        from solarian.conf import settings
        from solarian.utils import get_class
        from solarian.cli.commands.align_ik import IKAlignmentSteps

        RobotClass = get_class(settings.ROBOT_CLASS)
        robot = RobotClass(cfg)
        robot.is_sim = False
        robot.view = True
        print("\nConnecting to the real robot for the live sim mirror...")
        try:
            robot.connect(skip_calibrate=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not connect for the live mirror ({exc}). Span table above still stands.")
            return
        module = robot.modules[module_name]
        if not module.is_connected:
            module.connect()
        try:
            IKAlignmentSteps().run_mirror_only(robot, module, module_name)
        finally:
            try:
                robot.disconnect()
            except Exception:  # noqa: BLE001
                pass
