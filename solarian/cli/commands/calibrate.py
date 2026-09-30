"""``solarian calibrate`` -- limit sweep, then IK alignment when the module needs it."""

from __future__ import annotations

import os

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.module import Module
from solarian.utils import get_class


def _collect_modules_from_class(cls) -> dict[str, type]:
    modules = {}
    for name in cls.__dict__:
        val = getattr(cls, name, None)
        if isinstance(val, Module):
            modules[name] = val.__class__
    return modules


def _has_ik(module_cls) -> bool:
    return getattr(module_cls, "ik_backend", None) is not None


@CommandRegistry.register("calibrate", "robot")
class CalibrateCommand(BaseCommand):
    category = "Robot"
    help = "Calibrate a module: hard-stop sweep, then IK alignment for arms."
    description = """
    Lists the modules the project's Robot and Controller declare, then calibrates
    the one you pick and writes the result to its module yaml: real_range_* (the
    travel window), center_offset (encoder-zero shift), drive_mode (sign) and home
    (normalized home pose).

    Torque is released for the sweep so you can move the joints by hand through
    their full mechanical range. For an arm with an IK backend it continues into
    the sim alignment steps, which home the model at q=0 and fix each joint's
    sign against a live MuJoCo mirror.

    Grippers stop at the encoder range; how far the jaws actually open at each
    command is measured separately by calibrate-gripper-ranges.
    """
    examples = (
        "solarian calibrate",
        "solarian calibrate -d pi",
    )

    def handle(self, cfg) -> None:
        from solarian.conf import settings
        from solarian.utils.configs import load_config_yaml

        conf_path = os.environ.get("SOLARIAN_CONF_PATH", os.path.join(os.getcwd(), "conf"))
        all_modules: dict[str, tuple[type, str]] = {}
        for class_attr, source_label in [
            ("ROBOT_CLASS", "Robot"),
            ("CONTROLLER_CLASS", "Controller"),
        ]:
            class_path = getattr(settings, class_attr, None)
            if not class_path:
                continue
            parent_cls = get_class(class_path)
            for name, module_cls in _collect_modules_from_class(parent_cls).items():
                all_modules[name] = (module_cls, source_label)

        if not all_modules:
            print("No modules found in the project's Robot or Controller classes.")
            return

        module_entries = list(all_modules.items())
        print("\n=== Available Modules ===")
        for idx, (name, (module_cls, source)) in enumerate(module_entries, 1):
            extra = ", IK" if _has_ik(module_cls) else ""
            print(f"{idx}. {name} ({module_cls.__name__}, {source}{extra})")

        print("\n=== Select Module to Calibrate ===")
        while True:
            user_input = input(f"Enter module number (1-{len(module_entries)}): ").strip()
            if user_input.isdigit():
                idx = int(user_input)
                if 1 <= idx <= len(module_entries):
                    break
            print(f"Invalid input. Enter a number between 1 and {len(module_entries)}")

        module_name, (module_class, source) = module_entries[idx - 1]
        print(
            f"\nCalibrating '{module_name}'. "
            "real_range_* = Present travel window; "
            "center_offset = encoder-zero shift; "
            "drive_mode = sign; home = normalized home pose."
        )

        if source == "Robot" and _has_ik(module_class):
            self._calibrate_ik_module(cfg, module_name)
            return

        yaml_path = os.path.join(conf_path, f"{module_name}.yaml")
        module_cfg = load_config_yaml(yaml_path, module_class.config_class)
        module = module_class(module_cfg, module_name)
        module.connect()
        try:
            module.calibrate()
            self._maybe_gripper_note(module)
        finally:
            try:
                module.disconnect()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _calibrate_ik_module(cfg, module_name: str) -> None:
        from solarian.conf import settings
        from solarian.cli.commands.align_ik import IKAlignmentSteps

        RobotClass = get_class(settings.ROBOT_CLASS)
        robot = RobotClass(cfg)
        robot.is_sim = False
        robot.view = True
        print("\nConnecting to the real robot (hardware)...")
        robot.connect(skip_calibrate=True)
        module = robot.modules[module_name]
        if not module.is_connected:
            module.connect()
        try:
            module.calibrate()
            IKAlignmentSteps().run(robot, module, module_name)
        finally:
            try:
                robot.disconnect()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _maybe_gripper_note(module) -> None:
        if not hasattr(module.cfg, "aperture_open_mm"):
            return
        print(
            "\nEncoder range and drive_mode are saved. How far the jaws actually\n"
            "open at each servo command is not measured here — run\n"
            "`solarian calibrate-gripper-ranges` with a ruler to record it."
        )
