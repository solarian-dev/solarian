import os

from solarian.cli import BaseCommand, CommandRegistry
from solarian.utils import get_class
from solarian.core.module import Module


def _collect_modules_from_class(cls) -> dict[str, type]:
    """Return {name: ModuleSubclass} for all Module descriptors on *cls*."""
    modules = {}
    for name in cls.__dict__:
        val = getattr(cls, name, None)
        if isinstance(val, Module):
            modules[name] = val.__class__
    return modules


@CommandRegistry.register('calibrate-module', 'robot')
class CalibrateModuleCommand(BaseCommand):
    help = 'Calibrate module device.'

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
            print(f"{idx}. {name} ({module_cls.__name__}, {source})")

        print("\n=== Select Module to Calibrate ===")
        while True:
            user_input = input(f"Enter module number (1-{len(module_entries)}): ").strip()
            if user_input.isdigit():
                idx = int(user_input)
                if 1 <= idx <= len(module_entries):
                    break
                else:
                    print(f"Invalid number. Please enter a number between 1 and {len(module_entries)}")
            else:
                print(f"Invalid input. Please enter a number between 1 and {len(module_entries)}")

        module_name, (module_class, _) = module_entries[idx - 1]
        print(f"\nCalibrating module: {module_name}")

        yaml_path = os.path.join(conf_path, f"{module_name}.yaml")
        module_cfg = load_config_yaml(yaml_path, module_class.config_class)

        module = module_class(module_cfg, module_name)
        module.connect()
        module.calibrate()
