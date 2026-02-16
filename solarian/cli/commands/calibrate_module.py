import time
import serial.tools.list_ports
from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.utils import get_class
from solarian.core.module_bus import ModuleBus
from solarian.core.configs import RobotConfig


@CommandRegistry.register('calibrate-module', 'robot')
class CalibrateModuleCommand(BaseCommand):
    help = 'Calibrate module device.'

    def get_config_class(self):
        """Return the config class for this command."""
        return RobotConfig

    def handle(self, cfg: RobotConfig) -> None:
        from solarian.conf import settings
        
        # List all available modules
        print("\n=== Available Modules ===")
        if not hasattr(settings, 'MODULE_CLASSES') or not settings.MODULE_CLASSES:
            print("No modules are configured in settings.MODULE_CLASSES")
            return
        
        module_names = list(settings.MODULE_CLASSES.keys())
        for idx, module_name in enumerate(module_names, 1):
            module_class_path = settings.MODULE_CLASSES[module_name]
            print(f"{idx}. {module_name} ({module_class_path})")
        
        # Ask user to select a module
        print("\n=== Select Module to Setup ===")
        while True:
            user_input = input(f"Enter module number (1-{len(module_names)}): ").strip()
            
            # Check if input is a number
            if user_input.isdigit():
                idx = int(user_input)
                if 1 <= idx <= len(module_names):
                    module_name = module_names[idx - 1]
                    break
                else:
                    print(f"Invalid number. Please enter a number between 1 and {len(module_names)}")
            else:
                print(f"Invalid module number. Please enter a number between 1 and {len(module_names)}")
        
        print(f"\nCalibrating module: {idx}")
        
        # Proceed with module setup

        # create module instance
        module_name = module_names[idx - 1]
        module_class_path = settings.MODULE_CLASSES[module_name]
        module_class = get_class(module_class_path)
        component_bus = ModuleBus()
        module = module_class(cfg.modules[module_name], module_name, component_bus)
        component_bus.connect()
        module.calibrate()
        
    
    