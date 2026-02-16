from omegaconf import OmegaConf
from typing import Any
from solarian.core.configs import ModuleConfig
from pydantic import ValidationError
import os


def parse_with_friendly_errors(model_class, data):
    """Parse config data with user-friendly error messages"""
    try:
        return model_class(**data)
    except ValidationError as e:
        # Get the config filename if possible
        config_path = os.environ.get("SOLARIAN_CURRENT_CONFIG", "")
        
        # Format a user-friendly error message
        error_msg = f"Configuration Error in {model_class.__name__}:\n"
        error_msg += f"  {str(e)}\n"
        
        if config_path:
            error_msg += f"  Please check your configuration file: {config_path}\n"
            
        # Re-raise with the friendly message
        raise ValueError(error_msg) from e


def load_config_yaml(yaml_path: str, config_class: type) -> Any:
    conf = OmegaConf.load(yaml_path)
    data = OmegaConf.to_container(conf, resolve=True)
    return parse_with_friendly_errors(config_class, data)


def load_module_config_yaml(yaml_path: str) -> ModuleConfig:
    return load_config_yaml(yaml_path, ModuleConfig)


def load_module_configs_for_robot(robot_class) -> dict[str, tuple[type, ModuleConfig]]:
    """
    Load module configs from separate YAML files based on a Robot class's annotations.
    
    For each annotation on the robot class that is a Module subclass, looks for
    a corresponding {name}.yaml file in the conf directory and loads it using
    the module's own `config_class` (which must extend ModuleConfig).
    
    Args:
        robot_class: The Robot subclass (e.g. MySolarianRobot) with module annotations.
        
    Returns:
        Dict mapping module name to (module_type, config_instance) tuples.
    """
    from solarian.core.module import Module
    
    conf_path = os.environ.get("SOLARIAN_CONF_PATH", os.path.join(os.getcwd(), "conf"))
    module_configs = {}
    
    # Walk the MRO to find all Module annotations (not just the leaf class)
    seen = set()
    for cls in robot_class.__mro__:
        for name, anno in getattr(cls, '__annotations__', {}).items():
            if name in seen:
                continue
            seen.add(name)
            if isinstance(anno, type) and issubclass(anno, Module):
                yaml_path = os.path.join(conf_path, f"{name}.yaml")
                if not os.path.exists(yaml_path):
                    raise FileNotFoundError(
                        f"Module config file not found: {yaml_path}\n"
                        f"  Robot class '{robot_class.__name__}' declares module '{name}: {anno.__name__}'\n"
                        f"  Expected a config file at: {yaml_path}"
                    )
                # Use the module's own config_class (defaults to ModuleConfig)
                cfg_class = anno.config_class
                module_cfg = load_config_yaml(yaml_path, cfg_class)
                module_configs[name] = (anno, module_cfg)
    
    return module_configs

