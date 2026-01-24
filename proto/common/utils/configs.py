from omegaconf import OmegaConf
from typing import Any
from proto.common.configs import ModuleConfig
from pydantic import ValidationError
import os


def parse_with_friendly_errors(model_class, data):
    """Parse config data with user-friendly error messages"""
    try:
        return model_class(**data)
    except ValidationError as e:
        # Get the config filename if possible
        config_path = os.environ.get("PROTO_CURRENT_CONFIG", "")
        
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
