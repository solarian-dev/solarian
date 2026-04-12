from typing import Any, Dict, Optional, Union
from omegaconf import DictConfig
import os
from pathlib import Path

from solarian.utils.config_validator import ConfigValidationError


def safe_get(config: DictConfig, key_path: str, default: Any = None, 
             required: bool = False, config_path: Optional[str] = None) -> Any:
    """
    Safely access a field in an OmegaConf config with user-friendly error handling.
    
    Args:
        config: The configuration object
        key_path: Dot-separated path to the configuration field
        default: Default value to return if the field doesn't exist and not required
        required: Whether the field is required (will raise error if missing)
        config_path: Path to the config file for error reporting
        
    Returns:
        The value at the specified key_path or the default
        
    Raises:
        ConfigValidationError: If the field is required but doesn't exist
    """
    if config is None:
        if required:
            raise ConfigValidationError("Configuration object is None", field=key_path, file_path=config_path)
        return default

    try:
        # Navigate through the nested structure
        parts = key_path.split('.')
        current = config
        
        for i, part in enumerate(parts):
            if part not in current:
                if required:
                    path_so_far = '.'.join(parts[:i])
                    raise ConfigValidationError(
                        f"Required field '{part}' not found in '{path_so_far}'", 
                        field=key_path, 
                        file_path=config_path
                    )
                return default
            current = current[part]
            
        return current
    except Exception as e:
        if required:
            if isinstance(e, ConfigValidationError):
                raise
            raise ConfigValidationError(
                f"Error accessing config field '{key_path}': {str(e)}", 
                field=key_path, 
                file_path=config_path
            ) from e
        return default


def get_config_path_from_env() -> str:
    """Get the path to the current config file from environment variables."""
    project_path = os.environ.get("SOLARIAN_PROJECT_PATH", "")
    conf_path = os.environ.get("SOLARIAN_CONF_PATH", "conf")
    
    if not project_path:
        return ""
        
    # Try to determine the config path
    potential_paths = [
        Path(project_path) / conf_path / "config.yaml",
        Path(project_path) / conf_path / "train.yaml",
        Path(project_path) / conf_path / "controller.yaml",
    ]
    
    for path in potential_paths:
        if path.exists():
            return str(path)
            
    return ""


def get_requester_info() -> tuple[str, int]:
    """
    Get information about the file and line that requested the config value.
    
    Returns:
        Tuple of (filename, line_number)
    """
    import inspect
    stack = inspect.stack()
    # Look for the first frame that isn't in this module
    for frame in stack[1:]:  # Skip this function's frame
        if frame.filename != __file__:
            return frame.filename, frame.lineno
    return "", 0
