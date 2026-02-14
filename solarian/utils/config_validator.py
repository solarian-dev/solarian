from typing import Dict, Any, Optional, List, Set, Union, Type
import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

class ConfigValidationError(Exception):
    """Exception raised when a configuration is invalid."""
    def __init__(self, message, field=None, file_path=None):
        self.message = message
        self.field = field
        self.file_path = file_path
        super().__init__(self.format_message())
        
    def format_message(self) -> str:
        """Format the error message with helpful context."""
        parts = ["Configuration Error:"]
        parts.append(f"  {self.message}")
        
        if self.field:
            parts.append(f"  Missing or invalid field: '{self.field}'")
            
        if self.file_path:
            parts.append(f"  Please check your configuration file: {self.file_path}")
            
        return "\n".join(parts)


class ConfigValidator:
    """Utility to validate configuration objects and provide user-friendly error messages."""
    
    @staticmethod
    def validate_field_exists(config: DictConfig, field_path: str, 
                             file_path: Optional[str] = None, 
                             message: Optional[str] = None) -> Any:
        """
        Validate that a field exists in the configuration and return its value.
        
        Args:
            config: The configuration object to validate
            field_path: Dot-separated path to the field (e.g., "logger.run_id")
            file_path: Optional path to the config file for error reporting
            message: Optional custom error message
            
        Returns:
            The value of the requested field if it exists
            
        Raises:
            ConfigValidationError: If the field doesn't exist
        """
        try:
            # Use OmegaConf's select to safely navigate the config
            parts = field_path.split('.')
            current = config
            
            # Navigate through the parts of the path
            for i, part in enumerate(parts):
                if part not in current:
                    path_so_far = '.'.join(parts[:i+1])
                    raise KeyError(f"Key '{part}' not found in '{'.'.join(parts[:i])}'")
                current = current[part]
                
            return current
        except Exception as e:
            default_message = f"Required configuration field '{field_path}' is missing"
            raise ConfigValidationError(
                message=message or default_message,
                field=field_path,
                file_path=file_path
            ) from e
    
    @staticmethod
    def validate_fields(config: DictConfig, 
                       required_fields: List[str], 
                       file_path: Optional[str] = None) -> None:
        """
        Validate that all required fields exist in the configuration.
        
        Args:
            config: The configuration object to validate
            required_fields: List of dot-separated paths to required fields
            file_path: Optional path to the config file for error reporting
            
        Raises:
            ConfigValidationError: If any required field doesn't exist
        """
        missing_fields = []
        
        for field in required_fields:
            try:
                ConfigValidator.validate_field_exists(config, field, file_path)
            except ConfigValidationError:
                missing_fields.append(field)
                
        if missing_fields:
            fields_str = ', '.join([f"'{f}'" for f in missing_fields])
            raise ConfigValidationError(
                message=f"Missing required configuration fields: {fields_str}",
                file_path=file_path
            )
    
    @staticmethod
    def get_config_file_path(config_name: str, base_path: Optional[str] = None) -> str:
        """
        Get the absolute path to a configuration file.
        
        Args:
            config_name: Name of the configuration
            base_path: Optional base directory for configuration files
            
        Returns:
            Absolute path to the configuration file
        """
        if base_path is None:
            base_path = os.environ.get("SOLARIAN_PROJECT_PATH", "")
            
        # Check common config locations
        potential_paths = [
            Path(base_path) / "conf" / f"{config_name}.yaml",
            Path(base_path) / "conf" / config_name / "config.yaml",
            Path(base_path) / config_name / "conf" / "config.yaml",
        ]
        
        for path in potential_paths:
            if path.exists():
                return str(path)
                
        return ""
