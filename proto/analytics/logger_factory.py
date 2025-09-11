from enum import Enum
import os
from pathlib import Path

from proto.common.configs import LoggerConfig, WandBLoggerConfig
from proto.analytics.wandb_logger import WandBLogger
from proto.analytics.logger import Logger


class LoggerEnum(str, Enum):
    WANDB = "wandb"

loggers = {
    LoggerEnum.WANDB.value: WandBLogger,
}

class LoggerFactory():
    
    @staticmethod
    def create_logger(logger_cfg: LoggerConfig, **kwargs) -> Logger:
        """
        Create a logger instance based on configuration.
        
        Args:
            logger_cfg: The logger configuration
            **kwargs: Additional arguments to pass to the logger constructor
            
        Returns:
            Logger instance
            
        Raises:
            ValueError: If the logger type is not implemented or the configuration is invalid
        """
        # Validation is already handled by Pydantic models at this point
        try:
            if logger_cfg.type not in loggers:
                raise ValueError(f"Logger type '{logger_cfg.type}' is not implemented.")
                
            # Let the specific logger implementation handle additional validation
            return loggers[logger_cfg.type](logger_cfg, **kwargs)
            
        except Exception as e:
            # Handle non-validation errors
            if not isinstance(e, ValueError):
                # Wrap other exceptions with helpful context
                raise ValueError(f"Error creating logger: {str(e)}") from e
            # Re-raise ValueError exceptions
            raise