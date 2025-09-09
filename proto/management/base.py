from typing import Dict, Type, Optional, Any
from abc import ABC, abstractmethod
from omegaconf import DictConfig

class CommandRegistry:
    _commands: Dict[str, Type['BaseCommand']] = {}
    _config_names: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, config_name: str = None):
        def decorator(command_class: Type['BaseCommand']):
            cls._commands[name] = command_class
            cls._config_names[name] = config_name
            return command_class
        return decorator

    @classmethod
    def get_command(cls, name: str) -> Type['BaseCommand']:
        return cls._commands.get(name)()

    @classmethod
    def get_config_name(cls, name: str) -> str:
        return cls._config_names.get(name)

    @classmethod
    def get_all_commands(cls) -> Dict[str, Type['BaseCommand']]:
        return cls._commands.copy()

class BaseCommand(ABC):
    help = ""  # Command help text
    
    def get_config_class(self) -> Optional[Type[Any]]:
        """
        Return the Pydantic model class for this command's configuration.
        Override this method in subclasses that need config validation.
        
        Returns:
            Optional[Type[Any]]: A Pydantic model class or None if no validation is needed
        """
        return None

    @abstractmethod
    def handle(self, cfg: DictConfig) -> None:
        """
        The actual logic of the command. Override this method.
        """
        raise NotImplementedError("Subclass must implement handle() method")
