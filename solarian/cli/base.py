from typing import Dict, Type, Optional, Any, NamedTuple
from abc import ABC, abstractmethod
import importlib
from omegaconf import DictConfig


class _LazyCommand(NamedTuple):
    """Metadata for a command that hasn't been imported yet."""
    module_path: str  # e.g. "solarian.cli.commands.run_robot"
    help: str
    config_name: Optional[str]


class CommandRegistry:
    _commands: Dict[str, Type['BaseCommand']] = {}
    _lazy_commands: Dict[str, _LazyCommand] = {}
    _config_names: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, config_name: str = None):
        """Decorator that eagerly registers a command class (used at import time)."""
        def decorator(command_class: Type['BaseCommand']):
            cls._commands[name] = command_class
            cls._config_names[name] = config_name
            return command_class
        return decorator

    @classmethod
    def register_lazy(cls, name: str, *, module_path: str,
                      help: str = "", config_name: str = None):
        """Register a command by metadata only — no import happens until the command is run."""
        cls._lazy_commands[name] = _LazyCommand(
            module_path=module_path,
            help=help,
            config_name=config_name,
        )
        cls._config_names[name] = config_name

    @classmethod
    def _resolve_lazy(cls, name: str) -> None:
        """Import a lazily-registered command module.
        
        The module's @CommandRegistry.register decorator will fire on import
        and populate _commands automatically.
        """
        if name in cls._lazy_commands and name not in cls._commands:
            lazy = cls._lazy_commands[name]
            importlib.import_module(lazy.module_path)

    @classmethod
    def get_command(cls, name: str) -> 'BaseCommand':
        cls._resolve_lazy(name)
        cmd_class = cls._commands.get(name)
        if cmd_class is None:
            raise KeyError(f"Unknown command: {name}")
        return cmd_class()

    @classmethod
    def get_config_name(cls, name: str) -> str:
        return cls._config_names.get(name)

    @classmethod
    def get_all_commands(cls) -> Dict[str, Type['BaseCommand']]:
        """Return eagerly-loaded commands only (for backward compat)."""
        return cls._commands.copy()

    @classmethod
    def get_all_command_names(cls) -> list:
        """Return names of all registered commands (eager + lazy)."""
        return sorted(set(cls._commands.keys()) | set(cls._lazy_commands.keys()))

    @classmethod
    def get_command_help(cls, name: str) -> str:
        """Return help text without importing the module."""
        if name in cls._lazy_commands:
            return cls._lazy_commands[name].help
        if name in cls._commands:
            return getattr(cls._commands[name], 'help', '')
        return ''

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
