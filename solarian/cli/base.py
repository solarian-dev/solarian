from typing import Dict, Mapping, Type, Optional, Any, NamedTuple
from abc import ABC, abstractmethod
import importlib
from omegaconf import DictConfig


class _LazyCommand(NamedTuple):
    """Metadata for a command that hasn't been imported yet."""
    module_path: str  # e.g. "solarian.cli.commands.run_robot"
    help: str
    config_name: Optional[str]
    setup_modules: Optional[tuple] = None
    #: Help-section heading. Repeated from the command class so ``solarian
    #: --help`` can group commands without importing their modules.
    category: str = ""


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
                      help: str = "", config_name: str = None,
                      setup_modules: Optional[tuple] = None,
                      category: str = ""):
        """Register a command by metadata only — no import happens until the command is run.

        ``setup_modules`` controls which project-level modules (robot,
        observation, action, controller, server) are imported during
        ``setup()``.  *None* means all modules (backward-compatible
        default); an empty tuple means none.

        ``category`` is the help-section heading declared on the command
        class. It is repeated here so the command list can group commands
        without importing the module.
        """
        cls._lazy_commands[name] = _LazyCommand(
            module_path=module_path,
            help=help,
            config_name=config_name,
            setup_modules=setup_modules,
            category=category,
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
    def is_registered(cls, name: str) -> bool:
        return name in cls._commands or name in cls._lazy_commands

    @classmethod
    def get_command_help(cls, name: str) -> str:
        """Return help text without importing the module."""
        if name in cls._lazy_commands:
            return cls._lazy_commands[name].help
        if name in cls._commands:
            return getattr(cls._commands[name], 'help', '')
        return ''

    @classmethod
    def get_category(cls, name: str) -> str:
        """Section heading declared by the command.

        A loaded class wins. Otherwise the lazy record is used, so the command
        list can group a built-in before its module is imported.
        """
        command_cls = cls._commands.get(name)
        if command_cls is not None:
            category = getattr(command_cls, "category", "") or ""
            if category:
                return category
        lazy = cls._lazy_commands.get(name)
        if lazy is not None and lazy.category:
            return lazy.category
        return ""

    @classmethod
    def uses_config(cls, name: str) -> bool:
        """Whether a loaded command reads project yaml. Unloaded commands do."""
        command_cls = cls._commands.get(name)
        if command_cls is None:
            return True
        return bool(getattr(command_cls, "uses_config", True))

    @classmethod
    def get_setup_modules(cls, name: str) -> Optional[tuple]:
        """Return the setup_modules declared for *name*, or None (= all)."""
        if name in cls._lazy_commands:
            return cls._lazy_commands[name].setup_modules
        command_cls = cls._commands.get(name)
        return command_cls.setup_modules if command_cls is not None else None

class BaseCommand(ABC):
    #: One line, listed by ``solarian --help`` and repeated at the top of this
    #: command's own help.
    help = ""

    #: What the command does, printed by ``solarian <command> --help`` under
    #: ``help``: prerequisites, the flow it walks through, what it writes.
    description = ""

    #: Argument spec as the user types it, after the command name.
    usage = "[key=value ...]"

    #: Ad-hoc CLI parameters this command reads off ``cfg`` that are not keys of
    #: its yaml (``role``, ``module``, ``sweep``, ...), mapped to their help
    #: text. Hydra composes in struct mode, so ``key=value`` for a key the yaml
    #: never declares is rejected; naming it here seeds it as null so the plain
    #: form works. Undeclared names still fail, which is what catches a
    #: mistyped override.
    params: Mapping[str, str] = {}

    #: Positional arguments and flags the command parses itself rather than
    #: through Hydra, written as the user types them.
    options: Mapping[str, str] = {}

    #: Example invocations, one command line each.
    examples: tuple = ()

    #: Heading under which ``solarian --help`` lists this command. Empty falls
    #: back to Utilities. Repeated on ``register_lazy`` so the command list
    #: can group this command without importing the module.
    category = ""

    #: When False, the command runs without a project config or settings.py.
    uses_config = True

    #: When False, :meth:`handle` receives the composed config unresolved and
    #: resolves it itself, e.g. after registering resolvers only it knows.
    resolve_config = True

    #: Project modules ``setup()`` imports for this command (see
    #: ``solarian.cli.commands``); None imports all. Built-ins repeat it on
    #: ``register_lazy``.
    setup_modules: Optional[tuple] = None

    def parse_args(self, argv: list[str]) -> list[str]:
        """Parse this command's own argv and return leftover Hydra overrides.

        The default treats every token as a Hydra ``key=value`` override. A
        command with positionals or flags overrides this, stores what it
        parsed on itself, and returns an empty list.
        """
        return list(argv)

    
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
