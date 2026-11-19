#!/usr/bin/env python
"""Solarian's command-line utility for project commands."""
import sys
import os
import re
import argparse
from pathlib import Path

if sys.platform == "darwin":
    # Suppress harmless Objective-C warning when forking processes that later
    # use AVFoundation (e.g. OpenCV cameras).  The camera is only opened in
    # the child process, so the fork is safe — the runtime check is a false
    # positive.  Switching to the 'spawn' start method would also fix this but
    # requires all multiprocessing objects to be fully picklable.
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
from hydra import initialize, compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import colorama

from solarian.utils.env_file import load_env_file

from .base import CommandRegistry
from .help import GLOBAL_OPTIONS, format_command_help


def _find_settings_path(cwd: Path | None = None) -> str:
    """
    Auto-discover settings.py in the project directory.

    Search order:
      1. Look for {package_name}/settings.py  (derived from pyproject.toml name)
      2. Look for settings.py in any Python package (directory with __init__.py)
      3. Fall back to conf/settings.py  (legacy location)
    """
    if cwd is None:
        cwd = Path.cwd()

    # Try to derive package name from pyproject.toml
    pyproject = cwd / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(encoding="utf-8")
        match = re.search(r'name\s*=\s*"([^"]+)"', text)
        if match:
            package_name = match.group(1).replace("-", "_")
            candidate = cwd / package_name / "settings.py"
            if candidate.exists():
                return str(candidate)

    # Search any Python package directory for settings.py
    for child in sorted(cwd.iterdir()):
        if (
            child.is_dir()
            and (child / "__init__.py").exists()
            and (child / "settings.py").exists()
        ):
            return str(child / "settings.py")

    # Legacy fallback: conf/settings.py
    conf_settings = cwd / "conf" / "settings.py"
    if conf_settings.exists():
        return str(conf_settings)

    raise FileNotFoundError(
        "Could not find settings.py. "
        "Place it in your project package directory (e.g. my_project/settings.py) "
        "or in conf/settings.py."
    )


#: Entry-point group that plugins use to advertise extra CLI commands.
COMMAND_ENTRY_POINT_GROUP = "solarian.commands"

#: Help-section order. A heading a plugin declares that is not in this list
#: is appended after these.
_CATEGORY_ORDER = (
    "Project",
    "Robot",
    "Controller",
    "Recording",
    "Training & Data",
    "Utilities",
)


def _get_version() -> str:
    """Return the installed solarian version, or "unknown" outside an install."""
    import importlib.metadata

    try:
        return importlib.metadata.version("solarian")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _iter_entry_points(group: str):
    """Yield entry points for *group* across importlib.metadata API versions."""
    import importlib.metadata

    try:
        eps = importlib.metadata.entry_points()
    except Exception:  # noqa: BLE001 - metadata access can fail in odd envs
        return []
    # Python 3.10+: EntryPoints.select; older: dict-like .get
    if hasattr(eps, "select"):
        return list(eps.select(group=group))
    return list(eps.get(group, []))  # type: ignore[attr-defined]


def _register_plugin_commands() -> None:
    """Import plugin commands so their classes can declare category and config use.

    The module is imported, not merely named. ``category`` and ``uses_config``
    live on the command class; a new plugin command does not edit this file.
    """
    import importlib

    for ep in _iter_entry_points(COMMAND_ENTRY_POINT_GROUP):
        module_path, _, attr = ep.value.partition(":")
        if not module_path:
            continue
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001 - one bad plugin must not hide the others
            print(f"warning: could not load plugin command {ep.name}: {exc}", file=sys.stderr)
            continue
        if CommandRegistry.is_registered(ep.name):
            continue
        command_cls = getattr(module, attr, None) if attr else None
        if command_cls is None:
            print(f"warning: plugin command {ep.name} does not define {attr or 'a class'}", file=sys.stderr)
            continue
        CommandRegistry.register(ep.name)(command_cls)


class SolarianHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom formatter that adjusts help layout for the solarian CLI."""
    pass


def _build_command_list() -> str:
    """Build a formatted string listing all available commands.

    Each command names its section via ``category`` (on the class, or on the
    lazy registration when the module has not been imported yet).
    """
    all_names = CommandRegistry.get_all_command_names()
    if not all_names:
        return "  (no commands registered)"

    grouped: dict[str, list[tuple[str, str]]] = {}
    for name in all_names:
        help_text = CommandRegistry.get_command_help(name)
        category = CommandRegistry.get_category(name) or "Utilities"
        grouped.setdefault(category, []).append((name, help_text))

    ordered = [category for category in _CATEGORY_ORDER if category in grouped]
    ordered += [category for category in grouped if category not in _CATEGORY_ORDER]

    lines = []
    for category in ordered:
        cmds = grouped[category]
        lines.append(f"  {category}:")
        max_name_len = max(len(name) for name, _ in cmds)
        for name, help_text in cmds:
            lines.append(f"    {name:<{max_name_len}}  {help_text}")
        lines.append("")

    return "\n".join(lines)


# Only main prints this. A command that printed it again would show it twice.
BANNER = """\
█▀▀ █▀█ █   █▀█ █▀█ █ █▀█ █▄ █
▀▀█ █ █ █   █▀█ █▀▄ █ █▀█ █ ▀█
▀▀▀ ▀▀▀ ▀▀▀ ▀ ▀ ▀ ▀ ▀ ▀ ▀ ▀  ▀
"""


def main() -> None:
    """
    Given the command-line arguments, figure out which subcommand is being
    run, create a parser appropriate to that command, and run it.
    """
    # Piped stdout is read by a program (a job runner, an agent), which wants
    # only what the command prints.
    if sys.stdout.isatty():
        print(f"\n{BANNER}")

    # Register command metadata (lazy — no heavy imports happen here)
    from ..cli import commands  # noqa: F401

    # Discover plugin-provided commands advertised under the
    # "solarian.commands" entry-point group.  Only metadata is registered
    # here — the command module is imported lazily when the command runs.
    _register_plugin_commands()

    available_commands = CommandRegistry.get_all_command_names()

    epilog = (
        "available commands:\n"
        + _build_command_list()
        + "\nRun `solarian <command> --help` for that command's arguments and examples.\n"
        + "\ncoding agents:\n"
        + "  Inside a project or plugin, read AGENTS.md first. Docs for this version: `solarian docs list`.\n"
        + "  New project: `solarian create-project <name>`. New plugin: `solarian create-plugin <type> <name>`.\n"
        + "  Online docs index: https://solarian.dev/llms.txt\n"
    )

    parser = argparse.ArgumentParser(
        prog="solarian",
        description="Solarian's command-line utility for robotics projects.",
        epilog=epilog,
        formatter_class=SolarianHelpFormatter,
    )
    parser.add_argument(
        "subcommand",
        type=str,
        choices=available_commands,
        metavar="subcommand",
        help="The subcommand to run (see list below).",
    )
    for flags, kwargs in GLOBAL_OPTIONS:
        parser.add_argument(*flags, **kwargs)
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {_get_version()}",
                        help="Print the installed solarian version and exit.")
    
    # If no arguments provided, print help and exit
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    # `solarian <command> --help` describes that command. Answered before
    # argparse sees the flag, which would otherwise print the root help.
    if sys.argv[1] in available_commands and {"-h", "--help"}.intersection(sys.argv[2:]):
        # Loading the command is what makes uses_config readable: it lives on
        # the class, and built-ins are registered by name until then.
        help_cmd = CommandRegistry.get_command(sys.argv[1])
        print(
            format_command_help(
                sys.argv[1],
                help_cmd,
                config_name=CommandRegistry.get_config_name(sys.argv[1]),
                uses_config=CommandRegistry.uses_config(sys.argv[1]),
            ),
            end="",
        )
        sys.exit(0)

    args, remaining = parser.parse_known_args()

    try:
        cmd = CommandRegistry.get_command(args.subcommand)
    except KeyError:
        print(f"Unknown command: {args.subcommand}")
        sys.exit(1)

    # Leftover tokens are Hydra overrides. A command that parses its own argv
    # returns an empty list.
    overrides = cmd.parse_args(remaining)

    # The first component is the conf directory; the rest names a config in
    # it, folders included: conf/training/umi_pick composes training/umi_pick.
    if '/' in args.conf:
        CONF_PATH, CONF_NAME = args.conf.split("/", 1)
        CONF_NAME = CONF_NAME.removesuffix(".yaml")
    else:
        CONF_PATH = args.conf
        CONF_NAME = "default"

    subcommand = args.subcommand

    # Resolve settings.py location (skip for commands that don't need it)
    if CommandRegistry.uses_config(subcommand):
        if args.settings:
            settings_path = str(Path(os.getcwd()) / args.settings)
        else:
            settings_path = _find_settings_path()

        os.environ["SOLARIAN_SETTINGS"] = settings_path

    os.environ["SOLARIAN_PROJECT_PATH"] = os.getcwd()
    # Project secrets (model API keys, Onshape keys). Exported, so skill and
    # service child processes see them too.
    if Path(".env").is_file():
        load_env_file(".env")

    # Add project path to sys.path
    sys.path.insert(0, os.getcwd())

    # make conf path absolute
    CONF_PATH =  os.getcwd() + "/" + CONF_PATH
    os.environ["SOLARIAN_CONF_PATH"] = CONF_PATH

    # Exported (not just passed around) so skill child processes inherit the
    # deployment the run was launched with.
    if args.deployment:
        os.environ["SOLARIAN_DEPLOYMENT"] = args.deployment

    colorama.init()  # Initialize colorama for colored terminal output
    if not CommandRegistry.uses_config(subcommand):
        # For commands that don't need Hydra, just call handle with empty config
        try:
            cmd.handle(OmegaConf.create({}))
        except Exception as e:
            print(f"\n{colorama.Fore.RED}{str(e)}{colorama.Style.RESET_ALL}")
            sys.exit(1)
    else:
        # Commands that need Hydra config
        if CONF_NAME == "default":
            config_name = CommandRegistry.get_config_name(subcommand)
        else:
            config_name = CONF_NAME

        # Hydra nests a config named training/umi_pick under a training: key.
        # Composing it from its folder keeps its keys at the top level; the conf
        # root stays on the search path for defaults such as /controller@_global_.
        config_dir, searchpath = CONF_PATH, []
        if config_name is not None and "/" in config_name:
            folder, config_name = config_name.rsplit("/", 1)
            config_dir = os.path.join(CONF_PATH, folder)
            searchpath = [f"hydra.searchpath=['file://{CONF_PATH}']"]

        # try:
        with initialize_config_dir(version_base=None, config_dir=config_dir, job_name='solarian_app'):
            # Store the current config path for error reporting
            if config_name is not None:
                config_file = Path(config_dir) / f"{config_name}.yaml"
                if config_file.exists():
                    os.environ["SOLARIAN_CURRENT_CONFIG"] = str(config_file.absolute())
                
            # Seed the command's declared ad-hoc params as null so Hydra's
            # struct mode accepts `role=display` without a `+` prefix. Seeds go
            # first, so a user override still wins; names the yaml already
            # declares are left alone.
            seeds = []
            if cmd.params:
                base_cfg = compose(config_name=config_name, overrides=searchpath)
                seeds = [f"+{name}=null" for name in cmd.params if name not in base_cfg]

            # Load the config using hydra
            hydra_cfg = compose(config_name=config_name, overrides=searchpath + seeds + overrides)

            from solarian.utils.configs import ConfigError

            # Module configs are loaded while the command runs, so their
            # errors surface from handle(), not from parsing the command's own.
            try:
                if not cmd.resolve_config:
                    cfg = hydra_cfg
                else:
                    # Convert to dict for Pydantic parsing, resolving ${deployment.*}
                    from solarian.utils.configs import parse_with_friendly_errors, resolve_with_deployment
                    cfg_dict = resolve_with_deployment(hydra_cfg)
                    if cmd.get_config_class() is not None:
                        cfg = parse_with_friendly_errors(cmd.get_config_class(), cfg_dict)
                    else:
                        cfg = OmegaConf.create(cfg_dict)

                from solarian.setup import setup
                setup_modules = CommandRegistry.get_setup_modules(subcommand)
                setup(modules=setup_modules)

                cmd.handle(cfg)
            except KeyboardInterrupt:
                # Ctrl+C is how every long-running command is meant to end, and
                # commands that have shutdown work to do catch it themselves.
                # One that arrives during that shutdown lands here instead, and
                # a stack trace is a poor way to report a clean stop.
                print()
            except ConfigError as e:
                if os.environ.get("SOLARIAN_DEBUG"):
                    raise
                print(f"\n{colorama.Fore.RED}{e}{colorama.Style.RESET_ALL}", file=sys.stderr)
                sys.exit(1)


if __name__ == "__main__":
    main()
