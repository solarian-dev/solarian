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

#: Built-in commands that take no Hydra config and need no project
#: ``settings.py``. Plugin commands declare this on the class instead
#: (``uses_config = False``). The global ``--conf`` / ``--deployment`` flags
#: do nothing for these, so their help leaves those out.
NO_HYDRA_COMMANDS = {
    "install",
    "create-project",
    "create-plugin",
    "list-cameras",
    "detect-device",
    "run-policy",
    "onshape-export",
    "export-dataset",
    "keygen",
    "calibrate-servo",
}


def _uses_config(name: str) -> bool:
    """Project yaml applies unless the command is a built-in exception or says otherwise."""
    if name in NO_HYDRA_COMMANDS:
        return False
    return CommandRegistry.uses_config(name)


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

    Built-in commands are grouped from the map below. A plugin command that
    has been imported contributes the ``category`` declared on its class.
    """
    all_names = CommandRegistry.get_all_command_names()
    if not all_names:
        return "  (no commands registered)"

    # Group commands by category
    categories = {
        "Project": [],
        "Robot": [],
        "Controller": [],
        "Recording & Replay": [],
        "Training & Data": [],
        "Utilities": [],
    }

    category_map = {
        "create-project": "Project",
        "create-plugin": "Project",
        "install": "Project",
        "onshape-export": "Utilities",
        "run-robot": "Robot",
        "calibrate": "Robot",
        "calibrate-arm": "Robot",
        "calibrate-gripper": "Robot",
        "calibrate-gripper-ranges": "Robot",
        "calibrate-camera-latency": "Robot",
        "calibrate-hand-eye": "Robot",
        "setup-module": "Robot",
        "calibrate-servo": "Robot",
        "write-robot-calibration": "Robot",
        "verify-calibration": "Robot",
        "derive-model-limits": "Robot",
        "run-controller": "Controller",
        "calibrate-controller": "Controller",
        "calibrate-umi-camera": "Controller",
        "write-controller-calibration": "Controller",
        "run-policy": "Controller",
        "run-server": "Controller",
        "record": "Recording & Replay",
        "replay": "Recording & Replay",
        "export-dataset": "Training & Data",
        "train": "Training & Data",
        "detect-device": "Utilities",
        "list-cameras": "Utilities",
        "shell": "Utilities",
    }

    for name in all_names:
        help_text = CommandRegistry.get_command_help(name)
        # A loaded plugin command names its own section. Built-ins use the map.
        category = CommandRegistry.get_category(name) or category_map.get(name, "Utilities")
        categories.setdefault(category, []).append((name, help_text))

    lines = []
    for category, cmds in categories.items():
        if not cmds:
            continue
        lines.append(f"  {category}:")
        max_name_len = max(len(name) for name, _ in cmds)
        for name, help_text in cmds:
            lines.append(f"    {name:<{max_name_len}}  {help_text}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """
    Given the command-line arguments, figure out which subcommand is being
    run, create a parser appropriate to that command, and run it.
    """

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
        print(
            format_command_help(
                sys.argv[1],
                CommandRegistry.get_command(sys.argv[1]),
                config_name=CommandRegistry.get_config_name(sys.argv[1]),
                uses_config=_uses_config(sys.argv[1]),
            ),
            end="",
        )
        sys.exit(0)

    # Handle positional arguments for specific commands
    args, remaining = parser.parse_known_args()
    
    if args.subcommand == "install":
        positionals = [a for a in remaining if not a.startswith("-")]
        if len(positionals) < 1:
            parser.error("install requires a package name or a local path")
        os.environ["SOLARIAN_INSTALL_TARGET"] = positionals[0]
        overrides = []
    elif args.subcommand == "onshape-export":
        oe_parser = argparse.ArgumentParser(prog="solarian onshape-export")
        oe_parser.add_argument("assembly_dir", help="Directory containing the assembly config.json.")
        oe_parser.add_argument("--clean-only", action="store_true",
                               help="Re-run only the clean stage on a previous raw export (no network).")
        oe_parser.add_argument("--keys-file", default=None,
                               help="File with ONSHAPE_ACCESS_KEY/ONSHAPE_SECRET_KEY (default: onshape_keys.txt).")
        oe_parser.add_argument("--update-yaml", default=None, metavar="PATH",
                               help="After export, sync MJCF joint ranges into this module conf "
                                    "(sim_range_min/max) and config.json joint_properties.limits.")
        oe_args = oe_parser.parse_args(remaining)
        os.environ["SOLARIAN_ONSHAPE_EXPORT_DIR"] = oe_args.assembly_dir
        if oe_args.clean_only:
            os.environ["SOLARIAN_ONSHAPE_EXPORT_CLEAN_ONLY"] = "true"
        if oe_args.keys_file:
            os.environ["SOLARIAN_ONSHAPE_EXPORT_KEYS_FILE"] = oe_args.keys_file
        if oe_args.update_yaml:
            os.environ["SOLARIAN_ONSHAPE_EXPORT_UPDATE_YAML"] = oe_args.update_yaml
        overrides = []
    elif args.subcommand == "export-dataset":
        ed_parser = argparse.ArgumentParser(prog="solarian export-dataset")
        ed_parser.add_argument("session_dir", help="Recorded session to convert.")
        ed_parser.add_argument("output_dir", nargs="?", default=None,
                               help="Where to write (default: <session_dir>/export/lerobot).")
        ed_parser.add_argument("--fps", default=None,
                               help="Rate to resample onto (default: the session's video grid).")
        ed_parser.add_argument("--episodes", default=None, metavar="A,B,C",
                               help="Only these episode indices (default: all sealed ones).")
        ed_parser.add_argument("--no-remap", action="store_true",
                               help="Keep the archive geometry instead of producing policy frames.")
        ed_parser.add_argument("--overwrite", action="store_true",
                               help="Replace the output directory if it exists.")
        ed_args = ed_parser.parse_args(remaining)
        os.environ["SOLARIAN_EXPORT_SESSION"] = ed_args.session_dir
        if ed_args.output_dir:
            os.environ["SOLARIAN_EXPORT_OUT"] = ed_args.output_dir
        if ed_args.fps:
            os.environ["SOLARIAN_EXPORT_FPS"] = ed_args.fps
        if ed_args.episodes:
            os.environ["SOLARIAN_EXPORT_EPISODES"] = ed_args.episodes
        if ed_args.no_remap:
            os.environ["SOLARIAN_EXPORT_NO_REMAP"] = "true"
        if ed_args.overwrite:
            os.environ["SOLARIAN_EXPORT_OVERWRITE"] = "true"
        overrides = []
    elif args.subcommand == "create-project":
        if len(remaining) < 1:
            parser.error("create-project requires a project name argument")
        os.environ["SOLARIAN_CREATE_PROJECT_NAME"] = remaining[0]
        if "--force" in remaining:
            os.environ["SOLARIAN_CREATE_PROJECT_FORCE"] = "true"
        overrides = []
    elif args.subcommand == "create-plugin":
        positionals = [a for a in remaining if not a.startswith("-")]
        if len(positionals) < 2:
            parser.error("create-plugin requires a type and a name argument")
        os.environ["SOLARIAN_CREATE_PLUGIN_TYPE"] = positionals[0]
        os.environ["SOLARIAN_CREATE_PLUGIN_NAME"] = positionals[1]
        if "--force" in remaining:
            os.environ["SOLARIAN_CREATE_PLUGIN_FORCE"] = "true"
        overrides = []
    elif args.subcommand == "run-policy":
        rp_parser = argparse.ArgumentParser(prog="solarian run-policy")
        rp_parser.add_argument("skill", help="Name of the skill whose policy server to run.")
        rp_parser.add_argument("--config", default=None,
                               help="Skill yaml (a conf filename or a path). "
                                    "UMISkill reads policy_server from this file "
                                    "instead of conf/umi.yaml.")
        rp_parser.add_argument("--checkpoint", "-c", default=None,
                               help="Checkpoint path (overrides the skill's configured checkpoint).")
        rp_parser.add_argument("--host", default=None, help="Host to bind the server to.")
        rp_parser.add_argument("--port", default=None, help="Port to bind the server to.")
        rp_args = rp_parser.parse_args(remaining)
        os.environ["SOLARIAN_RUN_POLICY_SKILL"] = rp_args.skill
        if rp_args.config:
            os.environ["SOLARIAN_RUN_POLICY_CONFIG"] = rp_args.config
        if rp_args.checkpoint:
            os.environ["SOLARIAN_RUN_POLICY_CHECKPOINT"] = rp_args.checkpoint
        if rp_args.host:
            os.environ["SOLARIAN_RUN_POLICY_HOST"] = rp_args.host
        if rp_args.port:
            os.environ["SOLARIAN_RUN_POLICY_PORT"] = rp_args.port
        overrides = []
    else:
        overrides = remaining

    if '/' in args.conf:
        CONF_PATH = args.conf.split("/")[0]
        CONF_NAME = args.conf.split("/")[1]
    else:
        CONF_PATH = args.conf
        CONF_NAME = "default"

    subcommand = args.subcommand

    # Resolve settings.py location (skip for commands that don't need it)
    if _uses_config(subcommand):
        if args.settings:
            settings_path = str(Path(os.getcwd()) / args.settings)
        else:
            settings_path = _find_settings_path()

        os.environ["SOLARIAN_SETTINGS"] = settings_path

    os.environ["SOLARIAN_PROJECT_PATH"] = os.getcwd()

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
    try:
        cmd = CommandRegistry.get_command(subcommand)
    except KeyError:
        print(f"Unknown command: {subcommand}")
        sys.exit(1)
    
    if not _uses_config(subcommand):
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

        # try:
        with initialize_config_dir(version_base=None, config_dir=CONF_PATH, job_name='solarian_app'):
            # Store the current config path for error reporting
            if config_name is not None:
                config_file = Path(CONF_PATH) / config_name
                if config_file.exists():
                    os.environ["SOLARIAN_CURRENT_CONFIG"] = str(config_file.absolute())
                
            # Seed the command's declared ad-hoc params as null so Hydra's
            # struct mode accepts `role=display` without a `+` prefix. Seeds go
            # first, so a user override still wins; names the yaml already
            # declares are left alone.
            seeds = []
            if cmd.params:
                base_cfg = compose(config_name=config_name, overrides=[])
                seeds = [f"+{name}=null" for name in cmd.params if name not in base_cfg]

            # Load the config using hydra
            hydra_cfg = compose(config_name=config_name, overrides=seeds + overrides)
            
            # Convert to dict for Pydantic parsing, resolving ${deployment.*}
            from solarian.utils.configs import resolve_with_deployment
            cfg_dict = resolve_with_deployment(hydra_cfg)
            
            # Parse using the command's config class which now has built-in validation
            if cmd.get_config_class() is not None:
                from solarian.utils.configs import parse_with_friendly_errors
                cfg = parse_with_friendly_errors(cmd.get_config_class(), cfg_dict)
            else:
                cfg = OmegaConf.create(cfg_dict)
                
            from solarian.setup import setup
            setup_modules = CommandRegistry.get_setup_modules(subcommand)
            setup(modules=setup_modules)

            try:
                cmd.handle(cfg)
            except KeyboardInterrupt:
                # Ctrl+C is how every long-running command is meant to end, and
                # commands that have shutdown work to do catch it themselves.
                # One that arrives during that shutdown lands here instead, and
                # a stack trace is a poor way to report a clean stop.
                print()
        # except ValueError as e:
        #     # Handle configuration validation errors
        #     print(f"\n{colorama.Fore.RED}{str(e)}{colorama.Style.RESET_ALL}")
        #     sys.exit(1)


if __name__ == "__main__":
    main()
