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

#: Plugin commands that operate directly on hardware and therefore must run
#: without Hydra config or project ``settings.py`` discovery.
PLUGIN_NO_HYDRA_COMMANDS = {"feetech-set-id"}


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
    """Register commands advertised by installed plugins (lazy — no imports)."""
    for ep in _iter_entry_points(COMMAND_ENTRY_POINT_GROUP):
        try:
            module_path = ep.value.split(":", 1)[0]
        except Exception:  # noqa: BLE001 - a malformed entry point must not crash the CLI
            continue
        CommandRegistry.register_lazy(
            ep.name,
            module_path=module_path,
            help=f"{ep.name} (provided by plugin)",
        )


class SolarianHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom formatter that adjusts help layout for the solarian CLI."""
    pass


def _build_command_list() -> str:
    """Build a formatted string listing all available commands (without importing them)."""
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
        "add-module": "Project",
        "onshape-export": "Utilities",
        "run-robot": "Robot",
        "calibrate-module": "Robot",
        "calibrate-gripper": "Robot",
        "setup-module": "Robot",
        "write-robot-calibration": "Robot",
        "run-controller": "Controller",
        "calibrate-controller": "Controller",
        "write-controller-calibration": "Controller",
        "run-policy": "Controller",
        "run-server": "Controller",
        "record": "Recording & Replay",
        "replay": "Recording & Replay",
        "train": "Training & Data",
        "dataset2lerobot": "Training & Data",
        "detect-device": "Utilities",
        "list-cameras": "Utilities",
        "shell": "Utilities",
        "feetech-set-id": "Robot",
    }

    for name in all_names:
        help_text = CommandRegistry.get_command_help(name)
        category = category_map.get(name, "Utilities")
        categories[category].append((name, help_text))

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

    epilog = "available commands:\n" + _build_command_list()

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
    parser.add_argument("--conf", type=str, default="conf", help="The conf directory with yaml config files to use.")
    parser.add_argument("--settings", type=str, default=None, help="Path to settings.py (auto-discovered if not given).")
    
    # If no arguments provided, print help and exit
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    # Handle positional arguments for specific commands
    args, remaining = parser.parse_known_args()
    
    # For add-module command, extract positional arguments
    if args.subcommand == "add-module":
        if len(remaining) < 2:
            parser.error("add-module requires module_name and module_id arguments")
        module_name = remaining[0]
        module_id = remaining[1]
        overrides = remaining[2:] if len(remaining) > 2 else []
        # Store positional args in environment for the command to access
        os.environ["SOLARIAN_ADD_MODULE_NAME"] = module_name
        os.environ["SOLARIAN_ADD_MODULE_ID"] = module_id
    elif args.subcommand == "onshape-export":
        oe_parser = argparse.ArgumentParser(prog="solarian onshape-export")
        oe_parser.add_argument("assembly_dir", help="Directory containing the assembly config.json.")
        oe_parser.add_argument("--clean-only", action="store_true",
                               help="Re-run only the clean stage on a previous raw export (no network).")
        oe_parser.add_argument("--keys-file", default=None,
                               help="File with ONSHAPE_ACCESS_KEY/ONSHAPE_SECRET_KEY (default: onshape_keys.txt).")
        oe_parser.add_argument("--update-yaml", default=None, metavar="PATH",
                               help="After export, sync MJCF joint ranges into this module conf "
                                    "(sim.range_min/max) and config.json joint_properties.limits.")
        oe_args = oe_parser.parse_args(remaining)
        os.environ["SOLARIAN_ONSHAPE_EXPORT_DIR"] = oe_args.assembly_dir
        if oe_args.clean_only:
            os.environ["SOLARIAN_ONSHAPE_EXPORT_CLEAN_ONLY"] = "true"
        if oe_args.keys_file:
            os.environ["SOLARIAN_ONSHAPE_EXPORT_KEYS_FILE"] = oe_args.keys_file
        if oe_args.update_yaml:
            os.environ["SOLARIAN_ONSHAPE_EXPORT_UPDATE_YAML"] = oe_args.update_yaml
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
        rp_parser.add_argument("--checkpoint", "-c", default=None,
                               help="Checkpoint path (overrides the skill's configured checkpoint).")
        rp_parser.add_argument("--host", default=None, help="Host to bind the server to.")
        rp_parser.add_argument("--port", default=None, help="Port to bind the server to.")
        rp_args = rp_parser.parse_args(remaining)
        os.environ["SOLARIAN_RUN_POLICY_SKILL"] = rp_args.skill
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

    # Commands that don't need Hydra config or project settings
    no_hydra_commands = ['add-module', 'create-project', 'create-plugin', 'list-cameras', 'detect-device', 'run-policy', 'onshape-export']
    no_hydra_commands += list(PLUGIN_NO_HYDRA_COMMANDS)

    # Resolve settings.py location (skip for commands that don't need it)
    if subcommand not in no_hydra_commands:
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

    colorama.init()  # Initialize colorama for colored terminal output
    try:
        cmd = CommandRegistry.get_command(subcommand)
    except KeyError:
        print(f"Unknown command: {subcommand}")
        sys.exit(1)
    
    if subcommand in no_hydra_commands:
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
                
            # Load the config using hydra
            hydra_cfg = compose(config_name=config_name, overrides=overrides)
            
            # Convert to dict for Pydantic parsing
            cfg_dict = OmegaConf.to_container(hydra_cfg, resolve=True)
            
            # Parse using the command's config class which now has built-in validation
            if cmd.get_config_class() is not None:
                from solarian.utils.configs import parse_with_friendly_errors
                cfg = parse_with_friendly_errors(cmd.get_config_class(), cfg_dict)
            else:
                cfg = hydra_cfg
                
            from solarian.setup import setup
            setup_modules = CommandRegistry.get_setup_modules(subcommand)
            setup(modules=setup_modules)

            cmd.handle(cfg)
        # except ValueError as e:
        #     # Handle configuration validation errors
        #     print(f"\n{colorama.Fore.RED}{str(e)}{colorama.Style.RESET_ALL}")
        #     sys.exit(1)


if __name__ == "__main__":
    main()
