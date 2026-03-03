#!/usr/bin/env python
"""Solarian's command-line utility for project commands."""
import sys
import os
import re
import argparse
from pathlib import Path
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
        "add-module": "Project",
        "run-robot": "Robot",
        "calibrate-module": "Robot",
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
        "shell": "Utilities",
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
    elif args.subcommand == "create-project":
        if len(remaining) < 1:
            parser.error("create-project requires a project name argument")
        os.environ["SOLARIAN_CREATE_PROJECT_NAME"] = remaining[0]
        if "--force" in remaining:
            os.environ["SOLARIAN_CREATE_PROJECT_FORCE"] = "true"
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
    no_hydra_commands = ['add-module', 'create-project']

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
            setup()

            cmd.handle(cfg)
        # except ValueError as e:
        #     # Handle configuration validation errors
        #     print(f"\n{colorama.Fore.RED}{str(e)}{colorama.Style.RESET_ALL}")
        #     sys.exit(1)


if __name__ == "__main__":
    main()
