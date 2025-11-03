#!/usr/bin/env python
"""Proto's command-line utility for project commands."""
import sys
import os
import argparse
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError

parser = argparse.ArgumentParser(description="Proto's command-line utility for project commands.")
parser.add_argument("subcommand", type=str, help="The subcommand to run.")
parser.add_argument("--conf", type=str, default="conf", help="The conf directory with yaml config files to use.")
parser.add_argument("overrides", nargs="*", help="Hydra-style config overrides (e.g., name=value)")
args = parser.parse_args()

if '/' in args.conf:
    CONF_PATH = args.conf.split("/")[0]
    CONF_NAME = args.conf.split("/")[1]
else:
    CONF_PATH = args.conf
    CONF_NAME = "default"

os.environ["PROTO_SETTINGS"] = str(Path(__file__).parent / CONF_PATH / "settings.py")
os.environ["PROTO_PROJECT_PATH"] = str(Path(__file__).parent)

from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
import colorama

from proto.management import CommandRegistry


def parse_with_friendly_errors(model_class, data):
    """Parse config data with user-friendly error messages"""
    try:
        return model_class(**data)
    except ValidationError as e:
        # Get the config filename if possible
        config_path = os.environ.get("PROTO_CURRENT_CONFIG", "")
        
        # Format a user-friendly error message
        error_msg = f"Configuration Error in {model_class.__name__}:\n"
        error_msg += f"  {str(e)}\n"
        
        if config_path:
            error_msg += f"  Please check your configuration file: {config_path}\n"
            
        # Re-raise with the friendly message
        raise ValueError(error_msg) from e


def main() -> None:
    """
    Given the command-line arguments, figure out which subcommand is being
    run, create a parser appropriate to that command, and run it.
    """
    colorama.init()  # Initialize colorama for colored terminal output
    
    subcommand = args.subcommand or "help"
    try:
        cmd = CommandRegistry.get_command(subcommand)
    except Exception as e:
        print(f"Unknown command: {subcommand}")
        sys.exit(1)
    
    if CONF_NAME == "default":
        config_name = CommandRegistry.get_config_name(subcommand)
    else:
        config_name = CONF_NAME

    try:
        with initialize(version_base=None, config_path=CONF_PATH, job_name='proto_app'):
            # Store the current config path for error reporting
            if config_name is not None:
                config_file = Path(CONF_PATH) / config_name
                if config_file.exists():
                    os.environ["PROTO_CURRENT_CONFIG"] = str(config_file.absolute())
                
            # Load the config using hydra
            hydra_cfg = compose(config_name=config_name, overrides=args.overrides)
            
            # Convert to dict for Pydantic parsing
            cfg_dict = OmegaConf.to_container(hydra_cfg, resolve=True)
            
            # Parse using the command's config class which now has built-in validation
            # cmd_config_class = cmd.get_config_class()
            # if cmd_config_class:
            #     cfg = cmd_config_class.parse_with_friendly_errors(cfg_dict)
            # else:
            #     cfg = hydra_cfg
            if cmd.get_config_class() is not None:
                cfg = parse_with_friendly_errors(cmd.get_config_class(), cfg_dict)
            else:
                cfg = hydra_cfg
                
            cmd.handle(cfg)
    except ValueError as e:
        # Handle configuration validation errors
        print(f"\n{colorama.Fore.RED}{str(e)}{colorama.Style.RESET_ALL}")
        sys.exit(1)
    # except Exception as e:
    #     # Handle other exceptions
    #     print(f"\n{colorama.Fore.RED}Error: {str(e)}{colorama.Style.RESET_ALL}")
    #     import traceback
    #     print(traceback.format_exc())
    #     sys.exit(1)


if __name__ == '__main__':
    # clear sys.argv for hydra
    
    # sys.argv = [arg for arg in sys.argv if not arg.startswith("-") and not arg.startswith("run-")]
    # remove arg names its corresponding values where the arg name starts with '-' or 'run-'
    # index_remove = []
    # for i, arg in enumerate(sys.argv):
    #     if arg.startswith("-"):
    #         index_remove.append(i)
    #         index_remove.append(i + 1)
    #     elif arg.startswith("run-"):
    #         index_remove.append(i)
            
    # sys.argv = [arg for i, arg in enumerate(sys.argv) if i not in index_remove]
    main()
