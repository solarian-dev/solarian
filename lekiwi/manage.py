#!/usr/bin/env python
"""Proto's command-line utility for project commands."""
import sys
import os
import argparse
from pathlib import Path
from typing import Optional

parser = argparse.ArgumentParser(description="Proto's command-line utility for project commands.")
parser.add_argument("subcommand", type=str, help="The subcommand to run.")
parser.add_argument("--conf", type=str, default="conf", help="The conf directory with yaml config files to use.")
args = parser.parse_args()

os.environ["PROTO_SETTINGS"] = str(Path(__file__).parent / args.conf / "settings.py")


# import hydra
from hydra import initialize, compose
from omegaconf import DictConfig

from proto.management import CommandRegistry
# from proto.conf import SettingsManager

# SettingsManager.load_settings(project_root=__package__)

# set environment variable with the path to the settings class file



# cs = ConfigStore.instance()
# cs.store(name="config", node=RobotConfig)

# def get_subcommand() -> tuple[str, list[str]]:
#     """Extract subcommand and return modified argv."""
#     if len(sys.argv) > 1:
#         subcommand = sys.argv[1]
#         new_argv = [sys.argv[0]] + sys.argv[2:]
#     else:
#         subcommand = "help"
#         new_argv = sys.argv
#     return subcommand, new_argv

# Store original argv and get subcommand before Hydra processes args
# _subcommand: Optional[str] = None
# if __name__ == "__main__":
#     _subcommand, sys.argv = get_subcommand()

# @hydra.main(version_base=None, config_path="conf", config_name="config")
# def main(cfg: DictConfig) -> None:
# def main() -> None:
#     """
#     Given the command-line arguments, figure out which subcommand is being
#     run, create a parser appropriate to that command, and run it.
#     """
#     global _subcommand
#     subcommand = _subcommand or "help"

#     try:
#         cmd = CommandRegistry.get_command(subcommand)
#     except Exception as e:
#         print(f"Unknown command: {subcommand}")
#         sys.exit(1)
    
#     # cmd.handle(cfg)
#     cmd.handle()


def main() -> None:
    """
    Given the command-line arguments, figure out which subcommand is being
    run, create a parser appropriate to that command, and run it.
    """
    subcommand = args.subcommand or "help"

    

    try:
        cmd = CommandRegistry.get_command(subcommand)
    except Exception as e:
        print(f"Unknown command: {subcommand}")
        sys.exit(1)
    
    config_name = CommandRegistry.get_config_name(subcommand)

    with initialize(version_base=None, config_path=args.conf, job_name='proto_app'):
        cfg = compose(config_name=config_name)
        cmd.handle(cfg)


if __name__ == '__main__':
    
    
    # clear sys.argv for hydra
    sys.argv = []

    main()

