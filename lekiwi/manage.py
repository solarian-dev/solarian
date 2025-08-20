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

from hydra import initialize, compose
from omegaconf import DictConfig

from proto.management import CommandRegistry


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
    
    # sys.argv = [arg for arg in sys.argv if not arg.startswith("-") and not arg.startswith("run-")]
    # remove arg names its corresponding values where the arg name starts with '-' or 'run-'
    index_remove = []
    for i, arg in enumerate(sys.argv):
        if arg.startswith("-"):
            index_remove.append(i)
            index_remove.append(i + 1)
        elif arg.startswith("run-"):
            index_remove.append(i)
            
    sys.argv = [arg for i, arg in enumerate(sys.argv) if i not in index_remove]
    main()

