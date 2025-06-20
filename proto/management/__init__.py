from hydra import initialize, compose
import sys

from .base import BaseCommand, CommandRegistry
from . import commands  # Import commands to ensure they are registered


def execute_command(argv):
    """
    Given the command-line arguments, figure out which subcommand is being
    run, create a parser appropriate to that command, and run it.
    """
    try:
        subcommand = argv[1]
    except IndexError:
        subcommand = "help"  # Display help if no arguments were given.

    try:
        cmd = CommandRegistry.get_command(subcommand)
    except Exception as e:
        print(f"Unknown command: {subcommand}")
        sys.exit(1)
    
    overrides = argv[2:]

    # Initialize and compose Hydra config
    with initialize(version_base=None, config_path="conf", job_name="manage"):
        cfg = compose(config_name="config", overrides=overrides)
    cmd.handle(cfg)
    