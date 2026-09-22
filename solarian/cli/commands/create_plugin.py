"""
Create Plugin Command

Scaffolds a new Solarian plugin directory (or adds a new component type to an
existing plugin) with all necessary files, including a pyproject.toml wired with
the appropriate entry points for the chosen component type.

Usage:
    solarian create-plugin <type> <name> [--force]

Examples:
    solarian create-plugin actuator feetech
    solarian create-plugin module feetech      # adds a module to the same plugin
"""

import os
import sys

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry


@CommandRegistry.register('create-plugin')
class CreatePluginCommand(BaseCommand):
    help = 'Scaffold a new Solarian plugin (or add a component type to an existing one)'
    description = """
    Writes plugins/solarian-<name>/ with the source layout and a pyproject.toml
    whose entry points advertise the component to Solarian, so installing the
    plugin is all it takes for the CLI and the registries to find it.

    Run it again with a different type and the same name to add a second
    component to the plugin that already exists; the shared files are left alone.
    """
    usage = "<type> <name> [--force]"
    options = {
        "<type>": "Component to scaffold: actuator, sensor, module, robot, command "
                  "or policy.",
        "<name>": "Plugin name without the solarian- prefix (feetech, ur5, ...).",
        "--force": "Overwrite files that already exist.",
    }
    examples = (
        "solarian create-plugin actuator feetech",
        "solarian create-plugin module feetech    # second component, same plugin",
    )

    def handle(self, cfg: DictConfig) -> None:
        from solarian.cli.plugin_scaffolder import (
            ALLOWED_TYPES,
            scaffold_plugin,
            _print_plugin_success,
        )

        plugin_type = os.environ.get("SOLARIAN_CREATE_PLUGIN_TYPE")
        plugin_name = os.environ.get("SOLARIAN_CREATE_PLUGIN_NAME")
        force = os.environ.get("SOLARIAN_CREATE_PLUGIN_FORCE", "").lower() == "true"

        if not plugin_type or not plugin_name:
            print("Error: plugin type and name are required")
            print("Usage: solarian create-plugin <type> <name> [--force]")
            print(f"  <type> is one of: {', '.join(sorted(ALLOWED_TYPES))}")
            sys.exit(1)

        plugin_type = plugin_type.lower()
        if plugin_type not in ALLOWED_TYPES:
            print(f"Error: unknown plugin type '{plugin_type}'")
            print(f"  Valid types are: {', '.join(sorted(ALLOWED_TYPES))}")
            sys.exit(1)

        try:
            plugin_root, created, added = scaffold_plugin(plugin_type, plugin_name, force=force)
            _print_plugin_success(plugin_type, plugin_name, plugin_root, created, added)
        except FileExistsError as e:
            print(f"\n✗ {e}")
            print("  Use --force to overwrite existing files.")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ Failed to create plugin: {e}")
            sys.exit(1)
