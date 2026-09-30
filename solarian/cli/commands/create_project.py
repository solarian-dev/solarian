"""
Create Project Command

Scaffolds a new Solarian project directory with all necessary template files,
including a pyproject.toml where module plugins can be added as dependencies.
"""

import argparse
import sys

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry


@CommandRegistry.register('create-project')
class CreateProjectCommand(BaseCommand):
    category = "Project"
    uses_config = False
    help = 'Create a new Solarian project with the specified name'
    description = """
    Scaffolds a project directory in the current working directory: a Robot and a
    Controller class, a settings.py the CLI discovers, a conf/ tree with the yaml
    each command composes (robot, controller, server, record, replay, train) and
    a pyproject.toml where module plugins are added as dependencies.

    Add a plugin to the new project with `solarian install`.
    """
    usage = "<project_name> [--force]"
    options = {
        "<project_name>": "Directory and Python package to create. Use a valid "
                          "identifier: it is imported as a package.",
        "--force": "Overwrite files that already exist.",
    }
    examples = (
        "solarian create-project my_robot",
        "solarian create-project my_robot --force",
    )

    def parse_args(self, argv: list[str]) -> list[str]:
        parser = argparse.ArgumentParser(prog="solarian create-project")
        parser.add_argument("project_name", help="Directory and Python package to create.")
        parser.add_argument("--force", action="store_true", help="Overwrite files that already exist.")
        ns = parser.parse_args(argv)
        self.project_name = ns.project_name
        self.force = ns.force
        return []

    def handle(self, cfg: DictConfig) -> None:
        project_name = self.project_name
        force = self.force

        if not project_name:
            print("Error: project name is required")
            print("Usage: solarian create-project <project_name> [--force]")
            sys.exit(1)

        from solarian.cli.scaffolder import scaffold_project, _print_success

        try:
            project_root = scaffold_project(project_name, force=force)
            _print_success(project_name, project_root)
        except FileExistsError as e:
            print(f"\n✗ {e}")
            print("  Use --force to overwrite existing files.")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ Failed to create project: {e}")
            sys.exit(1)
