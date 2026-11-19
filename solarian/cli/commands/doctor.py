"""``solarian doctor``: check a project or plugin without running it."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry


@CommandRegistry.register("doctor")
class DoctorCommand(BaseCommand):
    category = "Project"
    uses_config = False
    setup_modules = ()
    help = "Check this project or plugin for setup mistakes, without running it"
    description = """
    Reads the project or plugin in the current folder, without importing it,
    and reports what would stop it from running or mislead a coding agent:

    - the package and the solarian-* plugins it depends on are installed, and
      a plugin's installed entry points match its pyproject.toml
    - settings.py names a robot, controller and server class that exist
    - each of the robot's modules has its conf/modules/<name>.yaml
    - every deployment profile sets the ${deployment.*} keys conf/ reads
    - AGENTS.md and the skills were written by the installed Solarian and are
      current with the installed plugins, and none were edited by hand
    - the solarian skill in your user skill folders is current

    Each problem comes with the command or change that fixes it. Exits 1 when
    there is an error, so a script or an agent can gate on it.
    """
    usage = "[--json]"
    options = {
        "--json": "Print the checks as JSON: solarian, root, kind and checks (status, message, fix).",
    }
    examples = (
        "solarian doctor",
        "solarian doctor --json",
    )

    def parse_args(self, argv: list[str]) -> list[str]:
        parser = argparse.ArgumentParser(prog="solarian doctor")
        parser.add_argument("--json", action="store_true")
        self.ns = parser.parse_args(argv)
        return []

    def handle(self, cfg: DictConfig) -> None:
        from solarian.cli.agent_files import solarian_version
        from solarian.cli.doctor import ERROR, OK, WARN, run_checks

        root = Path(os.environ.get("SOLARIAN_PROJECT_PATH", os.getcwd())).resolve()
        kind, checks = run_checks(root)
        errors = sum(check.status == ERROR for check in checks)

        if self.ns.json:
            print(json.dumps({"solarian": solarian_version(), "root": str(root), "kind": kind,
                              "checks": [asdict(check) for check in checks]}, indent=2))
        else:
            print(f"Solarian {solarian_version()}, {kind or 'no project'} at {root}\n")
            for check in checks:
                message = check.message.replace("\n", "\n         ")
                print(f"  {check.status:<6} {message}")
                if check.fix and check.status != OK:
                    print(f"         fix: {check.fix}")
            warnings = sum(check.status == WARN for check in checks)
            print(f"\n{errors} error{'s' * (errors != 1)}, {warnings} warning{'s' * (warnings != 1)}")
        if errors:
            raise SystemExit(1)
