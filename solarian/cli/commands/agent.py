"""``solarian agent``: the files that point coding agents at Solarian."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry

ACTIONS = ("sync", "install")


@CommandRegistry.register("agent")
class AgentCommand(BaseCommand):
    category = "Project"
    uses_config = False
    setup_modules = ()
    help = "Write the AGENTS.md and skills coding agents read, here or for your user"
    description = """
    sync regenerates the agent files of the project or plugin in the current
    folder for the installed Solarian and plugins: AGENTS.md, CLAUDE.md, and
    the skills in .agents/skills and .claude/skills. A plugin contributes its
    <package>/agent/AGENTS.md and agent/skills to every project it is installed
    in. Files that someone edited are reported and left alone; put project
    notes in .agents/NOTES.md, which sync never touches. create-project,
    create-plugin and install run sync for you.

    install copies the bootstrap skill, which leads an agent to Solarian from
    any folder, into ~/.agents/skills (Cursor, Codex) and, when Claude Code is
    set up, ~/.claude/skills. Run it again after upgrading Solarian.
    """
    usage = "sync [--force] | install [--target agents|claude|cursor] [--force]"
    options = {
        "sync": "Regenerate this project's or plugin's agent files.",
        "install": "Install the bootstrap skill for your user.",
        "--target": "install only: a user skill folder to write, repeatable. "
                    "agents is ~/.agents/skills, claude ~/.claude/skills, cursor ~/.cursor/skills.",
        "--force": "Overwrite files that were edited or that Solarian did not write.",
    }
    examples = (
        "solarian agent sync",
        "solarian agent sync --force    # after moving your edits into .agents/NOTES.md",
        "solarian agent install",
        "solarian agent install --target claude",
    )

    def parse_args(self, argv: list[str]) -> list[str]:
        from solarian.cli.agent_files import USER_SKILL_DIRS

        parser = argparse.ArgumentParser(prog="solarian agent")
        parser.add_argument("action", choices=ACTIONS)
        parser.add_argument("--target", action="append", choices=sorted(USER_SKILL_DIRS))
        parser.add_argument("--force", action="store_true")
        self.ns = parser.parse_args(argv)
        return []

    def handle(self, cfg: DictConfig) -> None:
        from solarian.cli import agent_files

        if self.ns.action == "install":
            targets = self.ns.target or agent_files.default_targets()
            for folder, outcome in agent_files.install_bootstrap_skill(targets, force=self.ns.force):
                if outcome == "skipped":
                    print(f"{folder}: a skill Solarian did not install is already there; --force replaces it")
                else:
                    print(f"{folder}: {outcome}")
            return

        if self.ns.target:
            raise SystemExit("--target applies to `solarian agent install`")
        root = Path(os.environ.get("SOLARIAN_PROJECT_PATH", os.getcwd()))
        kind = agent_files.detect_kind(root)
        if kind is None:
            raise SystemExit(f"No pyproject.toml in {root}: run sync in a Solarian project or plugin")
        print(f"Syncing the agent files of the {kind} at {root}")
        agent_files.print_report(agent_files.sync_agent_files(root, kind, force=self.ns.force))
