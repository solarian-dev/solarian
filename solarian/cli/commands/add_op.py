"""``solarian add-op``: a dataset op stub in the project's ``<package>/datasets/``."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry


@CommandRegistry.register("add-op")
class AddOpCommand(BaseCommand):
    category = "Project"
    uses_config = False
    help = "Write a dataset op stub into this project's <package>/datasets/"
    description = """
    Writes <package>/datasets/<name>.py from the op template: a filter that
    drops short episodes, to edit into the op you need. A convenience only:
    any file there that defines one DatasetOp is an op, named after the file
    with _ as -, and `solarian dataset ops` lists it as project:<file>.
    """
    usage = "<name> [--force]"
    options = {
        "<name>": "Op name, as split-by-success or split_by_success.",
        "--force": "Overwrite the file if it exists.",
    }
    examples = (
        "solarian add-op split-by-success",
        "solarian add-op split_by_success --force    # start over from the template",
    )

    def parse_args(self, argv: list[str]) -> list[str]:
        parser = argparse.ArgumentParser(prog="solarian add-op")
        parser.add_argument("name")
        parser.add_argument("--force", action="store_true")
        self.ns = parser.parse_args(argv)
        return []

    def handle(self, cfg: DictConfig) -> None:
        import solarian
        from solarian.cli.manage import _find_settings_path
        from solarian.cli.scaffolder import _render, op_context, pkg_files
        from solarian.datasets.op import PROJECT_OPS_DIR, find_project

        project = find_project()
        if project is None:
            raise SystemExit("Not in a Solarian project: run add-op inside one (a pyproject.toml and its settings.py)")
        package = Path(_find_settings_path(project)).parent
        ctx = op_context(self.ns.name)
        path = package / PROJECT_OPS_DIR / f"{ctx['op_file']}.py"
        if path.exists() and not self.ns.force:
            raise SystemExit(f"{path} exists; --force overwrites it")
        path.parent.mkdir(exist_ok=True)
        template = (pkg_files(solarian) / "templates" / "dataset_op.py.tmpl").read_text(encoding="utf-8")
        path.write_text(_render(template, ctx), encoding="utf-8")
        print(f"Wrote {path.relative_to(project)}")
        print(f"Run it with: solarian dataset apply <dataset> {ctx['op_name']}")
