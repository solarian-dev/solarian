from __future__ import annotations

import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Iterable, Tuple
import argparse

# importlib.resources works with files bundled in the package
try:  # Py 3.11+
    from importlib.resources import files as pkg_files
except ImportError:  # Py 3.9–3.10
    from importlib_resources import files as pkg_files  # type: ignore

import proto  # this package, used for locating templates


@dataclass(frozen=True)
class FileSpec:
    template_relpath: Path     # path inside proto/templates
    render_relpath: Path       # output path inside the project root


PROJECT_FILES: Tuple[FileSpec, ...] = (
    # Python modules go into {project_name}/ subdirectory
    FileSpec(Path("__init__.py.tmpl"), Path("{package_name}/__init__.py")),
    FileSpec(Path("robot.py.tmpl"), Path("{package_name}/robot.py")),
    FileSpec(Path("modules.py.tmpl"), Path("{package_name}/modules.py")),
    FileSpec(Path("controller.py.tmpl"), Path("{package_name}/controller.py")),
    FileSpec(Path("server.py.tmpl"), Path("{package_name}/server.py")),
    # conf/
    FileSpec(Path("conf/settings.py.tmpl"), Path("conf/settings.py")),
    FileSpec(Path("conf/config.yaml.tmpl"), Path("conf/config.yaml")),
    FileSpec(Path("conf/controller.yaml.tmpl"), Path("conf/controller.yaml")),
    FileSpec(Path("conf/record.yaml.tmpl"), Path("conf/record.yaml")),
    FileSpec(Path("conf/replay.yaml.tmpl"), Path("conf/replay.yaml")),
    FileSpec(Path("conf/robot.yaml.tmpl"), Path("conf/robot.yaml")),
    FileSpec(Path("conf/server.yaml.tmpl"), Path("conf/server.yaml")),
    FileSpec(Path("conf/train.yaml.tmpl"), Path("conf/train.yaml")),
)


def _render(text: str, context: dict) -> str:
    """Very small templating using string.Template with ${...} placeholders."""
    return Template(text).substitute(context)


def _write_file(dst: Path, content: str, make_executable: bool = False, force: bool = False) -> None:
    if dst.exists() and not force:
        raise FileExistsError(f"{dst} already exists. Use --force to overwrite.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content, encoding="utf-8")
    if make_executable:
        mode = dst.stat().st_mode
        dst.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _iter_template_files() -> Iterable[Tuple[Path, str]]:
    """Yield (relative_path_in_templates, template_text)."""
    root = pkg_files(proto) / "templates"
    # Explicitly read only the files we declared, to control structure.
    for spec in PROJECT_FILES:
        text = (root / spec.template_relpath.as_posix()).read_text(encoding="utf-8")
        yield (spec.template_relpath, text)


def create_project() -> Path:

    parser = argparse.ArgumentParser(description="Proto's command-line utility for project creation.")
    parser.add_argument("project_name", type=str, help="The name of the project to create.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files if present.")
    
    args = parser.parse_args()
    
    project_root = Path(os.getcwd()) / args.project_name
    if project_root.exists() and not args.force and any(project_root.iterdir()):
        raise FileExistsError(
            f"Target directory '{project_root}' already exists and is not empty. "
            "Choose another name or use --force."
        )
    project_root.mkdir(parents=True, exist_ok=True)

    ctx = {
        "project_name": args.project_name,
        "package_name": args.project_name.replace("-", "_"),
        "project_name_camel": args.project_name.replace("-", " ").title().replace(" ", ""),
    }

    for spec, text in zip(PROJECT_FILES, (t for _, t in _iter_template_files())):
        rendered = _render(text, ctx)
        make_exec = False
        # Format the render path with package_name to create the subfolder
        render_path = Path(str(spec.render_relpath).format(package_name=ctx["package_name"]))
        _write_file(project_root / render_path, rendered, make_executable=make_exec, force=args.force)

    print(f"Project '{args.project_name}' created successfully.")
