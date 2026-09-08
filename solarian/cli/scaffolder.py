from __future__ import annotations

import os
import stat
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

import solarian  # this package, used for locating templates


@dataclass(frozen=True)
class FileSpec:
    template_relpath: Path     # path inside solarian/templates
    render_relpath: Path       # output path inside the project root


PROJECT_FILES: Tuple[FileSpec, ...] = (
    # Project-level files
    FileSpec(Path("pyproject.toml.tmpl"), Path("pyproject.toml")),
    # Python package goes into {package_name}/ subdirectory
    FileSpec(Path("__init__.py.tmpl"), Path("{package_name}/__init__.py")),
    FileSpec(Path("robot.py.tmpl"), Path("{package_name}/robot.py")),
    # FileSpec(Path("modules.py.tmpl"), Path("{package_name}/modules.py")),
    FileSpec(Path("controller.py.tmpl"), Path("{package_name}/controller.py")),
    FileSpec(Path("server.py.tmpl"), Path("{package_name}/server.py")),
    FileSpec(Path("action.py.tmpl"), Path("{package_name}/action.py")),
    FileSpec(Path("observation.py.tmpl"), Path("{package_name}/observation.py")),
    FileSpec(Path("settings.py.tmpl"), Path("{package_name}/settings.py")),
    FileSpec(Path("example_module.py.tmpl"), Path("{package_name}/modules/example_module.py")),
    # conf/
    
    FileSpec(Path("conf/controller.yaml.tmpl"), Path("conf/controller.yaml")),
    FileSpec(Path("conf/record.yaml.tmpl"), Path("conf/record.yaml")),
    FileSpec(Path("conf/replay.yaml.tmpl"), Path("conf/replay.yaml")),
    FileSpec(Path("conf/robot.yaml.tmpl"), Path("conf/robot.yaml")),
    FileSpec(Path("conf/server.yaml.tmpl"), Path("conf/server.yaml")),
    FileSpec(Path("conf/train.yaml.tmpl"), Path("conf/train.yaml")),
    FileSpec(Path("conf/deployment/localhost.yaml.tmpl"), Path("conf/deployment/localhost.yaml")),
)


def _render(text: str, context: dict) -> str:
    """Very small templating using string.Template with ${...} placeholders."""
    return Template(text).safe_substitute(context)


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
    root = pkg_files(solarian) / "templates"
    for spec in PROJECT_FILES:
        text = (root / spec.template_relpath.as_posix()).read_text(encoding="utf-8")
        yield (spec.template_relpath, text)


def scaffold_project(project_name: str, target_dir: Path | None = None, force: bool = False) -> Path:
    """
    Scaffold a new Solarian project.

    Args:
        project_name: Name of the project (used for directory, package name, class names).
        target_dir: Parent directory to create the project in. Defaults to cwd.
        force: If True, overwrite existing files.

    Returns:
        Path to the created project root.
    """
    if target_dir is None:
        target_dir = Path(os.getcwd())

    project_root = target_dir / project_name
    if project_root.exists() and not force and any(project_root.iterdir()):
        raise FileExistsError(
            f"Target directory '{project_root}' already exists and is not empty. "
            "Choose another name or use --force."
        )
    project_root.mkdir(parents=True, exist_ok=True)

    # create a modules directory
    modules_dir = project_root / project_name / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    # create a __init__.py file in the modules directory
    (modules_dir / "__init__.py").touch()
    # create a module.py file in the modules directory

    ctx = {
        "project_name": project_name,
        "package_name": project_name.replace("-", "_"),
        "project_name_camel": project_name.replace("-", " ").replace("_", " ").title().replace(" ", ""),
    }

    for spec, (_, text) in zip(PROJECT_FILES, _iter_template_files()):
        rendered = _render(text, ctx)
        render_path = Path(str(spec.render_relpath).format(package_name=ctx["package_name"]))
        _write_file(project_root / render_path, rendered, force=force)

    return project_root


def create_project() -> None:
    """CLI entry point for 'create-solarian <project_name>'."""
    parser = argparse.ArgumentParser(description="Create a new Solarian project.")
    parser.add_argument("project_name", type=str, help="The name of the project to create.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files if present.")

    args = parser.parse_args()

    project_root = scaffold_project(args.project_name, force=args.force)
    _print_success(args.project_name, project_root)


def _print_success(project_name: str, project_root: Path) -> None:
    package_name = project_name.replace("-", "_")
    print(f"\n✓ Project '{project_name}' created at {project_root}\n")
    print("Project structure:")
    print(f"  {project_name}/")
    print(f"  ├── pyproject.toml")
    print(f"  ├── {package_name}/")
    print(f"  │   ├── __init__.py")
    print(f"  │   ├── settings.py")
    print(f"  │   ├── robot.py")
    print(f"  │   ├── modules.py")
    print(f"  │   ├── controller.py")
    print(f"  │   ├── server.py")
    print(f"  │   ├── action.py")
    print(f"  │   └── observation.py")
    print(f"  └── conf/")
    print(f"      ├── robot.yaml")
    print(f"      ├── controller.yaml")
    print(f"      ├── server.yaml")
    print(f"      ├── record.yaml")
    print(f"      ├── replay.yaml")
    print(f"      ├── train.yaml")
    print(f"      └── deployment/")
    print(f"          └── localhost.yaml")
    print()
    print("Next steps:")
    print(f"  cd {project_name}")
    print(f"  pip install -e .          # Install your project as a package")
    print(f"  solarian add-module <plugin> <module_id>   # Add a module plugin")
    print(f"  solarian run-robot        # Run the robot")
    print()
