"""
Install a plugin into the current environment and project.

    solarian install solarian-move-skill
    solarian install plugins/solarian-move-skill
    solarian install ur5
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry


class InstallError(Exception):
    """The package could not be resolved or installed."""


def looks_like_path(arg: str) -> bool:
    return "/" in arg or arg.startswith(".") or arg.startswith("~")


def distribution_name(name: str) -> str:
    """``ur5`` and ``solarian-ur5`` are the same distribution."""
    name = name.strip()
    if name.startswith("solarian-"):
        return name
    return f"solarian-{name}"


def _is_package(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() or (path / "setup.py").is_file()


def pypi_has_project(name: str) -> bool | None:
    """True when PyPI has *name*, False on 404, None when PyPI cannot be reached."""
    url = f"https://pypi.org/pypi/{name}/json"
    request = urllib.request.Request(url, headers={"User-Agent": "solarian-install"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status == 200
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        return None
    except OSError:
        return None


def find_local_package(package: str, cwd: Path) -> Path | None:
    """``<cwd>/<package>`` first, then ``<cwd>/plugins/<package>``."""
    for candidate in (cwd / package, cwd / "plugins" / package):
        if candidate.is_dir() and _is_package(candidate):
            return candidate.resolve()
    return None


def resolve_source(arg: str, cwd: Path, pypi_has) -> tuple[str, Path | None]:
    """Return ``(distribution, local_path)``. ``local_path`` is None for a PyPI install.

    *pypi_has* is ``pypi_has_project`` or a test double with the same contract.
    """
    if looks_like_path(arg):
        path = Path(arg).expanduser()
        if not path.is_absolute():
            path = cwd / path
        path = path.resolve()
        if not path.exists():
            raise InstallError(f"Path {arg!r} does not exist (resolved to {path}).")
        if not path.is_dir() or not _is_package(path):
            raise InstallError(
                f"{path} is not a plugin directory (no pyproject.toml or setup.py)."
            )
        return path.name, path

    package = distribution_name(arg)
    availability = pypi_has(package)
    if availability is True:
        return package, None

    local = find_local_package(package, cwd)
    if local is not None:
        return package, local

    looked = f"{cwd / package} and {cwd / 'plugins' / package}"
    if availability is None:
        raise InstallError(
            f"Could not reach PyPI for {package}, and it was not found at {looked}."
        )
    raise InstallError(f"{package} is not on PyPI and was not found at {looked}.")


def package_dirs(package: str, local_path: Path | None) -> list[Path]:
    """Directories that may hold the installed import package."""
    pkg = package.replace("-", "_")
    if local_path is not None:
        return [local_path / pkg, local_path / "src" / pkg]
    spec = importlib.util.find_spec(pkg)
    if spec is None or not spec.origin:
        return []
    return [Path(spec.origin).parent]


def config_sources(package: str, local_path: Path | None) -> list[tuple[Path, str]]:
    """``(source, conf filename)`` for templates this plugin seeds.

    ``templates/*.yaml`` keeps its filename. A plugin with no templates directory
    still seeds a legacy ``conf.yaml`` as ``conf/<short-name>.yaml``.
    """
    roots = [root for root in package_dirs(package, local_path) if root.is_dir()]
    for root in roots:
        templates = root / "templates"
        if templates.is_dir():
            files = sorted(path for path in templates.glob("*.yaml") if path.is_file())
            if files:
                return [(path, path.name) for path in files]
    short = package[len("solarian-"):] if package.startswith("solarian-") else package
    for root in roots:
        conf = root / "conf.yaml"
        if conf.is_file():
            return [(conf, f"{short}.yaml")]
    return []


def seed_configs(sources: list[tuple[Path, str]], conf_dir: Path) -> list[Path]:
    """Copy each source into *conf_dir* when the destination is absent."""
    conf_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for src, name in sources:
        dest = conf_dir / name
        if dest.exists():
            continue
        shutil.copy2(src, dest)
        written.append(dest)
    return written


def add_dependency(pyproject: Path, package: str) -> str:
    """Insert ``package>=0.1.0`` into ``[project] dependencies``.

    Returns ``added``, ``present``, ``missing-file``, or ``no-dependencies``.
    """
    if not pyproject.is_file():
        return "missing-file"
    text = pyproject.read_text(encoding="utf-8")
    uncommented = "\n".join(
        line for line in text.split("\n") if not line.strip().startswith("#")
    )
    if f'"{package}' in uncommented:
        return "present"
    dep_line = f'    "{package}>=0.1.0",'
    lines = text.split("\n")
    in_project = False
    in_deps = False
    insert_at = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("[project]"):
            in_project = True
            in_deps = False
            continue
        if line.strip().startswith("[") and not line.strip().startswith("[project]"):
            in_project = False
            in_deps = False
            continue
        if in_project and re.match(r"^\s*dependencies\s*=\s*\[\s*$", line):
            in_deps = True
            continue
        if in_project and re.match(r"^\s*dependencies\s*=\s*\[", line):
            in_deps = True
            if line.rstrip().endswith("]"):
                insert_at = i
                break
            continue
        if in_deps and line.strip() == "]":
            insert_at = i
            break
    if insert_at < 0:
        return "no-dependencies"
    lines.insert(insert_at, dep_line)
    pyproject.write_text("\n".join(lines), encoding="utf-8")
    return "added"


def pip_install(package: str, local_path: Path | None) -> None:
    """Install into the current interpreter. A local path is editable."""
    if local_path is None:
        try:
            import importlib.metadata
            importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError:
            command = [sys.executable, "-m", "pip", "install", package]
        else:
            return
    else:
        command = [sys.executable, "-m", "pip", "install", "-e", str(local_path)]
    try:
        subprocess.check_call(command, stdout=sys.stdout, stderr=sys.stderr)
    except subprocess.CalledProcessError as exc:
        raise InstallError(f"Failed to install {package}: {exc}") from exc


@CommandRegistry.register("install")
class InstallCommand(BaseCommand):
    help = "Install a plugin into this environment and project"
    uses_config = False
    description = """
    Installs a Solarian plugin into the current Python environment and adds it
    to this project's pyproject.toml.

    A name is looked up on PyPI first. A short name is prefixed, so ur5
    installs solarian-ur5. If PyPI has no such project, the command looks for
    a directory of that name in the current folder and then under plugins/,
    and installs the one it finds editable. A path skips PyPI and installs
    that directory editable.

    Config templates shipped in the package's templates/ directory are copied
    into conf/ when those files are not already there. A module plugin that
    only ships conf.yaml is copied to conf/<short-name>.yaml.
    """
    usage = "<package-or-path>"
    options = {
        "<package-or-path>": "Distribution name (solarian-move-skill, or a short "
                             "name such as ur5) or a path to a plugin directory.",
    }
    examples = (
        "solarian install solarian-move-skill",
        "solarian install plugins/solarian-move-skill",
        "solarian install ur5",
    )

    def handle(self, cfg: DictConfig) -> None:
        arg = os.environ.get("SOLARIAN_INSTALL_TARGET")
        if not arg:
            raise InstallError(
                "Usage: solarian install <package-or-path>\n"
                "  e.g. solarian install solarian-move-skill\n"
                "  e.g. solarian install plugins/solarian-move-skill"
            )
        project = Path(os.environ.get("SOLARIAN_PROJECT_PATH", os.getcwd()))
        package, local_path = resolve_source(arg, project, pypi_has_project)
        print(f"Installing {package}" + (f" from {local_path}" if local_path else " from PyPI"))
        pip_install(package, local_path)

        sources = config_sources(package, local_path)
        conf_dir = project / "conf"
        written = set(seed_configs(sources, conf_dir))
        if not sources:
            print(f"{package} ships no config template.")
        for _src, name in sources:
            dest = conf_dir / name
            if dest in written:
                print(f"Copied template to conf/{name}")
            else:
                print(f"conf/{name} already exists. Leaving it in place.")

        status = add_dependency(project / "pyproject.toml", package)
        if status == "added":
            print(f"Added {package} to pyproject.toml.")
        elif status == "present":
            print(f"{package} is already in pyproject.toml.")
        elif status == "missing-file":
            print(f"No pyproject.toml in {project}. Add {package} to your dependencies manually.")
        else:
            print(
                f"Could not find [project] dependencies in {project / 'pyproject.toml'}. "
                f"Add {package}>=0.1.0 manually."
            )
