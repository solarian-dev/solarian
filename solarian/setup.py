import importlib
import os
import re
from pathlib import Path


_is_setup = False


def _find_package_name(project_path: Path) -> str | None:
    """Derive the project's Python package name from the project directory.

    Uses the same discovery logic as ``manage.py``: first checks
    ``pyproject.toml``, then scans for a directory with ``__init__.py``.
    """
    pyproject = project_path / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(encoding="utf-8")
        match = re.search(r'name\s*=\s*"([^"]+)"', text)
        if match:
            return match.group(1).replace("-", "_")

    for child in sorted(project_path.iterdir()):
        if child.is_dir() and (child / "__init__.py").exists():
            return child.name

    return None


_ALL_MODULES = ("robot", "observation", "action", "controller", "server")


def setup(modules=None):
    """Auto-discover and import project modules.

    Derives the project package from ``SOLARIAN_PROJECT_PATH`` and imports
    conventional modules in dependency order:

    1. ``robot`` — defines the Robot subclass and its module descriptors
    2. ``observation`` / ``action`` — reference Robot attributes via
       SensorField / ActuatorField, triggering ``__init_subclass__``
       registration
    3. ``controller`` / ``server`` — may reference observation/action classes

    Only modules whose ``.py`` file actually exists are imported, so
    projects that omit some of these are fine.

    Parameters
    ----------
    modules : tuple[str, ...] | None
        Which conventional modules to import.  ``None`` (the default) means
        *all* modules — backward-compatible with the previous behaviour.
        Pass an explicit tuple to restrict imports, e.g.
        ``("robot", "observation", "action")`` for robot-side commands.
        An empty tuple skips all project-module imports.

    Idempotent — safe to call multiple times.
    """
    global _is_setup
    if _is_setup:
        return
    _is_setup = True

    project_path = Path(os.environ.get("SOLARIAN_PROJECT_PATH", os.getcwd()))
    package = _find_package_name(project_path)
    if package is None:
        return

    if modules is not None:
        wanted = set(modules)
        to_import = [m for m in _ALL_MODULES if m in wanted]
    else:
        to_import = list(_ALL_MODULES)

    for module_name in to_import:
        module_file = project_path / package / f"{module_name}.py"
        if module_file.exists():
            importlib.import_module(f"{package}.{module_name}")
