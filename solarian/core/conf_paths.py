"""Where each kind of config lives in a project's ``conf/``.

The directory is organized by consumer: ``robot.yaml``, ``server.yaml`` and
``controller.yaml`` at the root, and one folder per kind of config a plugin
can define::

    conf/modules/<module>.yaml        a module, named by the robot's attribute
    conf/skills/<skill>.yaml          a skill's config
    conf/services/<service>.yaml      a service or processor config
    conf/commands/<command>.yaml      a command's config (record, replay)
    conf/training/<run>.yaml          one training run, complete
    conf/calibrations/                calibration results, cameras/ included
    conf/deployment/<profile>.yaml    host layout

A plugin's ``templates/`` mirrors these folders, so ``solarian install``
seeds each file where its consumer looks.
"""

from __future__ import annotations

import os
from pathlib import Path

MODULES = "modules"
SKILLS = "skills"
SERVICES = "services"
CALIBRATIONS = "calibrations"
DEPLOYMENT = "deployment"


def project_dir() -> Path:
    """The active project: ``SOLARIAN_PROJECT_PATH``, else the working directory."""
    project = os.environ.get("SOLARIAN_PROJECT_PATH")
    return Path(project) if project else Path.cwd()


def project_path(path: str | os.PathLike) -> Path:
    """*path* as given when absolute, else inside the active project.

    How a recording ``root`` is resolved, so ``recordings/robot`` lands in the
    project that produced it on whichever machine it runs.
    """
    path = Path(path).expanduser()
    return path if path.is_absolute() else project_dir() / path


def conf_dir() -> Path:
    """The active project's ``conf/``.

    ``SOLARIAN_CONF_PATH`` is set by the CLI for every command and inherited
    by skill child processes; outside the CLI it is ``<cwd>/conf``.
    """
    conf_path = os.environ.get("SOLARIAN_CONF_PATH")
    if conf_path:
        return Path(conf_path)
    return project_dir() / "conf"


def _in_folder(folder: str, filename: str) -> Path:
    """``conf/<folder>/<filename>``.

    Raises when it is missing but the file still sits at the root of
    ``conf/``, where every config lived before the folders existed.
    """
    path = conf_dir() / folder / filename
    flat = conf_dir() / filename
    if not path.exists() and flat.is_file():
        raise FileNotFoundError(f"Move {flat} to {path}.")
    return path


def module_config_path(name: str) -> Path:
    """``conf/modules/<name>.yaml``, whether or not it exists."""
    return _in_folder(MODULES, f"{name}.yaml")


def skill_config_path(name_or_file: str) -> Path:
    """A skill config: a name (``umi``), a filename (``cup_arrange.yaml``) or a path.

    A name or bare filename is in ``conf/skills/``. A path with a directory is
    the caller's own: absolute, or relative to the working directory.
    """
    path = Path(name_or_file)
    if path.is_absolute() or len(path.parts) > 1:
        return path
    filename = name_or_file if path.suffix else f"{name_or_file}.yaml"
    return _in_folder(SKILLS, filename)


def service_config_path(name: str) -> Path:
    """``conf/services/<name>.yaml``, whether or not it exists."""
    return _in_folder(SERVICES, f"{name}.yaml")


def calibrations_dir() -> Path:
    return conf_dir() / CALIBRATIONS


def cameras_dir() -> Path:
    """Per-camera intrinsics, ``<serial>.json``."""
    return calibrations_dir() / "cameras"
