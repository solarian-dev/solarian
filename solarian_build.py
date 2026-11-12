"""PEP 517 backend. pip and uv both invoke it while installing Solarian.

On Apple Silicon the hooks build ``atomics`` from source, and ``xformers``
too when the install asked for the ``ai`` extra. A wheel built on another
machine still fixes both on first import.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import poetry.core.masonry.api as poetry_api


def _platform_deps():
    path = Path.cwd() / "solarian" / "_platform_deps.py"
    spec = importlib.util.spec_from_file_location("_solarian_platform_deps", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"solarian build could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    deps = _platform_deps()
    deps.ensure_atomics()
    deps.ensure_xformers(installing=True)
    return poetry_api.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    deps = _platform_deps()
    deps.ensure_atomics()
    deps.ensure_xformers(installing=True)
    return poetry_api.build_editable(wheel_directory, config_settings, metadata_directory)


get_requires_for_build_wheel = poetry_api.get_requires_for_build_wheel
get_requires_for_build_sdist = poetry_api.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = poetry_api.prepare_metadata_for_build_wheel
build_sdist = poetry_api.build_sdist

if hasattr(poetry_api, "get_requires_for_build_editable"):
    get_requires_for_build_editable = poetry_api.get_requires_for_build_editable
if hasattr(poetry_api, "prepare_metadata_for_build_editable"):
    prepare_metadata_for_build_editable = poetry_api.prepare_metadata_for_build_editable
