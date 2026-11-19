"""PEP 517 backend. pip and uv both invoke it while installing Solarian.

On Apple Silicon the hooks build ``atomics`` from source, and ``xformers``
too when the install asked for the ``ai`` extra. A wheel built on another
machine still fixes both on first import.
"""

from __future__ import annotations

import importlib.util
import shutil
from contextlib import contextmanager
from pathlib import Path

import poetry.core.masonry.api as poetry_api

#: Where ``solarian docs`` reads the pages from in an installed wheel.
BUNDLED_DOCS = Path("solarian") / "_docs"


@contextmanager
def _bundled_docs():
    """docs/**/page.mdx copied into the package for the length of a wheel build.

    Removed afterwards, so a source checkout keeps reading docs/ itself rather
    than a stale copy. A wheel built from an sdist finds docs/ there too.
    """
    source = Path.cwd() / "docs"
    target = Path.cwd() / BUNDLED_DOCS
    shutil.rmtree(target, ignore_errors=True)
    if source.is_dir():
        for page in source.rglob("page.mdx"):
            dest = target / page.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(page, dest)
    try:
        yield
    finally:
        shutil.rmtree(target, ignore_errors=True)


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
    with _bundled_docs():
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
