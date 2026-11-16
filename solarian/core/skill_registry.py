"""
Skill registry — discovery of installed skill plugins, and loading a skill by path.

Skill plugins advertise their :class:`Skill` subclasses under the
``solarian.skills`` entry-point group, e.g. in a plugin's ``pyproject.toml``::

    [project.entry-points."solarian.skills"]
    umi = "solarian_umi.skill:UMISkill"

Installing such a package makes the skill discoverable by any
:class:`~solarian.core.controller.Controller` without editing its code — the
foundation the skill marketplace builds on.

Starting a skill on the robot or server does not go through discovery: a
``START_SKILL`` carries the class's import path, and :func:`load_skill_class`
imports it there.
"""

import importlib
import importlib.metadata
import logging
from typing import Dict

logger = logging.getLogger(__name__)

#: Entry-point group skill plugins register under.
SKILL_ENTRY_POINT_GROUP = "solarian.skills"

_loaded: Dict[str, type] = {}


def _iter_entry_points(group: str):
    """Yield entry points for *group* across importlib.metadata API versions."""
    try:
        eps = importlib.metadata.entry_points()
    except Exception:  # noqa: BLE001 - metadata access can fail in odd envs
        logger.exception("Failed to read entry points")
        return []
    # Python 3.10+: EntryPoints.select; older: dict-like .get
    if hasattr(eps, "select"):
        return list(eps.select(group=group))
    return list(eps.get(group, []))  # type: ignore[attr-defined]


def discover_skill_classes() -> Dict[str, type]:
    """Return ``{entry_point_name: Skill subclass}`` for installed plugins."""
    out: Dict[str, type] = {}
    for ep in _iter_entry_points(SKILL_ENTRY_POINT_GROUP):
        try:
            out[ep.name] = ep.load()
        except Exception:  # noqa: BLE001 - a broken plugin must not kill discovery
            logger.exception("Failed to load skill entry point %r", ep.name)
    return out


def load_skill_class(path: str) -> type:
    """The :class:`Skill` subclass at ``"package.module:QualName"``.

    Raises:
        ValueError: ``path`` is malformed or names something that is not a
            ``Skill`` subclass.
        ImportError, AttributeError: the module or the class does not exist.
    """
    cls = _loaded.get(path)
    if cls is not None:
        return cls
    # Imported here to avoid a circular import at module load time.
    from solarian.core.skill import Skill

    module_name, sep, qualname = path.partition(":")
    if not (sep and module_name and qualname):
        raise ValueError(f"skill path {path!r} is not 'module:QualName'")
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not (isinstance(obj, type) and issubclass(obj, Skill)):
        raise ValueError(f"{path!r} is not a Skill subclass")
    _loaded[path] = obj
    return obj
