"""
Skill registry — discovery of installed skill plugins via entry points.

Skill plugins advertise their :class:`Skill` subclasses under the
``solarian.skills`` entry-point group, e.g. in a plugin's ``pyproject.toml``::

    [project.entry-points."solarian.skills"]
    cup_arrange = "solarian_skill_cup_arrange.skill:CupArrangeSkill"

Installing such a package makes the skill discoverable by an
:class:`~solarian.core.ai.ai_controller.AIController` without editing any
controller code — the foundation the skill marketplace builds on.
"""

import importlib.metadata
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

#: Entry-point group skill plugins register under.
SKILL_ENTRY_POINT_GROUP = "solarian.skills"

#: Auto-assigned skill ids for discovered skills start here, to avoid
#: colliding with the small, hand-assigned ids projects typically use.
_AUTO_SKILL_ID_START = 1000


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


def discover_available_skills(
    start_skill_id: int = _AUTO_SKILL_ID_START,
) -> Dict[str, Dict[str, Any]]:
    """Build an ``available_skills``-style registry from installed plugins.

    The returned mapping has the same shape as the one
    :meth:`Controller._collect_controller_skills` builds from its own
    ``skills`` list: ``name -> {cls, execution_place, execution_mode,
    skill_id, docstring}``.  Those fields are read from a
    default-constructed template instance when possible; skills that cannot
    be default-constructed fall back to sensible defaults.
    """
    # Imported here to avoid a circular import at module load time.
    from solarian.core.execution import ExecutionMode, ExecutionPlace
    from solarian.core.ai.system1.skill import Skill

    skills: Dict[str, Dict[str, Any]] = {}
    next_id = start_skill_id
    for name, cls in discover_skill_classes().items():
        if not (isinstance(cls, type) and issubclass(cls, Skill)):
            logger.warning(
                "Skill entry point %r does not point to a Skill subclass (%r); skipping",
                name,
                cls,
            )
            continue

        execution_place: Any = ExecutionPlace.CONTROLLER
        execution_mode: Any = ExecutionMode.PROCESS
        skill_id = None
        try:
            template = cls()
            execution_place = template.execution_place
            execution_mode = template.execution_mode
            skill_id = template.skill_id or None
        except Exception:  # noqa: BLE001 - skill may require constructor args
            logger.debug(
                "Could not default-construct skill %r for introspection; "
                "using defaults",
                name,
                exc_info=True,
            )

        if not skill_id:
            skill_id = next_id
            next_id += 1

        skills[name] = {
            "cls": cls,
            "execution_place": execution_place,
            "execution_mode": execution_mode,
            "skill_id": skill_id,
            "docstring": (cls.__doc__ or "").strip(),
        }
    return skills
