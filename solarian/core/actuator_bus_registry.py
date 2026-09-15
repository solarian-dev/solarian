"""
Actuator bus registry — discovery of installed servo buses via entry points.

Actuator plugins advertise their :class:`~solarian.core.actuator_bus.ActuatorBus`
subclasses under the ``solarian.actuator_buses`` entry-point group, e.g. in a
plugin's ``pyproject.toml``::

    [project.entry-points."solarian.actuator_buses"]
    feetech = "solarian_feetech.bus:FeetechBus"

The entry-point name is the servo brand, so ``solarian calibrate-servo`` can
offer a brand menu and then reach that brand's class-level provisioning helpers
(``available_ports`` / ``scan_port`` / ``provision_servo``) without importing any
plugin up front. Module configs resolve their bus through the actuator classes
instead, which is why this is only needed for brand-first flows.
"""

import importlib.metadata
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

#: Entry-point group actuator buses register under.
ACTUATOR_BUS_ENTRY_POINT_GROUP = "solarian.actuator_buses"


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


def discover_bus_classes() -> Dict[str, type]:
    """Return ``{entry_point_name: ActuatorBus subclass}`` for installed plugins."""
    out: Dict[str, type] = {}
    for ep in _iter_entry_points(ACTUATOR_BUS_ENTRY_POINT_GROUP):
        try:
            out[ep.name] = ep.load()
        except Exception:  # noqa: BLE001 - a broken plugin must not kill discovery
            logger.exception("Failed to load actuator bus entry point %r", ep.name)
    return out


def available_bus_names() -> List[str]:
    """Return the names of all installed buses (without importing them)."""
    return sorted(ep.name for ep in _iter_entry_points(ACTUATOR_BUS_ENTRY_POINT_GROUP))


def get_bus_class(name: str) -> type:
    """Resolve a bus *name* (servo brand) to its :class:`ActuatorBus` subclass.

    Raises:
        KeyError: if no installed plugin registers a bus under *name*. The error
            lists the buses that *are* available to make typos and missing
            installs obvious.
    """
    for ep in _iter_entry_points(ACTUATOR_BUS_ENTRY_POINT_GROUP):
        if ep.name == name:
            return ep.load()
    available = available_bus_names()
    hint = ", ".join(available) if available else "(none installed)"
    raise KeyError(
        f"No actuator bus registered as {name!r} under the "
        f"{ACTUATOR_BUS_ENTRY_POINT_GROUP!r} entry-point group. "
        f"Available buses: {hint}. "
        f"Install the actuator plugin that provides {name!r} "
        f"(e.g. `pip install solarian-{name}`)."
    )
