"""Which tracking backends exist, and which ones a run uses."""

from __future__ import annotations

import importlib.metadata
import logging
import os
from typing import Iterable

from solarian.tracking.tracker import Tracker

logger = logging.getLogger(__name__)

TRACKER_ENTRY_POINT_GROUP = "solarian.trackers"

#: Overrides the backend a config names: comma-separated names, or ``none``.
#: Solarian Base sets it on the jobs it launches so their runs always report
#: back to it.
TRACKING_ENV = "SOLARIAN_TRACKING"

DEFAULT_BACKEND = "solarian_base"

#: Where to get a backend that is named but not installed.
_INSTALL_HINTS = {"solarian_base": "pip install solarian-base"}

#: Plugin backends found by module path when their entry point is missing:
#: Modal ships plugins as bare sources, without the distribution metadata
#: entry points live in.
_KNOWN_PLUGIN_TRACKERS = {"solarian_base": ("solarian_base.tracker", "SolarianBaseTracker")}


def discover_trackers() -> dict[str, type[Tracker]]:
    """``{name: Tracker subclass}``: the built-ins, then installed plugins."""
    from solarian.tracking.backends.none import NoneTracker
    from solarian.tracking.backends.wandb import WandbTracker

    out: dict[str, type[Tracker]] = {"wandb": WandbTracker, "none": NoneTracker}
    for ep in importlib.metadata.entry_points(group=TRACKER_ENTRY_POINT_GROUP):
        try:
            out[ep.name] = ep.load()
        except Exception:  # noqa: BLE001 - a broken plugin must not kill discovery
            logger.exception("Failed to load tracker entry point %r", ep.name)
    for name, (module, attr) in _KNOWN_PLUGIN_TRACKERS.items():
        if name not in out:
            try:
                out[name] = getattr(importlib.import_module(module), attr)
            except ImportError:
                pass
    return out


def _names(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def resolve_backends(configured: str | Iterable[str] | None = None) -> list[str]:
    """The backend names a run logs to.

    ``$SOLARIAN_TRACKING`` wins over ``configured`` (a config's
    ``logging.backend``), which defaults to solarian_base. ``none`` means no
    tracking.
    """
    names = _names(os.environ.get(TRACKING_ENV)) or _names(configured) or [DEFAULT_BACKEND]
    names = list(dict.fromkeys(names))
    if names == ["none"]:
        return []
    names = [n for n in names if n != "none"]
    available = discover_trackers()
    missing = [n for n in names if n not in available]
    if missing:
        hints = "".join(f" ({_INSTALL_HINTS[n]})" for n in missing if n in _INSTALL_HINTS)
        raise ValueError(
            f"Unknown tracking backend {', '.join(missing)}{hints}. "
            f"Available: {', '.join(sorted(available))}."
        )
    return names


def forward_env(configured: str | Iterable[str] | None = None) -> dict[str, str]:
    """The environment a remote job needs to log to the backends it resolves to."""
    trackers = discover_trackers()
    env: dict[str, str] = {}
    for name in resolve_backends(configured):
        env.update(trackers[name].forward_env())
    if os.environ.get(TRACKING_ENV):
        env[TRACKING_ENV] = os.environ[TRACKING_ENV]
    return env
