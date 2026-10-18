from __future__ import annotations

from typing import Any

from solarian.tracking.tracker import Tracker


class NoneTracker(Tracker):
    """Tracks nothing: ``logging.backend: none``."""

    name = "none"

    def log(self, values: dict[str, Any], step: int | None) -> None:
        pass
