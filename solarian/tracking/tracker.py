"""The backend contract: one experiment-tracking service's view of one run."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class RunSpec:
    """What a run is started with; the same fields wandb.init takes.

    ``extra`` carries backend-specific keys from a config's ``logging:`` block
    (``entity``, ``save_code``, ...) through to the backends that know them.
    """

    project: str | None = None
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    #: wandb's values: None, "allow", "must", "never".
    resume: str | bool | None = None
    mode: str | None = None
    #: Where a backend may keep local files (wandb's run directory).
    dir: str | None = None
    job_type: str | None = None
    system: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class Tracker(ABC):
    """A backend, registered under the ``solarian.trackers`` entry-point group.

    Constructed once per run, on the process that logs (rank 0), and must not
    raise once constructed: a tracking outage should not end a training run.
    """

    #: Entry-point name, e.g. ``"wandb"`` or ``"solarian_base"``.
    name: ClassVar[str]

    def __init__(self, spec: RunSpec):
        self.spec = spec

    @abstractmethod
    def log(self, values: dict[str, Any], step: int | None) -> None:
        """Record one row of metrics at ``step``."""

    def update_summary(self, values: dict[str, Any]) -> None:
        """Set run-level values (best scores, final numbers)."""

    def finish(self, exit_code: int) -> None:
        """Close the run. ``exit_code`` 0 is success."""

    @property
    def url(self) -> str | None:
        return None

    @classmethod
    def forward_env(cls) -> dict[str, str]:
        """Environment a job on another machine needs to log here.

        ``solarian modal-train`` copies it into the container. Raise ValueError
        when the local credentials it needs are missing.
        """
        return {}
