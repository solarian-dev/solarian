"""Tracking for trainers built on Hugging Face Accelerate.

The vendored diffusion-policy workspaces log through ``accelerator.log``. This
gives them the same backends as everything else::

    log_with, init_kwargs = accelerate_trackers(logging_cfg)
    accelerator = Accelerator(log_with=log_with, ...)
    accelerator.init_trackers(project_name=logging_cfg["project"], config=cfg, init_kwargs=init_kwargs)
"""

from __future__ import annotations

from typing import Any, Mapping

from accelerate.tracking import GeneralTracker

from solarian.tracking.registry import resolve_backends
from solarian.tracking.run import Run, init

#: Keys of a Hydra ``logging:`` block that map onto tracking.init's arguments.
_INIT_KEYS = ("project", "name", "group", "tags", "notes", "id", "resume", "mode", "dir", "job_type")


class SolarianTracker(GeneralTracker):
    """One Accelerate tracker standing for every solarian backend of the run.

    Accelerate calls it on the main process only. The run starts when the
    config arrives, so it is recorded with it; a ``log`` before that starts it
    without one.
    """

    name = "solarian"
    requires_logging_directory = False

    def __init__(self, logging_cfg: Mapping[str, Any], backends: list[str]):
        super().__init__()
        self.logging_cfg = dict(logging_cfg)
        self.backends = backends
        self.run: Run | None = None

    @property
    def tracker(self) -> Run | None:
        return self.run

    def _start(self, config: Mapping[str, Any] | None) -> Run:
        if self.run is None:
            kwargs = {k: self.logging_cfg[k] for k in _INIT_KEYS if k in self.logging_cfg}
            extra = {k: v for k, v in self.logging_cfg.items() if k not in _INIT_KEYS}
            self.run = init(**kwargs, **extra, config=config, backend=self.backends)
        return self.run

    def store_init_configuration(self, values: dict) -> None:
        self._start(values)

    def log(self, values: dict, step: int | None = None, **kwargs) -> None:
        self._start(None).log(values, step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish(0)


def accelerate_trackers(logging_cfg: Mapping[str, Any]) -> tuple[list[GeneralTracker], dict]:
    """``(log_with, init_kwargs)`` for a Hydra ``logging:`` block.

    ``logging.backend`` picks the backends, defaulting to solarian_base; the
    other keys are wandb.init's.
    """
    cfg = dict(logging_cfg)
    backends = resolve_backends(cfg.pop("backend", None))
    if not backends:
        return [], {}
    return [SolarianTracker(cfg, backends)], {}
