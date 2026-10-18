"""Experiment tracking with wandb's surface and pluggable backends.

::

    from solarian import tracking

    run = tracking.init(project="umi", name="pick_side", tags=["bbox"], config=cfg)
    run.log({"train_loss": 0.04, "epoch": 3}, step=global_step)
    run.summary["best_val_loss"] = 0.01
    run.finish()

Where a run goes is configuration, not code: ``logging.backend`` in a train
config (``wandb``, ``solarian_base``, ``none``, or a list for several), and
``$SOLARIAN_TRACKING`` over it. Backends register under the
``solarian.trackers`` entry-point group; solarian ships ``wandb`` and ``none``,
and the solarian-base plugin adds ``solarian_base``.
"""

from solarian.tracking.registry import (
    DEFAULT_BACKEND,
    TRACKING_ENV,
    discover_trackers,
    forward_env,
    resolve_backends,
)
from solarian.tracking.run import Run, Summary, current, finish, init, log
from solarian.tracking.system import system_info
from solarian.tracking.tags import MAX_TAG_LENGTH, clean_tags
from solarian.tracking.tracker import RunSpec, Tracker

__all__ = [
    "DEFAULT_BACKEND",
    "MAX_TAG_LENGTH",
    "TRACKING_ENV",
    "Run",
    "RunSpec",
    "Summary",
    "Tracker",
    "clean_tags",
    "current",
    "discover_trackers",
    "finish",
    "forward_env",
    "init",
    "log",
    "resolve_backends",
    "system_info",
]
