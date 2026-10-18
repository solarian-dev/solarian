"""Weights & Biases, as solarian trainers have always logged to it."""

from __future__ import annotations

import netrc
import os
import re
from glob import glob
from pathlib import Path
from typing import Any

from solarian.tracking.tracker import RunSpec, Tracker


def wandb_netrc_key() -> str | None:
    """The api.wandb.ai key ``wandb login`` writes to ~/.netrc."""
    try:
        auth = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError):
        return None
    return auth[2] if auth else None


def run_id_from_dir(run_dir: str) -> str:
    """The id of the last wandb run under ``run_dir``, to resume it."""
    paths = glob(str(Path(run_dir) / "wandb/latest-run/run-*"))
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1]) if len(paths) == 1 else None
    if match is None:
        raise ValueError(f"No previous wandb run under {run_dir} to resume.")
    return match.group(1)


class WandbTracker(Tracker):
    name = "wandb"

    def __init__(self, spec: RunSpec):
        super().__init__(spec)
        import wandb

        run_id = spec.id
        if spec.resume and not run_id and spec.dir:
            run_id = run_id_from_dir(spec.dir)
        kwargs = {
            "project": spec.project,
            "name": spec.name,
            "group": spec.group,
            "tags": spec.tags or None,
            "notes": spec.notes,
            "config": spec.config,
            "id": run_id,
            "resume": spec.resume or None,
            "mode": spec.mode,
            "dir": spec.dir,
            "job_type": spec.job_type,
            "save_code": False,
            **spec.extra,
        }
        self.run = wandb.init(**{k: v for k, v in kwargs.items() if v is not None})

    def log(self, values: dict[str, Any], step: int | None) -> None:
        self.run.log(values, step=step)

    def update_summary(self, values: dict[str, Any]) -> None:
        self.run.summary.update(values)

    def finish(self, exit_code: int) -> None:
        self.run.finish(exit_code=exit_code)

    @property
    def url(self) -> str | None:
        return self.run.url

    @classmethod
    def forward_env(cls) -> dict[str, str]:
        if os.environ.get("WANDB_MODE") in ("offline", "disabled"):
            return {"WANDB_MODE": os.environ["WANDB_MODE"]}
        key = os.environ.get("WANDB_API_KEY") or wandb_netrc_key()
        if not key:
            raise ValueError(
                "logging to wandb needs WANDB_API_KEY, or a `wandb login` entry in ~/.netrc."
            )
        return {"WANDB_API_KEY": key}
