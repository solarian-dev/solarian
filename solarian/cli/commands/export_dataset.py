"""Turn a recorded session into a training dataset."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry
from solarian.utils import setup_process_logging


@CommandRegistry.register("export-dataset", "export-dataset")
class ExportDatasetCommand(BaseCommand):
    help = "Convert a recorded session into a LeRobot dataset"
    description = """
    Reads a session directory the robot wrote — archive mp4s plus one parquet per
    episode at native rates — and writes a LeRobot v2.1 dataset: every signal
    resampled onto one grid, every frame warped through the camera's optical remap
    and downscaled to the observation resolution.

    The session is never modified, so this is cheap to redo: recalibrate a camera
    or change your mind about the rate and export again from the same recording.
    Pass --no-remap to keep the archive geometry, which is what a higher-resolution
    experiment or a re-calibration wants.

    The output is what `solarian train` reads and what PrimaryKey ingests.
    """
    usage = (
        "<session_dir> [<output_dir>] [--fps N] [--episodes A,B,C] "
        "[--no-remap] [--overwrite]"
    )
    options = {
        "<session_dir>": "Session to read, e.g. datasets/cup_stacking__2026-09-22T19-04-11.",
        "<output_dir>": "Where to write. Defaults to <session_dir>/export/lerobot.",
        "--fps N": "Rate to resample onto. Defaults to the session's video grid.",
        "--episodes A,B,C": "Only these episode indices. Defaults to all sealed ones.",
        "--no-remap": "Keep the archive geometry instead of producing policy frames.",
        "--overwrite": "Replace the output directory if it exists.",
    }
    examples = (
        "solarian export-dataset datasets/cup_stacking__2026-09-22T19-04-11",
        "solarian export-dataset datasets/session out/cup_v3 --fps 30 --overwrite",
        "solarian export-dataset datasets/session --episodes 0,1,4 --no-remap",
    )

    def handle(self, cfg: DictConfig) -> None:
        setup_process_logging(level=logging.INFO, logger_name=None)

        from solarian.core.recording import export_lerobot

        session = Path(os.environ["SOLARIAN_EXPORT_SESSION"])
        out = Path(
            os.environ.get("SOLARIAN_EXPORT_OUT") or session / "export" / "lerobot"
        )
        fps = os.environ.get("SOLARIAN_EXPORT_FPS")
        episodes = os.environ.get("SOLARIAN_EXPORT_EPISODES")

        stats = export_lerobot(
            session,
            out,
            fps=float(fps) if fps else None,
            episodes=[int(e) for e in episodes.split(",")] if episodes else None,
            remap=os.environ.get("SOLARIAN_EXPORT_NO_REMAP") != "true",
            overwrite=os.environ.get("SOLARIAN_EXPORT_OVERWRITE") == "true",
        )

        print(
            f"\nWrote {stats.n_episodes} episodes, {stats.n_frames} frames and "
            f"{stats.n_videos} videos to\n  {stats.path}"
        )
