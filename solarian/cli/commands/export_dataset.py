"""Turn a recorded session into a training dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry
from solarian.utils import setup_process_logging


@CommandRegistry.register("export-dataset", "export-dataset")
class ExportDatasetCommand(BaseCommand):
    category = "Training & Data"
    uses_config = False
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

    def parse_args(self, argv: list[str]) -> list[str]:
        parser = argparse.ArgumentParser(prog="solarian export-dataset")
        parser.add_argument("session_dir", help="Recorded session to convert.")
        parser.add_argument("output_dir", nargs="?", default=None,
                            help="Where to write (default: <session_dir>/export/lerobot).")
        parser.add_argument("--fps", default=None,
                            help="Rate to resample onto (default: the session's video grid).")
        parser.add_argument("--episodes", default=None, metavar="A,B,C",
                            help="Only these episode indices (default: all sealed ones).")
        parser.add_argument("--no-remap", action="store_true",
                            help="Keep the archive geometry instead of producing policy frames.")
        parser.add_argument("--overwrite", action="store_true",
                            help="Replace the output directory if it exists.")
        ns = parser.parse_args(argv)
        self.session_dir = ns.session_dir
        self.output_dir = ns.output_dir
        self.fps = ns.fps
        self.episodes = ns.episodes
        self.no_remap = ns.no_remap
        self.overwrite = ns.overwrite
        return []

    def handle(self, cfg: DictConfig) -> None:
        setup_process_logging(level=logging.INFO, logger_name=None)

        from solarian.core.recording import export_lerobot

        session = Path(self.session_dir)
        out = Path(self.output_dir or session / "export" / "lerobot")
        fps = self.fps
        episodes = self.episodes

        stats = export_lerobot(
            session,
            out,
            fps=float(fps) if fps else None,
            episodes=[int(e) for e in episodes.split(",")] if episodes else None,
            remap=not self.no_remap,
            overwrite=self.overwrite,
        )

        print(
            f"\nWrote {stats.n_episodes} episodes, {stats.n_frames} frames and "
            f"{stats.n_videos} videos to\n  {stats.path}"
        )
