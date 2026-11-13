"""What happened during a recorded episode, beside its table and video.

The controller keeps one log per episode it records, and everyone who acts on
the robot writes into it: the controller its skill and service calls, tagged
``by`` the agent or the operator; the agentic controller what it was asked, saw,
said and did; the recorder each camera's tracker overlays::

    <project>/recordings/controller/<session>/episodes/<episode>/logs/
    ├── events.jsonl            every event, on the synced clock
    ├── turns/<k>.json          each planner call: the messages it added, and the reply
    ├── images/                 every image the planner or the monitor was shown
    └── overlays/<camera>.jsonl the boxes, points and masks drawn over each frame

A sealed episode's log stays open until the next one starts, so what happens
between takes lands after the end of the last video. Everything is stamped
with the synced clock the episode's rows use, so an event lines up with the
frame it was about. Older sessions have an ``agent/`` directory instead,
holding only the agentic controller's side.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from solarian.core.clock_sync import synced_time
from solarian.core.recording.session import EPISODES_DIR

logger = logging.getLogger(__name__)

LOGS_DIR = "logs"
#: What sessions recorded before ``logs/`` keep the agentic controller's runs in.
LEGACY_AGENT_DIR = "agent"


def _image_file(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") + ".jpg"


def _stamp() -> str:
    """The synced clock as local ISO 8601 with milliseconds and offset: readable, and
    ``datetime.fromisoformat(...).timestamp()`` gives back the episode rows' ``wall_ts``."""
    return datetime.fromtimestamp(synced_time()).astimezone().isoformat(timespec="milliseconds")


def _json_default(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


class EpisodeLog:
    """Safe from any thread. Writes after the episode was dropped are lost, not an error."""

    def __init__(self, directory: Path):
        self.dir = directory
        (self.dir / "images").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._turns = 0

    def _append(self, path: Path, line: str) -> None:
        try:
            with self._lock, open(path, "a") as f:
                f.write(line + "\n")
        except FileNotFoundError:
            logger.debug("%s is gone; the episode was dropped", path.parent)

    def event(self, kind: str, **data) -> None:
        self._append(self.dir / "events.jsonl",
                     json.dumps({"t": _stamp(), "kind": kind, **data}, default=_json_default))

    def image(self, label: str, jpeg: bytes) -> str:
        """Save an image once; returns the file name events and turns refer to it by."""
        name = _image_file(label)
        path = self.dir / "images" / name
        if not path.exists():
            try:
                path.write_bytes(jpeg)
            except FileNotFoundError:
                logger.debug("%s is gone; the episode was dropped", path.parent)
        return name

    def turn(self, payload: dict) -> str:
        """Save one planner call; returns its path in the log, numbered across the episode."""
        with self._lock:
            self._turns += 1
            name = f"turns/{self._turns:04d}.json"
        try:
            (self.dir / "turns").mkdir(exist_ok=True)
            (self.dir / name).write_text(json.dumps({"t": _stamp(), **payload}, indent=1, default=_json_default))
        except FileNotFoundError:
            logger.debug("%s is gone; the episode was dropped", self.dir)
        return name

    def overlay(self, camera: str, row: dict) -> None:
        """One frame's overlays on ``camera``, keyed by the frame's capture time."""
        directory = self.dir / "overlays"
        try:
            directory.mkdir(exist_ok=True)
        except FileNotFoundError:
            return
        self._append(directory / f"{camera}.jsonl", json.dumps(row, separators=(",", ":"), default=_json_default))


def merge_into_session(root: Path, session_dir: Path) -> list[Path]:
    """Copy each episode's log from the controller's recording into the robot's.

    *root* is the controller's recordings root; the session there has the
    same directory name as *session_dir* when both sides recorded it. Run
    where both are reachable — after the session is transferred off the
    robot, before it is uploaded. Returns the directories written.
    """
    source = root / session_dir.name / EPISODES_DIR
    if not source.is_dir():
        return []
    written = []
    for episode in sorted(p for p in source.iterdir() if p.is_dir()):
        logs = [episode / name for name in (LOGS_DIR, LEGACY_AGENT_DIR) if (episode / name).is_dir()]
        if not logs:
            continue
        target = session_dir / EPISODES_DIR / episode.name
        if not target.is_dir():
            logger.warning("Log for episode %s has no recorded episode in %s", episode.name, session_dir)
            continue
        for log in logs:
            shutil.copytree(log, target / log.name, dirs_exist_ok=True)
            written.append(target / log.name)
    return written
