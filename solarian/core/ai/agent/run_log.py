"""One agent run on disk: what it was asked, saw, said and did.

One run is one episode, and the agent's side of it is part of the controller's
recording of that episode, beside the table and video the controller kept::

    <project>/recordings/controller/<session>/episodes/<episode>/agent/
    ├── run.json        task, models, tools, status; rewritten as it changes
    ├── events.jsonl    every message, tool call, action transition, monitor answer
    ├── turns/<k>.json  each planner call: the messages it added, and the reply
    └── images/         every image the planner or the monitor was shown

A run the controller does not record is not written anywhere. Everything is
stamped with the synced clock the episode's rows use, so an event lines up
with the frame it was about.
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

from solarian.core.ai.agent.model import (
    AssistantTurn,
    ImagePart,
    Message,
    Part,
    UserTurn,
)
from solarian.core.clock_sync import synced_time
from solarian.core.recording.session import EPISODES_DIR

logger = logging.getLogger(__name__)

AGENT_DIR = "agent"


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


class RunLog:
    def __init__(self, directory: Path, meta: dict):
        self.dir = directory
        (self.dir / "turns").mkdir(parents=True, exist_ok=True)
        (self.dir / "images").mkdir(parents=True, exist_ok=True)
        self._meta = dict(meta)
        self._lock = threading.Lock()
        self._turns = 0
        self._write_meta()

    def _write_meta(self) -> None:
        tmp = self.dir / "run.json.tmp"
        tmp.write_text(json.dumps(self._meta, indent=2, default=_json_default))
        tmp.replace(self.dir / "run.json")

    def update(self, **meta) -> None:
        with self._lock:
            self._meta.update(meta)
            self._write_meta()

    def event(self, kind: str, **data) -> None:
        line = json.dumps({"t": _stamp(), "kind": kind, **data}, default=_json_default)
        with self._lock, open(self.dir / "events.jsonl", "a") as f:
            f.write(line + "\n")

    def image(self, part: ImagePart) -> str:
        """Save ``part`` once; returns the file name events and turns refer to it by."""
        name = _image_file(part.label)
        path = self.dir / "images" / name
        if not path.exists():
            path.write_bytes(part.jpeg)
        return name

    def _part(self, part: Part) -> dict:
        if isinstance(part, str):
            return {"text": part}
        return {"image": self.image(part), "label": part.label}

    def message(self, message: Message) -> dict:
        if isinstance(message, UserTurn):
            return {
                "role": "user",
                "content": [self._part(p) for p in message.content],
                "tool_results": [
                    {
                        "call_id": r.call_id,
                        "is_error": r.is_error,
                        "content": [self._part(p) for p in r.content],
                    }
                    for r in message.tool_results
                ],
            }
        assert isinstance(message, AssistantTurn)
        return {
            "role": "assistant",
            "text": message.text,
            "tool_calls": [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in message.tool_calls
            ],
            "stop_reason": message.stop_reason,
            "usage": message.usage.to_dict(),
        }

    def turn(self, added: list[Message], reply: AssistantTurn, latency_s: float) -> None:
        """One planner call: the messages added since the previous call, and the reply."""
        with self._lock:
            self._turns += 1
            k = self._turns
        payload = {
            "t": _stamp(),
            "latency_s": round(latency_s, 3),
            "input": [self.message(m) for m in added],
            "output": self.message(reply),
        }
        (self.dir / "turns" / f"{k:04d}.json").write_text(
            json.dumps(payload, indent=1, default=_json_default)
        )


def merge_into_session(root: Path, session_dir: Path) -> list[Path]:
    """Copy each episode's ``agent/`` log from the controller's recording into the robot's.

    *root* is the controller's recordings root; the session there has the
    same directory name as *session_dir* when both sides recorded it. Run
    where both are reachable — after the session is transferred off the
    robot, before it is uploaded. Returns the directories written.
    """
    source = root / session_dir.name / EPISODES_DIR
    if not source.is_dir():
        return []
    written = []
    for episode in sorted(p for p in source.iterdir() if (p / AGENT_DIR).is_dir()):
        target = session_dir / EPISODES_DIR / episode.name
        if not target.is_dir():
            logger.warning("Agent log for episode %s has no recorded episode in %s", episode.name, session_dir)
            continue
        shutil.copytree(episode / AGENT_DIR, target / AGENT_DIR, dirs_exist_ok=True)
        written.append(target / AGENT_DIR)
    return written
