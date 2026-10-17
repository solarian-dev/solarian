"""One open version's op, set up once, run on its episodes one at a time.

What online review runs on: the op's ``setup`` (a SAM 2 model, say) happens
once, then each ``redo`` reprocesses one episode with per-episode params over
the version's, in about the time the op takes on that episode alone. A MAP
op's episode is rewritten in place; a FILTER op's keep/drop decision rebuilds
the version, which holds only episode metadata. Every change is recorded in
``computed.json`` under the version's ``episodes``, so ``rebuild`` replays it.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Callable, Mapping

from solarian.core.recording.archive import EPISODE_FILE, EpisodeRef, Session, SessionWriter, episode_dir
from solarian.datasets.computed import ACCEPTED, DONE, OPEN, ComputedDataset, summarize
from solarian.datasets.op import Episode, OpContext, OpKind
from solarian.datasets.runner import get_op, validate_params


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class OpSession:
    """Redo and review the episodes of open version ``key`` of ``computed``."""

    def __init__(
        self,
        computed: ComputedDataset,
        key: str,
        log: Callable[[str], None] | None = None,
        preview: Callable[[dict], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ):
        """``preview`` and ``cancelled`` reach the op through its :class:`OpContext`;
        a redo the op stops raises :class:`~solarian.datasets.op.Cancelled` and changes nothing."""
        entry = computed.entry(key)
        if entry["state"] != OPEN:
            raise ValueError(f"{key} is {entry['state']}; only an open version is reviewed")
        self.computed = computed
        self.key = key
        self.op_cls = get_op(entry["op"])
        self.base = entry["params"]
        self.inputs: list[int] = entry["inputs"]
        self.parent = computed.session(entry["parent"])
        self.path = computed.version_path(key)
        scratch = self.path / ".work"
        scratch.mkdir(exist_ok=True)
        self.ctx = OpContext(
            inputs=[self.parent], scratch=scratch, log=log or (lambda message: None),
            preview=preview or (lambda update: None), cancelled=cancelled or (lambda: False),
        )
        self.op = self.op_cls()
        self.op.setup(validate_params(self.op_cls, self.base), self.ctx)

    @property
    def kind(self) -> OpKind:
        return self.op_cls.kind

    def _record(self, episode: int, **state) -> None:
        with self.computed.edit() as data:
            entry = data["versions"][self.key]
            entry["episodes"][str(episode)] = {**entry["episodes"].get(str(episode), {}), **state}
            entry["summary"] = summarize(self.path)

    def redo(self, episode: int, params: Mapping | None = None) -> dict:
        """MAP: reprocess the version's episode ``episode`` with ``params`` over the version's.

        Returns what changed in its ``episode.json`` (``targets`` coverage, say).
        """
        if self.kind is not OpKind.MAP:
            raise ValueError(f"{self.op_cls.name} is a {self.kind.value} op; decide its episodes with keep()")
        if not 0 <= episode < len(self.inputs):
            raise ValueError(f"{self.key} has episodes 0-{len(self.inputs) - 1}, not {episode}")
        overrides = dict(params or {})
        validated = validate_params(self.op_cls, {**self.base, **overrides})
        source = Episode(self.parent, self.inputs[episode])
        out = self.op.map(source, validated, self.ctx) or source.derive()

        prefix = self.path / episode_dir(episode)
        # The previous run's members go: a member it wrote and this one did not
        # must read through to the parent again.
        for old in [p for p in prefix.rglob("*") if p.is_file() and p.name != EPISODE_FILE]:
            old.unlink()
        for rel, data in out.files.items():
            if isinstance(data, (bytes, bytearray)):
                _write_atomic(prefix / rel, bytes(data))
            else:
                (prefix / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(data, prefix / rel)
        meta = {**out.meta, "index": episode}
        _write_atomic(prefix / EPISODE_FILE, json.dumps(meta, indent=2).encode())
        changed = {name: value for name, value in out.meta.items()
                   if name != "index" and source.meta.get(name) != value}
        self._record(episode, status=DONE, params=overrides, result=changed)
        return changed

    def keep(self, episode: int, keep: bool) -> None:
        """FILTER: keep or drop the parent's episode ``episode``, overriding the op's decision."""
        if self.kind is not OpKind.FILTER:
            raise ValueError(f"{self.op_cls.name} is a {self.kind.value} op; redo() its episodes")
        if episode not in self.inputs:
            raise ValueError(f"{self.parent.path.name} has no episode {episode}")
        with self.computed.edit() as data:
            entry = data["versions"][self.key]
            entry["episodes"][str(episode)] = {"status": ACCEPTED, "keep": keep}
            kept = [i for i in self.inputs if entry["episodes"][str(i)]["keep"]]
            op = Session(self.path).version["op"]
            with SessionWriter(self.path, parents=[self.parent], force=True, op=op) as writer:
                for index in kept:
                    writer.add_episode(self.parent.episode_meta(index), source=EpisodeRef(self.parent, index))
            entry["summary"] = summarize(self.path)

    def replay(self, episodes: Mapping[str, dict]) -> None:
        """Repeat a recorded review: every redo with its params, every decision, every accept."""
        for key, state in episodes.items():
            episode = int(key)
            if self.kind is OpKind.FILTER:
                if state.get("status") == ACCEPTED:
                    self.keep(episode, state["keep"])
                continue
            if "params" in state:
                self.redo(episode, state["params"])
            if state.get("status") == ACCEPTED:
                self.computed.accept(self.key, episode)
