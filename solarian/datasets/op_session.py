"""One open version's op, set up once, run on its episodes one at a time.

What online review runs on: the op's ``setup`` (a SAM 2 model, say) happens
once, then each ``redo`` reprocesses one episode with per-episode params over
the version's, in about the time the op takes on that episode alone. A MAP
op's episode is rewritten in place. ``keep`` decides whether an episode stays:
a FILTER op's parent episode, overriding the op, or a MAP op's own one, found
not worth keeping while it was reviewed; the decisions are written into the
version when it closes. Every change is recorded in ``computed.json`` under
the version's ``episodes``, so ``rebuild`` replays it.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Callable, Mapping

from solarian.core.recording.archive import EPISODE_FILE, episode_dir
from solarian.datasets.computed import ACCEPTED, DONE, OPEN, ComputedDataset, summarize
from solarian.datasets.op import Episode, OpContext, OpKind
from solarian.datasets.runner import get_op, validate_params


def _stored_bytes(prefix: Path) -> int:
    """Bytes under ``prefix`` as :func:`summarize` counts them."""
    return sum(p.stat().st_size for p in prefix.rglob("*") if p.is_file() and not p.name.startswith("."))


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
        stored_before = _stored_bytes(prefix)
        own_meta = prefix / EPISODE_FILE
        rows_before = int((json.loads(own_meta.read_bytes()) if own_meta.is_file() else source.meta)["n_rows"])
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
        with self.computed.edit() as data:
            entry = data["versions"][self.key]
            previous = entry["episodes"].get(str(episode), {})
            # A redo keeps the episode, even one dropped before.
            dropped = previous.get("keep") is False
            entry["episodes"][str(episode)] = {
                **{k: v for k, v in previous.items() if k != "keep"},
                "status": DONE, "params": overrides, "result": changed}
            summary = entry.get("summary")
            if summary:
                # Only this episode changed: summarize() reads every episode, seconds of NFS a redo.
                if dropped:
                    summary["episodes"] += 1
                    summary["frames"] += int(meta["n_rows"])
                else:
                    summary["frames"] += int(meta["n_rows"]) - rows_before
                summary["bytes"] += _stored_bytes(prefix) - stored_before
            else:
                entry["summary"] = summarize(self.path)
        return changed

    def keep(self, episode: int, keep: bool) -> None:
        """Keep or drop ``episode``, and accept the decision.

        FILTER: the parent's episode, overriding the op's decision. MAP: the
        version's own episode, redone or not; a dropped one is left out on close.
        """
        if self.kind is OpKind.FILTER:
            if episode not in self.inputs:
                raise ValueError(f"{self.parent.path.name} has no episode {episode}")
            rows = int(self.parent.episode_meta(episode)["n_rows"])
        else:
            if not 0 <= episode < len(self.inputs):
                raise ValueError(f"{self.key} has episodes 0-{len(self.inputs) - 1}, not {episode}")
            rows = int(json.loads((self.path / episode_dir(episode) / EPISODE_FILE).read_bytes())["n_rows"])
        with self.computed.edit() as data:
            entry = data["versions"][self.key]
            state = entry["episodes"][str(episode)]
            # A MAP episode is kept unless it says otherwise.
            was = state.get("keep", True)
            if self.kind is OpKind.FILTER:
                entry["episodes"][str(episode)] = {"status": ACCEPTED, "keep": keep}
            else:
                state = {k: v for k, v in state.items() if k != "keep"}
                entry["episodes"][str(episode)] = {**state, "status": ACCEPTED, **({} if keep else {"keep": False})}
            if keep != was:
                # The version's files are rewritten on close; until then only its totals follow.
                sign = 1 if keep else -1
                entry["summary"]["episodes"] += sign
                entry["summary"]["frames"] += sign * rows

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
            if state.get("keep") is False:
                self.keep(episode, False)
            elif state.get("status") == ACCEPTED:
                self.computed.accept(self.key, episode)
