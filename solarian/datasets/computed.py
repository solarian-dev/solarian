"""Computed datasets: a tree of versions over one source session, each made by an op.

::

    data/cups-pick/
        computed.json          the source and every version: parent, op, params, state, review
        versions/03-track/     a version (see :mod:`solarian.core.recording.archive`)

A version is built by one op from its parent: the source, or another version.
It stays *open* while its episodes can still be redone and reviewed one at a
time (:class:`~solarian.datasets.op_session.OpSession`), and is *closed* once
they cannot. Ops apply only on top of closed versions, so nothing a version
reads through to changes under it. ``export`` flattens a version into a
standalone session; ``drop`` deletes a version and everything built on it.

``computed.json`` holds every op, its params and every per-episode redo, which
is what :meth:`ComputedDataset.rebuild` replays: the file is the dataset's code.

Review state is keyed by the episodes the review shows: the version's own for
a MAP op (episode ``k`` is the ``k``-th of ``inputs``), the parent's for a
FILTER op, whose dropped episodes the version no longer has.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

from solarian.core.recording.archive import EpisodeRef, Session, SessionWriter, materialize
from solarian.datasets.op import OpKind
from solarian.datasets.runner import Progress, get_op, op_record, run_op, validate_params

COMPUTED_FILE = "computed.json"
VERSIONS_DIR = "versions"
FORMAT = 1
#: The key ``apply`` and ``resolve`` take for the source session.
SOURCE = "source"

PROCESSING, OPEN, CLOSED = "processing", "open", "closed"
#: Review status of one episode of an open version.
PENDING, DONE, ACCEPTED = "pending", "done", "accepted"

#: Kinds whose episodes can be redone or decided one at a time.
REVIEWABLE = (OpKind.MAP, OpKind.FILTER)

_NUMBER = re.compile(r"^(\d+)-")


def split_ref(ref: str | os.PathLike) -> tuple[Path, str | None]:
    """``data/cups-pick@03-track`` -> ``(data/cups-pick, "03-track")``."""
    text = str(ref)
    if "@" not in text:
        return Path(text), None
    path, _, key = text.rpartition("@")
    return Path(path), key or None


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def summarize(path: Path) -> dict:
    """Episodes, frames and the bytes stored in the version itself (not read through)."""
    session = Session(path)
    episodes = session.episodes()
    stored = sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.name.startswith("."))
    return {
        "episodes": len(episodes),
        "frames": sum(int(session.episode_meta(i)["n_rows"]) for i in episodes),
        "bytes": stored,
    }


class ComputedDataset:
    """A computed dataset directory; see the module docstring."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path).expanduser().resolve()
        self.file = self.path / COMPUTED_FILE
        if not self.file.is_file():
            raise FileNotFoundError(f"{self.path} is not a computed dataset (no {COMPUTED_FILE})")

    @classmethod
    def create(cls, source: str | os.PathLike, path: str | os.PathLike) -> ComputedDataset:
        session = Session(source)
        if session.version is not None:
            raise ValueError(f"{session.path} is a version; a computed dataset starts from a standalone session")
        path = Path(path).expanduser().resolve()
        if path.exists():
            raise FileExistsError(f"{path} exists")
        (path / VERSIONS_DIR).mkdir(parents=True)
        _write_json(path / COMPUTED_FILE, {
            "format": FORMAT,
            "source": os.path.relpath(session.path, path),
            "created_at": time.time(),
            "versions": {},
        })
        return cls(path)

    def __repr__(self) -> str:
        return f"ComputedDataset({str(self.path)!r})"

    # ── reading ─────────────────────────────────────────────────────

    def read(self) -> dict:
        return json.loads(self.file.read_text())

    @contextlib.contextmanager
    def edit(self) -> Iterator[dict]:
        """``computed.json``, locked against other writers, saved when the block ends."""
        with open(self.path / ".computed.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = self.read()
            yield data
            _write_json(self.file, data)

    @property
    def source(self) -> Path:
        return (self.path / self.read()["source"]).resolve()

    def versions(self) -> dict[str, dict]:
        return self.read()["versions"]

    def entry(self, key: str) -> dict:
        versions = self.versions()
        if key not in versions:
            raise KeyError(f"{self.path.name} has no version {key!r}; it has: {', '.join(versions) or 'none'}")
        return versions[key]

    def version_path(self, key: str) -> Path:
        return self.path / VERSIONS_DIR / key

    def session(self, key: str | None) -> Session:
        """The source for None, else version ``key``."""
        if key is None:
            return Session(self.source)
        self.entry(key)
        return Session(self.version_path(key))

    def leaves(self) -> list[str]:
        versions = self.versions()
        parents = {v["parent"] for v in versions.values()}
        return [k for k in versions if k not in parents]

    def resolve(self, key: str | None) -> str | None:
        """``key`` checked, ``source`` as None, or with no key the only leaf (None if no versions)."""
        if key == SOURCE:
            return None
        if key is not None:
            self.entry(key)
            return key
        leaves = self.leaves()
        if len(leaves) > 1:
            raise ValueError(
                f"{self.path.name} has several leaves ({', '.join(leaves)}); name one as {self.path.name}@<version>")
        return leaves[0] if leaves else None

    def subtree(self, key: str) -> list[str]:
        """``key`` and every version built on it, parents before children."""
        versions = self.versions()
        out, frontier = [key], [key]
        while frontier:
            frontier = [k for k, v in versions.items() if v["parent"] in frontier]
            out.extend(frontier)
        return out

    # ── building ────────────────────────────────────────────────────

    def apply(
        self,
        op_name: str,
        parent: str | None = None,
        params: Mapping | None = None,
        *,
        extra: Sequence[str] = (),
        review: bool = False,
        online: bool = False,
        workers: int = 4,
        progress: Progress | None = None,
        log: Callable[[str], None] | None = None,
        reserved: Callable[[list[str]], None] | None = None,
    ) -> list[str]:
        """Run ``op_name`` on ``parent`` (None: the source); returns the new version keys.

        ``extra`` are further inputs for a COMBINE op: version keys of this
        dataset or session paths. ``review`` leaves a MAP or FILTER version
        open for review; ``online`` also leaves its episodes unprocessed, to
        be done one at a time by an :class:`~solarian.datasets.op_session.OpSession`
        (a FILTER op's decisions are cheap, so they are made up front either way).
        ``reserved`` gets the keys once they are taken, before the op runs.
        """
        op = get_op(op_name)
        if (review or online) and op.kind not in REVIEWABLE:
            raise ValueError(f"{op.name} is a {op.kind.value} op; only map and filter ops are reviewed per episode")
        if parent is not None and self.entry(parent)["state"] != CLOSED:
            raise ValueError(f"{parent} is still open; close it before building on it")
        params = dict(params or {})
        validated = validate_params(op, params)
        extra_refs = [self._input_ref(ref) for ref in extra]
        inputs = [self.session(parent), *(self._input(ref) for ref in extra_refs)]

        with self.edit() as data:
            number = 1 + max((int(m.group(1)) for k in data["versions"] if (m := _NUMBER.match(k))), default=0)
            keys = {out: f"{number:02d}-{op.name}" + ("" if len(op.outputs) == 1 else f"-{out}") for out in op.outputs}
            for out, key in keys.items():
                data["versions"][key] = {
                    "parent": parent,
                    "extra": extra_refs,
                    "op": op.name,
                    "op_source": op.source,
                    "op_commit": op.commit,
                    "params": params,
                    "output": out,
                    "state": PROCESSING,
                    "online": online and op.kind is OpKind.MAP,
                    "inputs": inputs[0].episodes(),
                    "episodes": {},
                    "created_at": time.time(),
                }
        try:
            if reserved is not None:
                reserved(list(keys.values()))
            if online and op.kind is OpKind.MAP:
                self._write_passthrough(keys[op.outputs[0]], op, validated, inputs[0])
            else:
                run_op(op.name, [s.path for s in inputs], {out: self.version_path(k) for out, k in keys.items()},
                       params, workers=workers, progress=progress, log=log)
        except BaseException:
            with self.edit() as data:
                for key in keys.values():
                    data["versions"].pop(key, None)
            for key in keys.values():
                shutil.rmtree(self.version_path(key), ignore_errors=True)
            raise
        with self.edit() as data:
            for key in keys.values():
                entry = data["versions"][key]
                entry["state"] = OPEN if review or online else CLOSED
                entry["episodes"] = self._initial_review(op.kind, entry, key, online)
                entry["summary"] = summarize(self.version_path(key))
        return list(keys.values())

    def _input_ref(self, ref: str) -> str:
        """A version key as is; a path relative to this dataset, so the file moves with it."""
        if ref in self.versions():
            return ref
        return os.path.relpath(Path(ref).expanduser().resolve(), self.path)

    def _input(self, ref: str) -> Session:
        versions = self.versions()
        if ref in versions:
            if versions[ref]["state"] != CLOSED:
                raise ValueError(f"{ref} is still open; close it before building on it")
            return self.session(ref)
        return Session(self.path / ref)

    def _write_passthrough(self, key: str, op, params, parent: Session) -> None:
        """A version whose every episode is still its parent's, for an online MAP op to fill in."""
        with SessionWriter(self.version_path(key), parents=[parent], op={**op_record(op, params, [parent]),
                                                                          "output": op.outputs[0]}) as writer:
            for index in parent.episodes():
                writer.add_episode(parent.episode_meta(index), source=EpisodeRef(parent, index))

    def _initial_review(self, kind: OpKind, entry: dict, key: str, online: bool) -> dict:
        if entry["state"] == CLOSED:
            return {}
        if kind is OpKind.MAP:
            status = PENDING if online else DONE
            return {str(k): {"status": status} for k in range(len(entry["inputs"]))}
        kept = {src[1] for src in (e["src"] for e in Session(self.version_path(key)).version["episodes"]) if src}
        return {str(i): {"status": PENDING, "keep": i in kept} for i in entry["inputs"]}

    def accept(self, key: str, episode: int) -> None:
        """Episode ``episode`` of open version ``key`` is right as it is."""
        with self.edit() as data:
            entry = data["versions"].get(key)
            if entry is None or entry["state"] != OPEN:
                raise ValueError(f"{key} is not an open version of {self.path.name}")
            state = entry["episodes"].get(str(episode))
            if state is None:
                raise ValueError(f"{key} has no episode {episode} to review")
            state["status"] = ACCEPTED

    def close(self, key: str) -> None:
        with self.edit() as data:
            entry = data["versions"].get(key)
            if entry is None:
                raise KeyError(f"{self.path.name} has no version {key!r}")
            if entry["state"] != OPEN:
                raise ValueError(f"{key} is {entry['state']}, not open")
            entry["state"] = CLOSED
            entry["closed_at"] = time.time()
        shutil.rmtree(self.version_path(key) / ".work", ignore_errors=True)

    def drop(self, key: str) -> list[str]:
        """Delete ``key`` and every version built on it; returns the keys deleted."""
        self.entry(key)
        doomed = self.subtree(key)
        with self.edit() as data:
            for k in doomed:
                data["versions"].pop(k, None)
        for k in doomed:
            shutil.rmtree(self.version_path(k), ignore_errors=True)
        return doomed

    def export(self, key: str | None, output: str | os.PathLike, force: bool = False) -> Path:
        """Version ``key`` (None: the source) flattened into a standalone session at ``output``."""
        return materialize(self.session(key), output, force=force)

    def rebuild(self, into: str | os.PathLike, log: Callable[[str], None] | None = None) -> ComputedDataset:
        """Replay every op and per-episode redo of this dataset into a new one at ``into``."""
        from solarian.datasets.op_session import OpSession

        log = log or (lambda message: None)
        target = ComputedDataset.create(self.source, into)
        renamed: dict[str, str] = {}
        applications: dict[str, list[tuple[str, dict]]] = {}
        for key, entry in sorted(self.versions().items(), key=lambda item: item[1]["created_at"]):
            applications.setdefault(_NUMBER.match(key).group(1), []).append((key, entry))
        for group in applications.values():
            key, entry = group[0]
            log(f"{key}: {entry['op']}")
            parent = renamed[entry["parent"]] if entry["parent"] is not None else None
            extra = [renamed.get(ref, str((self.path / ref).resolve())) for ref in entry["extra"]]
            new_keys = target.apply(entry["op"], parent, entry["params"], extra=extra,
                                    review=entry["state"] != CLOSED or bool(entry["episodes"]),
                                    online=entry["online"], log=log)
            for (old, old_entry), new in zip(group, new_keys):
                renamed[old] = new
                if old_entry["episodes"]:
                    session = OpSession(target, new, log=log)
                    session.replay(old_entry["episodes"])
                if old_entry["state"] == CLOSED and target.entry(new)["state"] == OPEN:
                    target.close(new)
        return target
