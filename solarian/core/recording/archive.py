"""A recorded session as a read-only archive, and the writer that derives new ones.

A session is a directory or a ``.solarian.zip`` holding the layout of
:mod:`solarian.core.recording.session` plus the sidecars UMI training reads::

    session.json, manifest.jsonl
    episodes/000000/
        episode.json             index, n_rows, start_ts / end_ts, stats, umi demo poses
        episode.parquet          one row per frame
        video/<camera>.mp4
        tags/<camera>.json       per frame, {tag_id: 4x2 corners}
        targets/<camera>.npz     bbox (T, 4+) and/or bit-packed mask, in video pixels

A *version* is a session directory that also has ``version.json``: it names its
parent sessions and, per episode, the parent episode it starts from. Only what
the version changed is stored in it; every other member of an episode is read
from the parent, recursively. Deleting episodes therefore costs a few JSON
files, and tracking an object costs one npz per episode, however large the
videos are. :func:`materialize` flattens a version into a standalone session.
A version always holds its own ``episode.json`` and ``manifest.jsonl``, so a
reader that only lists episodes needs no layering at all.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator, Mapping, Sequence

import numpy as np

from solarian.core.recording.session import (
    EPISODE_FILE,
    EPISODES_DIR,
    MANIFEST_FILE,
    SESSION_FILE,
    TABLE_FILE,
    VIDEO_DIR,
)

VERSION_FILE = "version.json"
VERSION_FORMAT = 1
TAGS_DIR = "tags"
TARGETS_DIR = "targets"
SUFFIX = ".solarian.zip"
#: Already compressed, so stored: an mp4 member then decodes straight out of the zip.
STORED_SUFFIXES = (".mp4", ".parquet", ".npz")

_EPISODE_JSON = re.compile(rf"^{EPISODES_DIR}/(\d+)/{re.escape(EPISODE_FILE)}$")
_CHUNK = 1 << 20


def episode_dir(episode: int) -> str:
    return f"{EPISODES_DIR}/{episode:06d}"


def video_member(episode: int, camera: str) -> str:
    return f"{episode_dir(episode)}/{VIDEO_DIR}/{camera}.mp4"


def tags_member(episode: int, camera: str) -> str:
    return f"{episode_dir(episode)}/{TAGS_DIR}/{camera}.json"


def targets_member(episode: int, camera: str) -> str:
    return f"{episode_dir(episode)}/{TARGETS_DIR}/{camera}.npz"


def _compression(member: str) -> int:
    return zipfile.ZIP_STORED if member.endswith(STORED_SUFFIXES) else zipfile.ZIP_DEFLATED


def _episode_of(member: str) -> int | None:
    parts = member.split("/")
    if len(parts) > 2 and parts[0] == EPISODES_DIR and parts[1].isdigit():
        return int(parts[1])
    return None


class Session:
    """A session directory, ``.solarian.zip`` or version, read by member path."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path).expanduser().resolve()
        self.is_zip = self.path.is_file()
        if self.is_zip:
            if not zipfile.is_zipfile(self.path):
                raise ValueError(f"{self.path} is not a zip")
            with zipfile.ZipFile(self.path) as zf:
                own = {n for n in zf.namelist() if not n.endswith("/")}
        elif self.path.is_dir():
            # Dot entries are scratch space (a writer's .tmp, Finder's .DS_Store).
            own = {
                p.relative_to(self.path).as_posix()
                for p in self.path.rglob("*")
                if p.is_file() and not any(part.startswith(".") for part in p.relative_to(self.path).parts)
            }
        else:
            raise FileNotFoundError(self.path)
        self._own = own
        #: Inherited member -> (parent, member in the parent).
        self._route: dict[str, tuple[Session, str]] = {}
        self.version: dict | None = None
        self.parents: list[Session] = []
        if VERSION_FILE in own:
            self.version = self.read_json(VERSION_FILE)
            self.parents = [Session(self.path / p) for p in self.version["parents"]]
            self._inherit()
        self.names = own | set(self._route)
        if SESSION_FILE not in self.names:
            raise ValueError(f"{self.path} is not a solarian session (no {SESSION_FILE})")
        self._meta: dict | None = None
        self._by_episode: dict[int, list[str]] | None = None

    def _inherit(self) -> None:
        if SESSION_FILE not in self._own:
            self._route[SESSION_FILE] = (self.parents[0], SESSION_FILE)
        for index, entry in enumerate(self.version["episodes"]):
            if entry.get("src") is None:
                continue
            position, source = entry["src"]
            parent = self.parents[position]
            prefix = episode_dir(index)
            for rel in parent.episode_members(source, with_meta=False):
                member = f"{prefix}/{rel}"
                if member not in self._own:
                    self._route[member] = (parent, f"{episode_dir(source)}/{rel}")

    def __repr__(self) -> str:
        return f"Session({str(self.path)!r})"

    # ── reading ─────────────────────────────────────────────────────

    def exists(self, member: str) -> bool:
        return member in self.names

    @contextlib.contextmanager
    def open(self, member: str) -> Iterator[IO[bytes]]:
        """A binary file for ``member``, wherever in the layers it is stored."""
        if member in self._route:
            parent, source = self._route[member]
            with parent.open(source) as fh:
                yield fh
            return
        if member not in self._own:
            raise FileNotFoundError(f"{member} is not in {self.path}")
        if not self.is_zip:
            with open(self.path / member, "rb") as fh:
                yield fh
            return
        with zipfile.ZipFile(self.path) as zf, zf.open(member) as fh:
            yield fh

    def locate(self, member: str) -> Path | None:
        """The file ``member`` is stored in, or None when it sits inside a zip."""
        if member in self._route:
            parent, source = self._route[member]
            return parent.locate(source)
        if member not in self._own:
            raise FileNotFoundError(f"{member} is not in {self.path}")
        return None if self.is_zip else self.path / member

    def read_bytes(self, member: str) -> bytes:
        with self.open(member) as fh:
            return fh.read()

    def read_json(self, member: str):
        return json.loads(self.read_bytes(member))

    @property
    def meta(self) -> dict:
        """``session.json``."""
        if self._meta is None:
            self._meta = self.read_json(SESSION_FILE)
        return self._meta

    @property
    def cameras(self) -> list[str]:
        return list(self.meta.get("cameras", {}))

    def episodes(self) -> list[int]:
        """Indices of the sealed ``ok`` episodes, the ones training reads."""
        indices = sorted(int(m.group(1)) for m in map(_EPISODE_JSON.match, self.names) if m)
        return [i for i in indices if self.episode_meta(i).get("status") == "ok"]

    def episode_meta(self, episode: int) -> dict:
        return self.read_json(f"{episode_dir(episode)}/{EPISODE_FILE}")

    def episode_members(self, episode: int, with_meta: bool = False) -> list[str]:
        """Members of ``episode`` relative to its directory, e.g. ``video/camera0.rgb.mp4``."""
        if self._by_episode is None:
            by_episode: dict[int, list[str]] = {}
            for name in self.names:
                index = _episode_of(name)
                if index is not None:
                    by_episode.setdefault(index, []).append(name.split("/", 2)[2])
            self._by_episode = {k: sorted(v) for k, v in by_episode.items()}
        members = self._by_episode.get(episode, [])
        return members if with_meta else [m for m in members if m != EPISODE_FILE]

    def read_table(self, episode: int):
        import pyarrow as pa
        import pyarrow.parquet as pq

        return pq.read_table(pa.BufferReader(self.read_bytes(f"{episode_dir(episode)}/{TABLE_FILE}")))

    def read_targets(self, episode: int, camera: str) -> dict[str, np.ndarray] | None:
        """The arrays of a targets npz (``mask`` stays bit-packed), or None if absent."""
        member = targets_member(episode, camera)
        if member not in self.names:
            return None
        with np.load(io.BytesIO(self.read_bytes(member))) as npz:
            return {key: npz[key] for key in npz.files}

    def lineage(self) -> list[dict]:
        """This version's ``version.json`` and its ancestors', nearest first."""
        out: list[dict] = []
        seen: set[Path] = set()
        stack: list[Session] = [self]
        while stack:
            session = stack.pop(0)
            if session.path in seen or session.version is None:
                continue
            seen.add(session.path)
            out.append({"path": str(session.path), **session.version})
            stack.extend(session.parents)
        return out


# ── writing ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EpisodeRef:
    """Episode ``index`` of ``session``: what a new episode starts from."""

    session: Session
    index: int


class SessionWriter:
    """A new session at ``path``, built under a hidden name and moved into place by ``close``.

    With ``parents`` it is a version: a directory whose episodes may start from
    a parent's, storing only the members written here. Without, every member
    is stored, in a directory or, when ``path`` ends in ``.zip``, a zip.
    ``close`` writes ``manifest.jsonl`` (and ``version.json``) from the
    episodes added, so an interrupted run never leaves a partial session.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        parents: Sequence[Session] = (),
        force: bool = False,
        op: Mapping | None = None,
    ):
        self.path = Path(path).expanduser().resolve()
        if self.path.exists() and not force:
            raise FileExistsError(f"{self.path} exists (pass --force to overwrite)")
        self.parents = list(parents)
        if any(p.path == self.path for p in self.parents):
            raise ValueError(f"{self.path} is one of its own inputs")
        self.is_zip = self.path.name.endswith(".zip")
        if self.is_zip and self.parents:
            raise ValueError("A version is a directory; materialize it to get a zip")
        self.op = dict(op or {})
        self.tmp = self.path.with_name(f".{self.path.name}.tmp")
        _remove(self.tmp)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.is_zip:
            self._zip: zipfile.ZipFile | None = zipfile.ZipFile(self.tmp, "w", allowZip64=True)
            self.scratch = self.path.with_name(f".{self.path.name}.work")
            _remove(self.scratch)
            self.scratch.mkdir()
        else:
            self._zip = None
            self.tmp.mkdir()
            self.scratch = self.tmp / ".work"
            self.scratch.mkdir()
        self._episodes: list[dict] = []
        self.metas: list[dict] = []
        self._bytes: dict[int, int] = {}
        self._wrote_session = False

    def __enter__(self) -> SessionWriter:
        return self

    def __exit__(self, exc_type, *exc) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    @contextlib.contextmanager
    def _open(self, member: str) -> Iterator[IO[bytes]]:
        if self._zip is None:
            path = self.tmp / member
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as fh:
                yield fh
            return
        info = zipfile.ZipInfo(member, date_time=time.localtime()[:6])
        info.compress_type = _compression(member)
        with self._zip.open(info, "w", force_zip64=True) as fh:
            yield fh

    def _count(self, member: str, size: int) -> None:
        index = _episode_of(member)
        if index is not None:
            self._bytes[index] = self._bytes.get(index, 0) + size

    def write(self, member: str, data: bytes) -> None:
        with self._open(member) as fh:
            fh.write(data)
        self._count(member, len(data))
        if member == SESSION_FILE:
            self._wrote_session = True

    def write_json(self, member: str, obj) -> None:
        self.write(member, json.dumps(obj, indent=2).encode())

    def copy(self, session: Session, src_member: str, member: str) -> None:
        size = 0
        with session.open(src_member) as fin, self._open(member) as fout:
            while chunk := fin.read(_CHUNK):
                fout.write(chunk)
                size += len(chunk)
        self._count(member, size)

    def move(self, member: str, path: Path) -> None:
        """Add the file at ``path`` (scratch space, typically) as ``member``, consuming it."""
        size = os.path.getsize(path)
        if self._zip is None:
            target = self.tmp / member
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(path, target)
        else:
            with open(path, "rb") as fin, self._open(member) as fout:
                shutil.copyfileobj(fin, fout, _CHUNK)
            os.unlink(path)
        self._count(member, size)

    @property
    def next_index(self) -> int:
        return len(self._episodes)

    def add_episode(
        self,
        meta: Mapping,
        files: Mapping[str, bytes | Path] | None = None,
        source: EpisodeRef | None = None,
    ) -> int:
        """Append an episode: ``files`` (relative to its directory) over ``source``'s members.

        In a version the members of ``source`` not in ``files`` are read
        through from its session; otherwise they are copied. Returns the index.
        """
        index = len(self._episodes)
        prefix = episode_dir(index)
        files = dict(files or {})
        for rel, data in files.items():
            if isinstance(data, (bytes, bytearray)):
                self.write(f"{prefix}/{rel}", bytes(data))
            else:
                self.move(f"{prefix}/{rel}", Path(data))
        entry: dict = {"src": None}
        if source is not None:
            position = next((i for i, p in enumerate(self.parents) if p is source.session), None)
            if position is not None:
                entry["src"] = [position, source.index]
            else:
                for rel in source.session.episode_members(source.index):
                    if rel not in files:
                        self.copy(source.session, f"{episode_dir(source.index)}/{rel}", f"{prefix}/{rel}")
        meta = {**meta, "index": index}
        self.write_json(f"{prefix}/{EPISODE_FILE}", meta)
        self._episodes.append(entry)
        self.metas.append(meta)
        return index

    def abort(self) -> None:
        if self._zip is not None:
            self._zip.close()
        _remove(self.tmp)
        _remove(self.scratch)

    def close(self) -> None:
        if not self._wrote_session:
            if not self.parents:
                raise ValueError("A session without parents needs its own session.json")
        lines = []
        for meta in self.metas:
            lines.append(json.dumps({
                "episode": meta["index"],
                "sealed_at": time.time(),
                "status": meta.get("status", "ok"),
                "bytes": self._bytes.get(meta["index"], 0),
                "n_rows": meta["n_rows"],
                "duration": round(meta["end_ts"] - meta["start_ts"], 3),
                "transfer": {"state": "pending"},
                "upload": {"state": "pending"},
            }, separators=(",", ":")))
        self.write(MANIFEST_FILE, "".join(line + "\n" for line in lines).encode())
        if self.parents:
            self.write_json(VERSION_FILE, {
                "format": VERSION_FORMAT,
                "parents": [os.path.relpath(p.path, self.path) for p in self.parents],
                "op": self.op,
                "created_at": time.time(),
                "episodes": self._episodes,
            })
        if self._zip is not None:
            self._zip.close()
        _remove(self.scratch)
        _remove(self.path)
        os.replace(self.tmp, self.path)


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


LINEAGE_FILE = "lineage.json"


def materialize(session: Session, output: str | os.PathLike, force: bool = False) -> Path:
    """``session`` with every layer flattened into one standalone session at ``output``.

    Its ``ok`` episodes are renumbered from 0. ``lineage.json`` records the
    versions it was derived through.
    """
    with SessionWriter(output, force=force) as writer:
        writer.write_json(SESSION_FILE, session.meta)
        lineage = session.lineage()
        if lineage:
            writer.write_json(LINEAGE_FILE, lineage)
        for index in session.episodes():
            writer.add_episode(session.episode_meta(index), source=EpisodeRef(session, index))
    return writer.path


class _Sink:
    """A write-only stream: zipfile writes data descriptors when it cannot seek."""

    def __init__(self):
        self.chunks: list[bytes] = []

    def write(self, data) -> int:
        self.chunks.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        pass

    def drain(self) -> Iterator[bytes]:
        chunks, self.chunks = self.chunks, []
        yield from chunks


def iter_zip(session: Session) -> Iterator[bytes]:
    """What :func:`materialize` writes, as a ``.solarian.zip`` in chunks, for streaming a download."""
    sink = _Sink()

    def put(member: str, data: bytes) -> None:
        sink_zip.writestr(member, data, compress_type=_compression(member))

    with zipfile.ZipFile(sink, "w", allowZip64=True) as sink_zip:
        put(SESSION_FILE, json.dumps(session.meta, indent=2).encode())
        lineage = session.lineage()
        if lineage:
            put(LINEAGE_FILE, json.dumps(lineage, indent=2).encode())
        manifest = []
        for index, source in enumerate(session.episodes()):
            for rel in session.episode_members(source):
                info = zipfile.ZipInfo(f"{episode_dir(index)}/{rel}", date_time=time.localtime()[:6])
                info.compress_type = _compression(rel)
                with session.open(f"{episode_dir(source)}/{rel}") as fin, \
                        sink_zip.open(info, "w", force_zip64=True) as fout:
                    while chunk := fin.read(_CHUNK):
                        fout.write(chunk)
                        yield from sink.drain()
            meta = {**session.episode_meta(source), "index": index}
            put(f"{episode_dir(index)}/{EPISODE_FILE}", json.dumps(meta, indent=2).encode())
            manifest.append(json.dumps({
                "episode": index, "sealed_at": time.time(), "status": meta.get("status", "ok"),
                "n_rows": meta["n_rows"], "duration": round(meta["end_ts"] - meta["start_ts"], 3),
                "transfer": {"state": "pending"}, "upload": {"state": "pending"},
            }, separators=(",", ":")))
            yield from sink.drain()
        put(MANIFEST_FILE, "".join(line + "\n" for line in manifest).encode())
    yield from sink.drain()


# ── per-episode helpers ────────────────────────────────────────────


def targets_bytes(**arrays) -> bytes:
    """A targets npz, as solarian_umi's ``write_targets`` writes it."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def table_bytes(table) -> bytes:
    import pyarrow.parquet as pq

    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    return buffer.getvalue()


def column_stats(table, previous: Mapping) -> dict:
    """episode.json's running sums, recomputed for the columns ``previous`` has."""
    stats = {}
    for name in previous:
        if name not in table.column_names:
            continue
        values = table.column(name).to_numpy().astype(np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            stats[name] = {"min": float(finite.min()), "max": float(finite.max()),
                           "sum": float(finite.sum()), "sumsq": float(np.square(finite).sum()),
                           "count": int(finite.size)}
    return stats
