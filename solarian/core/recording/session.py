"""On-disk layout of a recording session, and the three files that describe it.

::

    recordings/<side>/<number>_<hash>_<name>/
    ├── session.json      written once at session start, then immutable
    ├── manifest.jsonl    appended once per sealed episode
    └── episodes/000000/
        ├── episode.json
        ├── episode.parquet
        └── video/<module>.<field>.mp4

They are three files rather than one because they have three write lifetimes
and three readers. ``session.json`` is what makes an episode directory
self-describing — column dtypes and units, camera geometry, and the optical
remap and resize that export has to apply because the archive is stored raw.
``episode.json`` exists so ``dataset ls`` and the exporter can choose episodes
without opening a parquet file. ``manifest.jsonl`` is the durability ledger:
append-only, so a hard kill costs you a torn last line instead of the whole
record of what has been uploaded.

A session directory is named ``<number>_<hash>_<name>``: the number orders
sessions within one recordings root, and the hash keeps two roots that reached
the same number apart once their sessions meet, on upload for one.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from solarian.core.recording.schema import RecordingSchema

SESSION_FILE = "session.json"
MANIFEST_FILE = "manifest.jsonl"
EPISODE_FILE = "episode.json"
TABLE_FILE = "episode.parquet"
EPISODES_DIR = "episodes"
VIDEO_DIR = "video"

#: ``0042_7f3a9c_cup_stacking`` -> number, hash, name.
_SESSION_DIR = re.compile(r"^(\d+)_([0-9a-f]{6})_(.+)$")


def session_name(name: str) -> str:
    """*name* reduced to what is safe in a directory name on every platform."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "session"


def parse_session_dir(dir_name: str) -> tuple[int, str, str] | None:
    """``(number, hash, name)`` of a session directory, or ``None`` for an older one."""
    match = _SESSION_DIR.match(dir_name)
    if match is None:
        return None
    return int(match.group(1)), match.group(2), match.group(3)


@dataclass(frozen=True)
class SessionPaths:
    root: Path

    @staticmethod
    def create(root: str | os.PathLike) -> SessionPaths:
        paths = SessionPaths(Path(root))
        (paths.root / EPISODES_DIR).mkdir(parents=True, exist_ok=True)
        return paths

    @staticmethod
    def allocate(root: str | os.PathLike, name: str) -> SessionPaths:
        """A fresh ``<number>_<hash>_<name>`` directory under *root*.

        The directory is the claim: created exclusively, so two recorders
        sharing a root that pick the same number both see it and one moves on.
        """
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        name = session_name(name)
        while True:
            number = SessionPaths.next_session_number(root)
            candidate = root / f"{number:04d}_{secrets.token_hex(3)}_{name}"
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            return SessionPaths.create(candidate)

    @staticmethod
    def next_session_number(root: Path) -> int:
        """One past the highest numbered session under *root*."""
        if not root.is_dir():
            return 0
        numbers = [
            parsed[0]
            for p in root.iterdir()
            if p.is_dir() and (parsed := parse_session_dir(p.name)) is not None
        ]
        return max(numbers) + 1 if numbers else 0

    @property
    def session_json(self) -> Path:
        return self.root / SESSION_FILE

    @property
    def manifest(self) -> Path:
        return self.root / MANIFEST_FILE

    @property
    def episodes(self) -> Path:
        return self.root / EPISODES_DIR

    def episode_dir(self, index: int) -> Path:
        return self.episodes / f"{index:06d}"

    def video_path(self, index: int, camera_name: str) -> Path:
        return self.episode_dir(index) / VIDEO_DIR / f"{camera_name}.mp4"

    def next_episode_index(self) -> int:
        """One past the highest directory present, so a resumed session appends."""
        if not self.episodes.is_dir():
            return 0
        indices = [
            int(p.name)
            for p in self.episodes.iterdir()
            if p.is_dir() and p.name.isdigit()
        ]
        return max(indices) + 1 if indices else 0


@dataclass
class CameraMeta:
    """Everything export needs to reconstruct a policy-resolution frame."""

    width: int
    height: int
    color_mode: str = "rgb"
    source_fps: float = 30.0
    sensor_mode: str | None = None
    #: The archive is stored before ``cv2.remap``, so the calibration stays a
    #: separate artifact and a later recalibration re-derives past datasets.
    optical_remap: str | None = None
    #: What the robot published as the observation: the capture, downscaled.
    observation_resize: tuple[int, int] | None = None
    #: GoPro plane the export should warp into. Independent of the two above:
    #: the observation is pre-remap and the archive is a third resolution.
    remap_size: tuple[int, int] | None = None


@dataclass
class SessionMeta:
    #: The robot's expanded structure from
    #: :func:`~solarian.core.recording.structure.build_structure` -- modules,
    #: components, IK joint mapping and attachment graph, not just a class name.
    #: ``None`` from the controller, which has no robot to introspect.
    robot: dict | None
    schema: RecordingSchema
    cameras: dict[str, CameraMeta]
    video: dict
    fps: float
    task: str | None = None
    #: Which side recorded: ``robot`` or ``controller``.
    source: str = "robot"
    #: ``<number>_<hash>``; with ``number`` and ``name``, what the directory
    #: name encodes, so no reader has to parse it. ``None`` for a directory
    #: adopted under a name that does not follow the scheme.
    id: str | None = None
    number: int | None = None
    name: str | None = None
    created_at: float = field(default_factory=time.time)

    def write(self, paths: SessionPaths) -> None:
        payload = {
            "solarian_version": _solarian_version(),
            "git_sha": _git_sha(),
            "id": self.id,
            "number": self.number,
            "name": self.name,
            "source": self.source,
            "robot": self.robot,
            "task": self.task,
            "created_at": self.created_at,
            "fps": self.fps,
            "cameras": {
                name: {k: v for k, v in asdict(cam).items() if v is not None}
                for name, cam in self.cameras.items()
            },
            "video": self.video,
            "schema": self.schema.to_json(),
            # Which fields each module tabulated, in the order the schema walked
            # them. Computed all along but never written, and it is what groups
            # columns by module without splitting names on dots.
            "obs_fields": dict(self.schema.obs_fields),
            "act_fields": dict(self.schema.act_fields),
        }
        _write_atomic(paths.session_json, json.dumps(payload, indent=2))

    @staticmethod
    def load(paths: SessionPaths) -> dict:
        return json.loads(paths.session_json.read_text())


@dataclass
class EpisodeMeta:
    """Written once, at seal time. Selection metadata, never bulk data."""

    index: int
    start_ts: float
    end_ts: float
    n_rows: int
    #: ``"ok"`` or ``"aborted"``. An explicit field so nobody encodes operator
    #: intent into the task string.
    status: str = "ok"
    task: str | None = None
    skill: str | None = None
    n_frames: dict[str, int] = field(default_factory=dict)
    video_coverage: dict[str, float] = field(default_factory=dict)
    dropped_frames: dict[str, int] = field(default_factory=dict)
    measured_hz: dict[str, float] = field(default_factory=dict)
    #: Running sums, not finals, so global normalisation aggregates in
    #: O(episodes) rather than re-reading every frame.
    stats: dict[str, dict] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts

    def write(self, paths: SessionPaths) -> None:
        _write_atomic(
            paths.episode_dir(self.index) / EPISODE_FILE,
            json.dumps(asdict(self), indent=2),
        )

    @staticmethod
    def load(path: Path) -> dict:
        return json.loads(path.read_text())


def append_manifest(paths: SessionPaths, entry: dict) -> None:
    """Append one ledger line and fsync it.

    The fsync is the whole point: an entry that survives only in the page cache
    tells a post-crash uploader that an episode is safe when it is not.
    """
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    with open(paths.manifest, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def read_manifest(paths: SessionPaths) -> list[dict]:
    """Every intact ledger line. A torn final line is discarded, not raised."""
    if not paths.manifest.exists():
        return []
    entries = []
    for line in paths.manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def directory_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _write_atomic(path: Path, text: str) -> None:
    """Rename into place so a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _solarian_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("solarian")
    except PackageNotFoundError:
        return "unknown"


def _git_sha() -> str | None:
    """The working tree's commit, when recording from a checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None
