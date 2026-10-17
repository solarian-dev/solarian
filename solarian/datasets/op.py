"""The dataset operation contract: one class per op, run by :func:`~solarian.datasets.runner.run_op`.

An op says what it does to an episode; the runner owns everything else:
opening the inputs, choosing what runs in parallel, writing each output as a
version of its inputs, recording provenance, and reporting progress. The same
op therefore runs from ``solarian dataset run``, from Solarian Base, or from a
script, and an agent can discover it, read its parameters as JSON Schema, and
run it without reading its code.

Kinds, by what one input episode becomes:

* ``MAP``: one episode, with some members replaced or added (tracking an object).
* ``FILTER``: the episode, or nothing (deleting episodes).
* ``EXPAND``: any number of episodes, into any of the op's named ``outputs``
  (splitting into pick and place, mirroring).
* ``COMBINE``: several input sessions into one (merging).

Plugins register ops under the ``solarian.dataset_ops`` entry-point group. A
project needs no registration: each ``<package>/dataset_ops/<op_name>.py`` of
the project being run in defines one op, which can import the project's code.
"""

from __future__ import annotations

import enum
import functools
import importlib
import importlib.metadata
import logging
import os
import subprocess
import sys
from abc import ABC
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Callable, ClassVar, Iterable, Sequence

from pydantic import BaseModel

from solarian.core.recording.archive import (
    EpisodeRef,
    Session,
    episode_dir,
    targets_member,
    video_member,
)

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "solarian.dataset_ops"


class OpKind(str, enum.Enum):
    MAP = "map"
    FILTER = "filter"
    EXPAND = "expand"
    COMBINE = "combine"


class Resources(BaseModel):
    """What a machine needs to run the op; Solarian Base schedules on it."""

    gpu: bool = False
    gpu_memory_gb: float | None = None


@dataclass
class Episode:
    """One input episode as an op sees it."""

    session: Session
    index: int

    @cached_property
    def meta(self) -> dict:
        return self.session.episode_meta(self.index)

    @property
    def n_rows(self) -> int:
        return int(self.meta["n_rows"])

    @property
    def ref(self) -> EpisodeRef:
        return EpisodeRef(self.session, self.index)

    def member(self, rel: str) -> str:
        return f"{episode_dir(self.index)}/{rel}"

    def members(self) -> list[str]:
        """Members relative to the episode directory, but ``episode.json``."""
        return self.session.episode_members(self.index)

    def has(self, rel: str) -> bool:
        return self.session.exists(self.member(rel))

    def read_bytes(self, rel: str) -> bytes:
        return self.session.read_bytes(self.member(rel))

    def read_json(self, rel: str):
        return self.session.read_json(self.member(rel))

    def table(self):
        return self.session.read_table(self.index)

    def video(self, camera: str) -> str:
        return video_member(self.index, camera)

    def targets(self, camera: str) -> dict | None:
        return self.session.read_targets(self.index, camera)

    def has_targets(self, camera: str) -> bool:
        return self.session.exists(targets_member(self.index, camera))

    def derive(self, files: dict[str, bytes | Path] | None = None, **meta) -> EpisodeOut:
        """This episode with ``files`` replaced or added and ``meta`` merged into episode.json."""
        return EpisodeOut(meta={**self.meta, **meta}, files=dict(files or {}), source=self)


@dataclass
class EpisodeOut:
    """An episode an op produces.

    ``files`` (paths relative to the episode directory) hold bytes or a file in
    :attr:`OpContext.scratch`, which the writer moves in. Every other member is
    ``source``'s, read through from the input rather than copied.
    """

    meta: dict
    files: dict[str, bytes | Path] = field(default_factory=dict)
    source: Episode | None = None


@dataclass
class OpContext:
    """What the runner gives an op beyond its parameters."""

    inputs: Sequence[Session]
    #: Per-run scratch directory on the outputs' filesystem; files left here are deleted.
    scratch: Path
    workers: int = 1
    log: Callable[[str], None] = logger.info
    #: Partial results of the episode in hand, for someone watching it run
    #: (``{"frame": k, ...}`` from ``track``); outside a review session they go nowhere.
    preview: Callable[[dict], None] = lambda update: None
    #: True once the episode in hand is no longer wanted, superseded by a newer
    #: request; a long op checks it and raises :class:`Cancelled`.
    cancelled: Callable[[], bool] = lambda: False

    @property
    def session(self) -> Session:
        """The input, for an op that takes one."""
        return self.inputs[0]


class Cancelled(Exception):
    """An op stopped work on an episode because :attr:`OpContext.cancelled` said so."""


class DatasetOp(ABC):
    """Subclass, set the class attributes, and implement the method of your ``kind``."""

    #: CLI and API name, e.g. ``delete-episodes``.
    name: ClassVar[str]
    title: ClassVar[str]
    #: What it does, for a person choosing an op or an agent planning a pipeline.
    description: ClassVar[str] = ""
    kind: ClassVar[OpKind]
    #: Parameters, validated before the run and published as JSON Schema.
    Params: ClassVar[type[BaseModel]]
    #: Output names. EXPAND ops yield episodes into these; every other kind has one.
    outputs: ClassVar[tuple[str, ...]] = ("out",)
    resources: ClassVar[Resources] = Resources()
    #: Episode members the op writes, with ``{camera}`` standing for each camera.
    produces: ClassVar[tuple[str, ...]] = ()
    #: The ``solarian.services`` entry whose model computes the same thing live, if any.
    service: ClassVar[str | None] = None
    #: Whether episodes may be processed concurrently on threads.
    parallel: ClassVar[bool] = True
    #: COMBINE ops take two or more inputs; every other kind takes one.
    min_inputs: ClassVar[int] = 1
    max_inputs: ClassVar[int | None] = 1
    #: Distribution the op ships in; set by :func:`discover_ops` for plugins.
    package: ClassVar[str] = "solarian"
    #: Where the op comes from: ``builtin``, ``plugin:<dist>`` or ``project:<file>``.
    source: ClassVar[str] = "builtin"
    #: The project's git commit, for a project op; ``-dirty`` when its ops have edits.
    commit: ClassVar[str | None] = None
    #: ``source`` of the op of the same name this one takes the place of.
    overrides: ClassVar[str | None] = None

    def setup(self, params: BaseModel, ctx: OpContext) -> None:
        """Validate the inputs against ``params`` and load models, once per run."""

    def session_meta(self, metas: list[dict], params: BaseModel) -> dict:
        """``session.json`` of the outputs, from the inputs'."""
        return metas[0]

    def map(self, episode: Episode, params: BaseModel, ctx: OpContext) -> EpisodeOut | None:
        """MAP: the episode transformed, or None to keep it unchanged."""
        raise NotImplementedError

    def keep(self, episode: Episode, params: BaseModel) -> bool:
        """FILTER: whether the episode stays."""
        raise NotImplementedError

    def expand(self, episode: Episode, params: BaseModel, ctx: OpContext) -> Iterable[tuple[str, EpisodeOut]]:
        """EXPAND: ``(output name, episode)`` pairs, in output order."""
        raise NotImplementedError

    def combine(self, params: BaseModel, ctx: OpContext) -> Iterable[EpisodeOut]:
        """COMBINE: the output's episodes, drawn from any of ``ctx.inputs``."""
        raise NotImplementedError

    @classmethod
    def describe(cls) -> dict:
        """The catalog entry: everything a GUI form or an agent needs to run the op."""
        return {
            "name": cls.name,
            "title": cls.title,
            "description": " ".join(cls.description.split()),
            "kind": cls.kind.value,
            "outputs": list(cls.outputs),
            "inputs": {"min": cls.min_inputs, "max": cls.max_inputs},
            "resources": cls.resources.model_dump(),
            "produces": list(cls.produces),
            "service": cls.service,
            "params": cls.Params.model_json_schema(),
            "package": cls.package,
            "source": cls.source,
            "commit": cls.commit,
            "overrides": cls.overrides,
            "online": cls.kind in (OpKind.MAP, OpKind.FILTER),
        }


#: Directory of a project's package that holds its ops, one per file.
PROJECT_OPS_DIR = "dataset_ops"


@dataclass
class Catalog:
    ops: dict[str, type[DatasetOp]]
    #: ``(where, message)`` for each plugin or project op file that did not load.
    errors: list[tuple[str, str]] = field(default_factory=list)
    project: Path | None = None


def find_project(start: Path | None = None) -> Path | None:
    """The project checkout ops are run in: ``$SOLARIAN_PROJECT_DIR``, else the
    nearest directory at or above ``start`` (the cwd) with a pyproject.toml and a
    settings.py the CLI would find there; None outside a project."""
    from solarian.cli.manage import _find_settings_path

    if os.environ.get("SOLARIAN_PROJECT_DIR"):
        return Path(os.environ["SOLARIAN_PROJECT_DIR"]).expanduser().resolve()
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        if not (directory / "pyproject.toml").is_file():
            continue
        try:
            _find_settings_path(directory)
        except FileNotFoundError:
            continue
        return directory
    return None


@functools.lru_cache(maxsize=None)
def _project_commit(project: Path, ops_dir: Path) -> str | None:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(project), *args], capture_output=True, text=True,
                              check=True).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--", str(ops_dir), ":(exclude)**/__pycache__/**")
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if dirty else commit


def _project_ops(project: Path, errors: list[tuple[str, str]]) -> dict[str, type[DatasetOp]]:
    from solarian.cli.manage import _find_settings_path

    package = Path(_find_settings_path(project)).parent
    ops_dir = package / PROJECT_OPS_DIR
    if not (package / "__init__.py").is_file() or not ops_dir.is_dir():
        return {}
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    ops: dict[str, type[DatasetOp]] = {}
    for file in sorted(ops_dir.glob("*.py")):
        if file.name.startswith("_"):
            continue
        where = str(file.relative_to(project))
        try:
            module = importlib.import_module(f"{package.name}.{PROJECT_OPS_DIR}.{file.stem}")
        except Exception as exc:  # noqa: BLE001 - a broken op file must not hide the others
            errors.append((where, f"{type(exc).__name__}: {exc}"))
            continue
        found = [value for value in vars(module).values()
                 if isinstance(value, type) and issubclass(value, DatasetOp) and value.__module__ == module.__name__]
        if len(found) != 1:
            errors.append((where, f"defines {len(found)} DatasetOp subclasses; an op file defines one"))
            continue
        [op] = found
        if "name" not in vars(op):
            op.name = file.stem.replace("_", "-")
        if "title" not in vars(op):
            op.title = op.name.replace("-", " ").capitalize()
        op.package = package.name
        op.source = f"project:{where}"
        op.commit = _project_commit(project, ops_dir)
        ops[op.name] = op
    return ops


def discover(project: Path | None = None) -> Catalog:
    """Built-in ops, then installed plugins', then the project's (``find_project()``
    by default), each taking the place of an earlier op of the same name."""
    from solarian.datasets.ops import BUILTIN_OPS

    catalog = Catalog(ops={op.name: op for op in BUILTIN_OPS}, project=project or find_project())

    def add(name: str, op: type[DatasetOp]) -> None:
        shadowed = catalog.ops.get(name)
        op.overrides = shadowed.source if shadowed is not None and shadowed is not op else None
        catalog.ops[name] = op

    for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        try:
            op = ep.load()
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not hide the others
            logger.exception("Failed to load dataset op entry point %r", ep.name)
            catalog.errors.append((f"entry point {ep.name} ({ep.value})", f"{type(exc).__name__}: {exc}"))
            continue
        op.package = ep.dist.name if ep.dist is not None else ep.value.split(":")[0]
        op.source = f"plugin:{op.package}"
        add(ep.name, op)
    if catalog.project is not None:
        for name, op in _project_ops(catalog.project, catalog.errors).items():
            add(name, op)
    return catalog


def discover_ops() -> dict[str, type[DatasetOp]]:
    """``{name: op class}`` of :func:`discover`."""
    return discover().ops
