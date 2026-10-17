"""Run a dataset op over sessions and write its outputs as versions of them."""

from __future__ import annotations

import concurrent.futures
import importlib.metadata
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

from solarian.core.recording.archive import SESSION_FILE, Session, SessionWriter
from solarian.datasets.op import DatasetOp, Episode, EpisodeOut, OpContext, OpKind, discover_ops

#: ``progress(done, total, message)``.
Progress = Callable[[int, int, str], None]


@dataclass
class OutputSummary:
    name: str
    path: Path
    episodes: int
    frames: int

    def as_dict(self) -> dict:
        return {"name": self.name, "path": str(self.path), "episodes": self.episodes, "frames": self.frames}


def get_op(name: str) -> type[DatasetOp]:
    ops = discover_ops()
    if name not in ops:
        raise KeyError(f"Unknown dataset op {name!r}; installed: {', '.join(sorted(ops)) or '(none)'}")
    return ops[name]


def validate_params(op: type[DatasetOp], params: Mapping | None):
    """``params`` as the op's ``Params``; an unknown name is an error, not a default."""
    params = dict(params or {})
    unknown = sorted(set(params) - set(op.Params.model_fields))
    if unknown:
        raise ValueError(f"{op.name} has no parameter {', '.join(unknown)}; it takes {sorted(op.Params.model_fields)}")
    return op.Params.model_validate(params)


def op_record(op: type[DatasetOp], params, inputs: Sequence[Session]) -> dict:
    """What a version's ``version.json`` says made it."""
    return {
        "name": op.name,
        "kind": op.kind.value,
        "package": op.package,
        "package_version": _version(op.package),
        "source": op.source,
        "commit": op.commit,
        "solarian_version": _version("solarian"),
        "params": params.model_dump(mode="json"),
        "inputs": [str(s.path) for s in inputs],
        "host": platform.node(),
        "started_at": time.time(),
    }


def default_outputs(op: type[DatasetOp], first_input: Path) -> dict[str, Path]:
    """``<input>-<op>`` beside the input, plus ``-<output>`` for an op with several."""
    stem = first_input.name
    for suffix in (".solarian.zip", ".zip"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    base = first_input.with_name(f"{stem}-{op.name}")
    if len(op.outputs) == 1:
        return {op.outputs[0]: base}
    return {name: base.with_name(f"{base.name}-{name}") for name in op.outputs}


def run_op(
    name: str,
    inputs: Sequence[str | os.PathLike],
    outputs: Mapping[str, str | os.PathLike] | None = None,
    params: Mapping | None = None,
    *,
    workers: int = 4,
    flat: bool = False,
    force: bool = False,
    progress: Progress | None = None,
    log: Callable[[str], None] | None = None,
) -> list[OutputSummary]:
    """Run op ``name`` on ``inputs``; returns one summary per output.

    Each output is a version of the inputs (only what the op wrote is stored)
    unless ``flat``, or its path ends in ``.zip``, which stores every member.
    """
    op_cls = get_op(name)
    op = op_cls()
    validated = validate_params(op_cls, params)
    sessions = [Session(p) for p in inputs]
    if len(sessions) < op_cls.min_inputs or (op_cls.max_inputs is not None and len(sessions) > op_cls.max_inputs):
        bound = op_cls.max_inputs if op_cls.max_inputs == op_cls.min_inputs else f"{op_cls.min_inputs}+"
        raise ValueError(f"{name} takes {bound} input(s), got {len(sessions)}")
    targets = {k: Path(v) for k, v in (outputs or default_outputs(op_cls, sessions[0].path)).items()}
    if set(targets) != set(op_cls.outputs):
        raise ValueError(f"{name} writes outputs {list(op_cls.outputs)}, got {sorted(targets)}")

    record = op_record(op_cls, validated, sessions)
    writers: dict[str, SessionWriter] = {}
    try:
        for out_name in op_cls.outputs:
            path = targets[out_name]
            layered = not flat and not path.name.endswith(".zip")
            writers[out_name] = SessionWriter(
                path, parents=sessions if layered else (), force=force,
                op={**record, "output": out_name},
            )
        scratch = next(iter(writers.values())).scratch
        ctx = OpContext(inputs=sessions, scratch=scratch, workers=max(1, workers),
                        log=log or (lambda message: None))
        op.setup(validated, ctx)
        meta = op.session_meta([s.meta for s in sessions], validated)
        for writer in writers.values():
            if meta != sessions[0].meta or not writer.parents:
                writer.write_json(SESSION_FILE, meta)

        for out_name, episode in _produce(op, validated, ctx, progress):
            writer = writers[out_name]
            writer.add_episode(
                episode.meta, episode.files,
                source=episode.source.ref if episode.source is not None else None,
            )
        for writer in writers.values():
            writer.op["finished_at"] = time.time()
            writer.close()
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise
    return [
        OutputSummary(name=out_name, path=writer.path, episodes=len(writer.metas),
                      frames=sum(int(m["n_rows"]) for m in writer.metas))
        for out_name, writer in writers.items()
    ]


def _produce(op: DatasetOp, params, ctx: OpContext, progress: Progress | None) -> Iterator[tuple[str, EpisodeOut]]:
    """``(output, episode)`` in output order, episodes processed ``ctx.workers`` at a time."""
    kind = op.kind
    out = op.outputs[0]
    if kind is OpKind.COMBINE:
        total = sum(len(s.episodes()) for s in ctx.inputs)
        for done, episode in enumerate(op.combine(params, ctx), start=1):
            yield out, episode
            _report(progress, done, total, "")
        return

    episodes = [Episode(ctx.session, i) for i in ctx.session.episodes()]
    total = len(episodes)
    _report(progress, 0, total, "")
    if kind is OpKind.FILTER:
        for done, episode in enumerate(episodes, start=1):
            if op.keep(episode, params):
                yield out, episode.derive()
            _report(progress, done, total, f"episode {episode.index}")
        return

    if kind is OpKind.MAP:
        def work(episode: Episode) -> list[tuple[str, EpisodeOut]]:
            result = op.map(episode, params, ctx)
            return [(out, result if result is not None else episode.derive())]
    else:
        def work(episode: Episode) -> list[tuple[str, EpisodeOut]]:
            return list(op.expand(episode, params, ctx))

    if op.parallel and ctx.workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=ctx.workers) as pool:
            results = pool.map(work, episodes)
            for done, (episode, produced) in enumerate(zip(episodes, results), start=1):
                yield from produced
                _report(progress, done, total, f"episode {episode.index}")
        return
    for done, episode in enumerate(episodes, start=1):
        yield from work(episode)
        _report(progress, done, total, f"episode {episode.index}")


def _report(progress: Progress | None, done: int, total: int, message: str) -> None:
    if progress is not None:
        progress(done, total, message)


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
