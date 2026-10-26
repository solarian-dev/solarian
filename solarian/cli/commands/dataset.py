"""``solarian dataset``: run dataset ops, and build computed datasets out of them."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

import yaml
from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry

ACTIONS = ("ops", "describe", "info", "run", "materialize", "new", "apply", "log", "close", "drop",
           "export", "rebuild", "session", "redo", "accept")


def _set_dotted(target: dict, key: str, value) -> None:
    *path, last = key.split(".")
    for part in path:
        target = target.setdefault(part, {})
    target[last] = value


def _parse_params(pairs: list[str], params_json: str | None) -> dict:
    """``--params-json`` (inline or ``@file``), then each ``--param key=value`` over it."""
    params: dict = {}
    if params_json:
        text = Path(params_json[1:]).read_text() if params_json.startswith("@") else params_json
        params = json.loads(text)
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise SystemExit(f"--param wants key=value, got {pair!r}")
        # YAML so that 3, 0.5, true, [1, 2] and {x: 1} mean what they look like.
        _set_dotted(params, key, yaml.safe_load(raw))
    return params


def _parse_outputs(values: list[str], names: tuple[str, ...]) -> dict[str, Path] | None:
    if not values:
        return None
    outputs = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep:
            if len(names) != 1 or len(values) != 1:
                raise SystemExit(f"This op writes {list(names)}: pass -o NAME=PATH for each")
            name, path = names[0], value
        outputs[name] = Path(path)
    return outputs


def _emit(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}), flush=True)


def _reporters(as_json: bool) -> tuple[Callable[[int, int, str], None], Callable[[str], None], Callable[[], None]]:
    """``(progress, log, done)``: JSON lines for a job runner, else a progress bar."""
    if as_json:
        def progress(done: int, total: int, message: str) -> None:
            _emit("progress", done=done, total=total, message=message)

        return progress, lambda message: _emit("log", message=message), lambda: None
    from tqdm import tqdm

    bar = tqdm(unit="ep")

    def progress(done: int, total: int, message: str) -> None:
        bar.total = total
        bar.n = done
        bar.set_postfix_str(message, refresh=False)
        bar.refresh()

    return progress, bar.write, bar.close


def _computed(ref: str):
    """``(ComputedDataset, key)`` for ``dir[@version]``; key None is the source."""
    from solarian.datasets.computed import ComputedDataset, split_ref

    path, key = split_ref(ref)
    computed = ComputedDataset(path)
    return computed, computed.resolve(key)


def _tree(computed) -> list[str]:
    """``log``'s lines: the source, then each version under its parent."""
    from solarian.datasets.computed import ACCEPTED

    versions = computed.versions()
    lines = [f"{computed.source.name}  (source)"]

    def walk(parent: str | None, indent: str) -> None:
        children = [k for k, v in versions.items() if v["parent"] == parent]
        for n, key in enumerate(children):
            entry = versions[key]
            last = n == len(children) - 1
            summary = entry.get("summary", {})
            review = ""
            if entry["episodes"]:
                accepted = sum(e["status"] == ACCEPTED for e in entry["episodes"].values())
                review = f", {accepted}/{len(entry['episodes'])} reviewed"
            lines.append(f"{indent}{'└─' if last else '├─'} {key}  {entry['state']}, "
                         f"{summary.get('episodes', '?')} episodes{review}  {json.dumps(entry['params'])}")
            walk(key, indent + ("   " if last else "│  "))

    walk(None, "")
    return lines


@CommandRegistry.register("dataset")
class DatasetCommand(BaseCommand):
    category = "Training & Data"
    uses_config = False
    help = "Curate and enrich recorded sessions: delete, split, track, mirror, merge"
    description = """
    Every op reads one or more sessions and writes new ones, never modifying its
    inputs. An output is a version of its inputs: only what the op changed is
    stored, the rest is read through from them, so deleting episodes or
    tracking an object costs megabytes however large the videos are. An output
    path ending in .zip (or --flat) stores everything instead; `materialize`
    flattens a version into a standalone session, e.g. a .solarian.zip to train on.

    A computed dataset groups the versions built from one source session: `new`
    starts one, `apply` adds a version on top of another (a tree: split forks
    it), `log` shows the tree, `export` flattens a version into a standalone
    session and `drop` deletes a version with everything built on it. A map
    or filter version applied with --review or --online stays open, to redo
    and accept its episodes one at a time (`redo`, `accept`, or `session` for a
    program), until `close`. computed.json records every op and redo, so
    `rebuild` reproduces the dataset from its source.

    Ops come from solarian, from installed plugins (entry-point group
    solarian.dataset_ops) and from the project's <package>/dataset_ops/*.py.
    `describe <op>` prints an op's parameters as JSON Schema, which is also
    what Solarian Base renders its forms from.
    """
    usage = (
        "[--project DIR] ops [--json] | describe <op> | info <session> [--json] | "
        "run <op> <session>... [-o [NAME=]PATH]... [--param KEY=VALUE]... [--params-json JSON|@FILE] "
        "[--flat] [-j N] [--force] [--progress-json] | materialize <session> <output> [--force] | "
        "new <source> <dir> | apply <dir>[@VERSION] <op> [--with REF]... [--param KEY=VALUE]... "
        "[--review | --online] [--progress-json] | log <dir> [--json] | close <dir>@VERSION | "
        "drop <dir>@VERSION | export <dir>[@VERSION] <output> | rebuild <dir> <into> | "
        "session <dir>[@VERSION] | redo <dir>[@VERSION] <episode> [--param KEY=VALUE]... [--keep | --drop] | "
        "accept <dir>[@VERSION] <episode>"
    )
    options = {
        "ops": "List the installed ops.",
        "describe <op>": "The op's catalog entry, parameters as JSON Schema.",
        "info <session>": "Episodes, frames, cameras, targets and lineage of a session or version.",
        "run <op> <session>...": "Run an op. Outputs default to <session>-<op>[-<output>] beside the input.",
        "-o [NAME=]PATH": "Where to write an output; NAME= is needed when the op writes several.",
        "--param KEY=VALUE": "One parameter, VALUE read as YAML; dotted keys reach nested ones.",
        "--params-json JSON|@FILE": "All parameters as one JSON object, inline or from a file.",
        "--flat": "Store every member in the outputs instead of reading through to the inputs.",
        "-j N": "Episodes processed in parallel, for ops that allow it (default 4).",
        "--progress-json": "Print progress and the result as JSON lines on stdout, for a job runner.",
        "materialize <session> <output>": "Flatten a version; an output ending in .zip is a .solarian.zip.",
        "materialize <session> -": "Stream the flattened .solarian.zip to stdout, for a server to send on.",
        "--project DIR": "Project whose <package>/dataset_ops/ load (default: the one around the cwd).",
        "new <source> <dir>": "Start a computed dataset over a standalone session.",
        "apply <dir>[@VERSION] <op>": "Add a version made by <op> on VERSION (default: the only leaf).",
        "VERSION": "A version's key (see log), or source for the computed dataset's source.",
        "--with REF": "A further input of a combine op: a version of this dataset or a path.",
        "--review": "Leave a map or filter version open, to review its episodes.",
        "--online": "As --review, and process no episode yet: redo them one at a time.",
        "log <dir>": "The version tree, with each version's state and review progress.",
        "close <dir>@VERSION": "End a version's review; ops can then build on it.",
        "drop <dir>@VERSION": "Delete a version and every version built on it.",
        "export <dir>[@VERSION] <output>": "Flatten a version; .zip is a .solarian.zip, - is stdout.",
        "rebuild <dir> <into>": "Replay every op and redo of <dir> from its source into <into>.",
        "session <dir>[@VERSION]": "Redo and review an open version over JSON lines on stdin/stdout.",
        "redo <dir>[@VERSION] <episode>": "Reprocess one episode (map), --param over the version's.",
        "--keep / --drop": "With redo on a filter version: keep or drop the parent's episode.",
        "accept <dir>[@VERSION] <episode>": "Mark an episode of an open version as right.",
    }
    examples = (
        "solarian dataset ops",
        "solarian dataset run delete-episodes data/cups --param 'episodes=[3, 7]'",
        "solarian dataset new data/cups data/cups-curated",
        "solarian dataset apply data/cups-curated split-pick-place",
        "solarian dataset apply data/cups-curated@02-split-pick-place-pick track --online",
        "solarian dataset redo data/cups 4 --param 'prompt={frame: -1, point: [730, 711]}'",
        "solarian dataset log data/cups-curated",
        "solarian dataset export data/cups-curated@04-mirror cups.solarian.zip",
    )

    def parse_args(self, argv: list[str]) -> list[str]:
        parser = argparse.ArgumentParser(prog="solarian dataset")
        parser.add_argument("--project", default=None)
        sub = parser.add_subparsers(dest="action", required=True)
        ops = sub.add_parser("ops")
        ops.add_argument("--json", action="store_true")
        describe = sub.add_parser("describe")
        describe.add_argument("op")
        info = sub.add_parser("info")
        info.add_argument("session")
        info.add_argument("--json", action="store_true")
        run = sub.add_parser("run")
        run.add_argument("op")
        run.add_argument("inputs", nargs="+")
        run.add_argument("-o", "--output", action="append", default=[])
        run.add_argument("--param", action="append", default=[])
        run.add_argument("--params-json", default=None)
        run.add_argument("--flat", action="store_true")
        run.add_argument("-j", "--jobs", type=int, default=4)
        run.add_argument("--force", action="store_true")
        run.add_argument("--progress-json", action="store_true")
        materialize = sub.add_parser("materialize")
        materialize.add_argument("session")
        materialize.add_argument("output")
        materialize.add_argument("--force", action="store_true")

        new = sub.add_parser("new")
        new.add_argument("source")
        new.add_argument("dir")
        apply = sub.add_parser("apply")
        apply.add_argument("ref")
        apply.add_argument("op")
        apply.add_argument("--with", dest="extra", action="append", default=[])
        apply.add_argument("--param", action="append", default=[])
        apply.add_argument("--params-json", default=None)
        mode = apply.add_mutually_exclusive_group()
        mode.add_argument("--review", action="store_true")
        mode.add_argument("--online", action="store_true")
        apply.add_argument("-j", "--jobs", type=int, default=4)
        apply.add_argument("--progress-json", action="store_true")
        log = sub.add_parser("log")
        log.add_argument("dir")
        log.add_argument("--json", action="store_true")
        for action in ("close", "drop", "session"):
            sub.add_parser(action).add_argument("ref")
        export = sub.add_parser("export")
        export.add_argument("ref")
        export.add_argument("output")
        export.add_argument("--force", action="store_true")
        rebuild = sub.add_parser("rebuild")
        rebuild.add_argument("dir")
        rebuild.add_argument("into")
        redo = sub.add_parser("redo")
        redo.add_argument("ref")
        redo.add_argument("episode", type=int)
        redo.add_argument("--param", action="append", default=[])
        redo.add_argument("--params-json", default=None)
        decision = redo.add_mutually_exclusive_group()
        decision.add_argument("--keep", dest="keep", action="store_const", const=True, default=None)
        decision.add_argument("--drop", dest="keep", action="store_const", const=False)
        accept = sub.add_parser("accept")
        accept.add_argument("ref")
        accept.add_argument("episode", type=int)
        self.ns = parser.parse_args(argv)
        return []

    def handle(self, cfg: DictConfig) -> None:
        if self.ns.project:
            # The environment, not an argument, so op sessions and child processes see it too.
            os.environ["SOLARIAN_PROJECT_DIR"] = str(Path(self.ns.project).expanduser().resolve())
        getattr(self, f"_{self.ns.action}")()

    def _ops(self) -> None:
        from solarian.datasets.op import discover

        catalog = discover()
        for where, message in catalog.errors:
            print(f"warning: {where} did not load: {message}", file=sys.stderr)
        if self.ns.json:
            print(json.dumps([op.describe() for op in catalog.ops.values()], indent=2))
            return
        if catalog.project is not None:
            print(f"Project {catalog.project}")
        for name, op in sorted(catalog.ops.items()):
            outputs = "" if op.outputs == ("out",) else f" -> {', '.join(op.outputs)}"
            gpu = " [gpu]" if op.resources.gpu else ""
            overrides = f", overrides {op.overrides}" if op.overrides else ""
            print(f"{name:20s} {op.kind.value:8s} {op.title}{outputs}{gpu}  ({op.source}{overrides})")

    def _describe(self) -> None:
        from solarian.datasets.runner import get_op

        print(json.dumps(get_op(self.ns.op).describe(), indent=2))

    def _info(self) -> None:
        from solarian.core.recording.archive import Session, targets_member

        session = Session(self.ns.session)
        episodes = session.episodes()
        metas = [session.episode_meta(i) for i in episodes]
        info = {
            "path": str(session.path),
            "episodes": len(episodes),
            "frames": sum(int(m["n_rows"]) for m in metas),
            "cameras": session.cameras,
            "targets": {c: sum(session.exists(targets_member(i, c)) for i in episodes) for c in session.cameras},
            "lineage": [{"path": v["path"], "op": v["op"].get("name"), "params": v["op"].get("params")}
                        for v in session.lineage()],
        }
        if self.ns.json:
            print(json.dumps(info, indent=2))
            return
        print(f"{info['path']}\n  {info['episodes']} episodes, {info['frames']} frames")
        for camera, n in info["targets"].items():
            print(f"  {camera}: targets on {n}/{info['episodes']} episodes")
        for version in info["lineage"]:
            print(f"  <- {version['op']} {json.dumps(version['params'])}  ({version['path']})")

    def _run(self) -> None:
        from solarian.datasets.runner import get_op, run_op

        ns = self.ns
        op = get_op(ns.op)
        params = _parse_params(ns.param, ns.params_json)
        outputs = _parse_outputs(ns.output, op.outputs)
        progress, log, done = _reporters(ns.progress_json)
        try:
            summaries = run_op(ns.op, ns.inputs, outputs, params, workers=ns.jobs, flat=ns.flat,
                               force=ns.force, progress=progress, log=log)
        except Exception as exc:
            if not ns.progress_json:
                raise
            _emit("error", message=f"{type(exc).__name__}: {exc}")
            sys.exit(1)
        if ns.progress_json:
            _emit("result", outputs=[s.as_dict() for s in summaries])
            return
        done()
        for s in summaries:
            print(f"{s.name}: {s.episodes} episodes, {s.frames} frames -> {s.path}")

    def _materialize(self) -> None:
        from solarian.core.recording.archive import Session, iter_zip, materialize

        session = Session(self.ns.session)
        if self.ns.output == "-":
            for chunk in iter_zip(session):
                sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            return
        path = materialize(session, self.ns.output, force=self.ns.force)
        print(f"Wrote {path}")

    def _new(self) -> None:
        from solarian.datasets.computed import ComputedDataset

        computed = ComputedDataset.create(self.ns.source, self.ns.dir)
        print(f"Computed dataset {computed.path} over {computed.source}")

    def _apply(self) -> None:
        ns = self.ns
        computed, parent = _computed(ns.ref)
        params = _parse_params(ns.param, ns.params_json)
        progress, log, done = _reporters(ns.progress_json)
        try:
            keys = computed.apply(ns.op, parent, params, extra=ns.extra, review=ns.review, online=ns.online,
                                  workers=ns.jobs, progress=progress, log=log,
                                  reserved=(lambda keys: _emit("versions", keys=keys)) if ns.progress_json else None)
        except Exception as exc:
            if not ns.progress_json:
                raise
            _emit("error", message=f"{type(exc).__name__}: {exc}")
            sys.exit(1)
        versions = computed.versions()
        if ns.progress_json:
            _emit("result", versions=[{"key": k, **versions[k]} for k in keys])
            return
        done()
        for key in keys:
            summary = versions[key]["summary"]
            print(f"{computed.path.name}@{key}  {versions[key]['state']}: "
                  f"{summary['episodes']} episodes, {summary['frames']} frames, {summary['bytes']} bytes stored")

    def _log(self) -> None:
        from solarian.datasets.computed import ComputedDataset

        computed = ComputedDataset(self.ns.dir)
        if self.ns.json:
            print(json.dumps({"path": str(computed.path), **computed.read(),
                              "source_path": str(computed.source)}, indent=2))
            return
        print("\n".join(_tree(computed)))

    def _close(self) -> None:
        computed, key = _computed(self.ns.ref)
        computed.close(key)
        print(f"Closed {computed.path.name}@{key}")

    def _drop(self) -> None:
        computed, key = _computed(self.ns.ref)
        if key is None:
            raise SystemExit("The source is not a version; delete the computed dataset's directory instead")
        print(f"Dropped {', '.join(computed.drop(key))}")

    def _export(self) -> None:
        from solarian.core.recording.archive import iter_zip

        computed, key = _computed(self.ns.ref)
        if self.ns.output == "-":
            for chunk in iter_zip(computed.session(key)):
                sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            return
        print(f"Wrote {computed.export(key, self.ns.output, force=self.ns.force)}")

    def _rebuild(self) -> None:
        from solarian.datasets.computed import ComputedDataset

        target = ComputedDataset(self.ns.dir).rebuild(self.ns.into, log=print)
        print("\n".join(_tree(target)))

    def _session(self) -> None:
        """One JSON request per stdin line, one event per stdout line, until ``close`` or EOF.

        ``{"episode": k, "params": {...}}`` redoes a map episode,
        ``{"episode": i, "keep": false}`` drops one (a filter's parent episode,
        or a map's own), ``{"accept": k}`` accepts one and ``{"close": true}``
        closes the version. A request's ``id`` comes back on its events.

        While a redo runs, the op's partial results come out as ``preview``
        events. A newer redo of the same episode, a decision on it, or
        ``{"cancel": k}``, stops the one running or queued, which ends with
        ``cancelled``.
        """
        import queue
        import threading

        from solarian.datasets.op import Cancelled
        from solarian.datasets.op_session import OpSession

        computed, key = _computed(self.ns.ref)
        # The newest redo of each episode, set by the reader as requests arrive:
        # any other redo of it is stale, running or not. None after a cancel.
        wanted: dict[int, dict | None] = {}
        current: dict = {}

        def preview(update: dict) -> None:
            _emit("preview", episode=current["episode"], **current["tag"], **update)

        def cancelled() -> bool:
            return wanted.get(current["episode"]) is not current["request"]

        try:
            session = OpSession(computed, key, log=lambda message: _emit("log", message=message),
                                preview=preview, cancelled=cancelled)
        except Exception as exc:
            _emit("error", message=f"{type(exc).__name__}: {exc}")
            sys.exit(1)
        _emit("ready", version=key, kind=session.kind.value)

        requests: queue.Queue[dict | None] = queue.Queue()

        def read() -> None:
            for line in sys.stdin:
                if not line.strip():
                    continue
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as exc:
                    request = {"invalid": f"JSONDecodeError: {exc}"}
                if "cancel" in request:
                    wanted[int(request["cancel"])] = None
                elif "episode" in request:
                    wanted[int(request["episode"])] = None if "keep" in request else request
                requests.put(request)
            requests.put(None)

        threading.Thread(target=read, daemon=True).start()
        while (request := requests.get()) is not None:
            tag = {"id": request["id"]} if "id" in request else {}
            try:
                if "invalid" in request:
                    raise ValueError(request["invalid"])
                if request.get("close"):
                    computed.close(key)
                    _emit("closed", version=key, **tag)
                    return
                if "accept" in request:
                    computed.accept(key, int(request["accept"]))
                    _emit("accepted", episode=int(request["accept"]), **tag)
                    continue
                if "cancel" in request:
                    _emit("cancelled", episode=int(request["cancel"]), **tag)
                    continue
                episode = int(request["episode"])
                if "keep" in request:
                    session.keep(episode, bool(request["keep"]))
                    _emit("kept", episode=episode, keep=bool(request["keep"]), **tag)
                    continue
                if wanted.get(episode) is not request:
                    _emit("cancelled", episode=episode, **tag)
                    continue
                current.update(episode=episode, request=request, tag=tag)
                _emit("started", episode=episode, **tag)
                try:
                    changed = session.redo(episode, request.get("params"))
                except Cancelled:
                    _emit("cancelled", episode=episode, **tag)
                    continue
                _emit("result", episode=episode, changed=changed, **tag)
            except Exception as exc:  # noqa: BLE001 - one bad request must not end the review
                _emit("error", message=f"{type(exc).__name__}: {exc}", episode=request.get("episode"), **tag)

    def _redo(self) -> None:
        from solarian.datasets.op_session import OpSession

        computed, key = _computed(self.ns.ref)
        session = OpSession(computed, key, log=print)
        if self.ns.keep is not None:
            session.keep(self.ns.episode, self.ns.keep)
            print(f"{key}: episode {self.ns.episode} {'kept' if self.ns.keep else 'dropped'}")
            return
        changed = session.redo(self.ns.episode, _parse_params(self.ns.param, self.ns.params_json))
        print(f"{key}: episode {self.ns.episode} redone: {json.dumps(changed)}")

    def _accept(self) -> None:
        computed, key = _computed(self.ns.ref)
        computed.accept(key, self.ns.episode)
        print(f"{key}: episode {self.ns.episode} accepted")
