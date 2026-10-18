"""A run fanned out to its backends, with wandb's surface."""

from __future__ import annotations

import atexit
import logging
import sys
from typing import Any, Iterable, Mapping

from solarian.tracking import registry
from solarian.tracking.system import system_info
from solarian.tracking.tags import clean_tags
from solarian.tracking.tracker import RunSpec, Tracker

logger = logging.getLogger(__name__)

_current: "Run | None" = None
_uncaught = False
_hooks_installed = False


def _note_uncaught(previous):
    def hook(exc_type, exc, tb):
        global _uncaught
        _uncaught = True
        previous(exc_type, exc, tb)
    return hook


class Summary(dict):
    """``run.summary["best"] = x`` reaches every backend."""

    def __init__(self, run: "Run"):
        super().__init__()
        self._run = run

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self._run._each("update_summary", {key: value})

    def update(self, values: Mapping[str, Any] = (), **more: Any) -> None:  # type: ignore[override]
        merged = {**dict(values), **more}
        super().update(merged)
        self._run._each("update_summary", merged)


class Run:
    def __init__(self, spec: RunSpec, trackers: list[Tracker]):
        self.spec = spec
        self.trackers = trackers
        self.summary = Summary(self)
        self.finished = False
        self._failed: set[str] = set()

    @property
    def name(self) -> str | None:
        return self.spec.name

    @property
    def tags(self) -> list[str]:
        return self.spec.tags

    @property
    def config(self) -> dict[str, Any]:
        return self.spec.config

    @property
    def urls(self) -> dict[str, str]:
        return {t.name: t.url for t in self.trackers if t.url}

    def _each(self, method: str, *args: Any) -> None:
        for tracker in self.trackers:
            try:
                getattr(tracker, method)(*args)
            except Exception as exc:  # noqa: BLE001 - tracking must not end training
                if tracker.name not in self._failed:
                    self._failed.add(tracker.name)
                    print(f"[tracking] {tracker.name}.{method} failed, continuing: {exc}", file=sys.stderr)

    def log(self, values: Mapping[str, Any], step: int | None = None) -> None:
        if not self.finished:
            self._each("log", dict(values), step)

    def finish(self, exit_code: int = 0) -> None:
        global _current
        if self.finished:
            return
        self.finished = True
        self._each("finish", exit_code)
        if _current is self:
            _current = None


def init(
    project: str | None = None,
    name: str | None = None,
    *,
    group: str | None = None,
    tags: Iterable[object] | None = None,
    notes: str | None = None,
    config: Mapping[str, Any] | None = None,
    id: str | None = None,
    resume: str | bool | None = None,
    mode: str | None = None,
    dir: str | None = None,
    job_type: str | None = None,
    backend: str | Iterable[str] | None = None,
    **extra: Any,
) -> Run:
    """Start a run on the backends ``backend`` (or ``$SOLARIAN_TRACKING``) names.

    Takes wandb.init's arguments, so moving a script off wandb is a change of
    import. Keys no argument names go to the backends as ``extra``.
    """
    global _current
    spec = RunSpec(
        project=project, name=name, group=group, tags=clean_tags(tags), notes=notes,
        config=dict(config or {}), id=id, resume=resume, mode=mode, dir=dir,
        job_type=job_type, system=system_info(), extra=extra,
    )
    names = registry.resolve_backends(backend)
    available = registry.discover_trackers()
    trackers = [available[n](spec) for n in names]
    run = Run(spec, trackers)
    global _hooks_installed
    if not _hooks_installed:
        _hooks_installed = True
        sys.excepthook = _note_uncaught(sys.excepthook)
        atexit.register(_finish_at_exit)
    _current = run
    for tracker_name, url in run.urls.items():
        print(f"[tracking] {tracker_name}: {url}")
    return run


def current() -> Run | None:
    return _current


def log(values: Mapping[str, Any], step: int | None = None) -> None:
    if _current is None:
        raise RuntimeError("tracking.log before tracking.init")
    _current.log(values, step)


def finish(exit_code: int = 0) -> None:
    if _current is not None:
        _current.finish(exit_code)


def _finish_at_exit() -> None:
    if _current is not None and not _current.finished:
        _current.finish(1 if _uncaught else 0)
