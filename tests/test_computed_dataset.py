"""Computed datasets: the version tree, per-episode review, and the CLI over both."""

from __future__ import annotations

import io
import json
import sys
import threading
import time

import numpy as np
import pytest
from pydantic import BaseModel

from test_dataset_ops import CAMERA, N, make_session

from solarian.cli.commands.dataset import DatasetCommand
from solarian.core.recording.archive import Session, targets_bytes, targets_member
from solarian.datasets import Cancelled, DatasetOp, Episode, EpisodeOut, OpContext, OpKind, runner
from solarian.datasets.computed import ACCEPTED, CLOSED, DONE, OPEN, PENDING, ComputedDataset, summarize
from solarian.datasets.op_session import OpSession


class MarkParams(BaseModel):
    value: float | None = None


class MarkOp(DatasetOp):
    """Writes ``value`` as every box of an episode; with no value it leaves the episode alone."""

    name = "mark"
    kind = OpKind.MAP
    Params = MarkParams
    setups = 0

    def setup(self, params: MarkParams, ctx: OpContext) -> None:
        type(self).setups += 1

    def map(self, episode: Episode, params: MarkParams, ctx: OpContext) -> EpisodeOut | None:
        if params.value is None:
            return None
        bbox = np.full((episode.n_rows, 4), params.value, dtype=np.float32)
        return episode.derive({f"targets/{CAMERA}.npz": targets_bytes(bbox=bbox)},
                              targets={CAMERA: {"value": params.value}})


class SlowMarkOp(MarkOp):
    """``mark``, previewing each frame; ``value: 0`` previews until it is cancelled."""

    name = "slow-mark"
    previewed = threading.Event()

    def map(self, episode: Episode, params: MarkParams, ctx: OpContext) -> EpisodeOut | None:
        for frame in range(500 if params.value == 0 else 1):
            ctx.preview({"frame": frame})
            type(self).previewed.set()
            if ctx.cancelled():
                raise Cancelled
            time.sleep(0.01)
        return super().map(episode, params, ctx)


@pytest.fixture(autouse=True)
def mark_op(monkeypatch):
    ops = {**runner.discover_ops(), "mark": MarkOp, "slow-mark": SlowMarkOp}
    monkeypatch.setattr(runner, "discover_ops", lambda: ops)
    MarkOp.setups = 0


@pytest.fixture
def computed(tmp_path) -> ComputedDataset:
    return ComputedDataset.create(make_session(tmp_path / "source", episodes=4), tmp_path / "cups")


def _box(session: Session, episode: int) -> float:
    return float(np.load(io.BytesIO(session.read_bytes(targets_member(episode, CAMERA))))["bbox"][0, 0])


def test_versions_stack_into_a_tree(computed):
    [deleted] = computed.apply("delete-episodes", params={"episodes": [1]})
    pick, place = computed.apply("split-pick-place", deleted)
    [mirrored] = computed.apply("mirror", pick)

    assert (deleted, pick, place, mirrored) == (
        "01-delete-episodes", "02-split-pick-place-pick", "02-split-pick-place-place", "03-mirror")
    versions = computed.versions()
    assert versions[mirrored]["parent"] == pick and versions[pick]["parent"] == deleted
    assert all(v["state"] == CLOSED for v in versions.values())
    assert versions[deleted]["summary"]["episodes"] == 3
    assert sorted(computed.leaves()) == [place, mirrored]
    with pytest.raises(ValueError, match="several leaves"):
        computed.resolve(None)
    assert computed.resolve("source") is None
    assert len(computed.session(mirrored).episodes()) == 2 * versions[pick]["summary"]["episodes"]


def test_drop_takes_the_subtree_and_export_flattens(computed, tmp_path):
    [deleted] = computed.apply("delete-episodes", params={"episodes": [0]})
    pick, place = computed.apply("split-pick-place", deleted)
    [mirrored] = computed.apply("mirror", pick)

    out = computed.export(mirrored, tmp_path / "flat")
    assert Session(out).version is None
    assert Session(out).episodes() == computed.session(mirrored).episodes()

    assert computed.drop(pick) == [pick, mirrored]
    assert set(computed.versions()) == {deleted, place}
    assert not computed.version_path(mirrored).exists()


def test_a_failed_op_leaves_no_version(computed):
    with pytest.raises(ValueError):
        computed.apply("delete-episodes", params={"episodes": [9]})
    assert computed.versions() == {}
    assert list((computed.path / "versions").iterdir()) == []


def test_an_open_version_cannot_be_built_on(computed):
    [marked] = computed.apply("mark", params={"value": 1}, review=True)
    assert computed.entry(marked)["state"] == OPEN
    with pytest.raises(ValueError, match="still open"):
        computed.apply("mirror", marked)
    with pytest.raises(ValueError, match="only map and filter"):
        computed.apply("split-pick-place", review=True)
    computed.close(marked)
    computed.apply("mirror", marked)


def test_online_map_reads_through_until_redone(computed):
    [key] = computed.apply("mark", online=True)
    entry = computed.entry(key)
    assert entry["state"] == OPEN and entry["online"]
    assert {e["status"] for e in entry["episodes"].values()} == {PENDING}
    version, source = computed.session(key), computed.session(None)
    assert _box(version, 2) == _box(source, 2) == 2.0

    session = OpSession(computed, key)
    changed = session.redo(2, {"value": 7})
    assert changed == {"targets": {CAMERA: {"value": 7.0}}}
    assert _box(computed.session(key), 2) == 7.0
    assert _box(computed.session(key), 1) == 2.0
    session.redo(2, {"value": 5})
    assert _box(computed.session(key), 2) == 5.0
    assert MarkOp.setups == 1

    computed.accept(key, 2)
    assert computed.entry(key)["episodes"]["2"] == {"status": ACCEPTED, "params": {"value": 5}, "result": changed
                                                   | {"targets": {CAMERA: {"value": 5.0}}}}
    with pytest.raises(ValueError, match="episodes 0-3"):
        session.redo(4, {"value": 1})


def test_a_redo_keeps_the_summary_as_a_full_count_would(computed):
    [key] = computed.apply("mark", online=True)
    session = OpSession(computed, key)
    for episode, value in [(2, 7), (2, 5), (0, None), (1, 3), (2, None)]:
        session.redo(episode, {"value": value})
        assert computed.entry(key)["summary"] == summarize(computed.version_path(key))


def test_a_filter_review_overrides_the_decisions(computed):
    [key] = computed.apply("delete-episodes", params={"episodes": [1]}, review=True)
    review = computed.entry(key)["episodes"]
    assert {i: e["keep"] for i, e in review.items()} == {"0": True, "1": False, "2": True, "3": True}

    session = OpSession(computed, key)
    session.keep(1, True)
    session.keep(3, False)
    session.keep(3, False)
    open_summary = computed.entry(key)["summary"]
    version = computed.session(key)
    assert [version.episode_meta(i)["start_ts"] for i in version.episodes()] == [0.0, 200.0, 300.0], \
        "the version is rewritten when it closes"

    computed.close(key)
    version = computed.session(key)
    assert [version.episode_meta(i)["start_ts"] for i in version.episodes()] == [0.0, 100.0, 200.0]
    assert computed.entry(key)["summary"] == summarize(computed.version_path(key))
    assert {k: open_summary[k] for k in ("episodes", "frames")} == \
        {k: computed.entry(key)["summary"][k] for k in ("episodes", "frames")}
    with pytest.raises(ValueError, match="keep"):
        session.redo(0)


def test_a_map_review_drops_episodes_when_the_version_closes(computed):
    [key] = computed.apply("mark", online=True)
    session = OpSession(computed, key)
    session.redo(2, {"value": 7})
    session.redo(1, {"value": 4})
    session.keep(1, False)
    session.keep(1, False)
    assert computed.entry(key)["episodes"]["1"] == {
        "status": ACCEPTED, "params": {"value": 4}, "result": {"targets": {CAMERA: {"value": 4.0}}}, "keep": False}
    open_summary = computed.entry(key)["summary"]
    assert open_summary["episodes"] == 3
    assert computed.session(key).episodes() == [0, 1, 2, 3], "the version is rewritten when it closes"

    computed.close(key)
    version = computed.session(key)
    assert [version.episode_meta(i)["start_ts"] for i in version.episodes()] == [0.0, 200.0, 300.0]
    assert [_box(version, i) for i in version.episodes()] == [2.0, 7.0, 2.0]
    assert computed.entry(key)["summary"] == summarize(computed.version_path(key))
    assert {k: open_summary[k] for k in ("episodes", "frames")} == \
        {k: computed.entry(key)["summary"][k] for k in ("episodes", "frames")}


def test_a_redo_or_a_keep_brings_a_dropped_map_episode_back(computed):
    [key] = computed.apply("mark", online=True)
    session = OpSession(computed, key)
    session.keep(0, False)
    session.keep(3, False)
    session.redo(0, {"value": 5})
    session.keep(3, True)
    review = computed.entry(key)["episodes"]
    assert "keep" not in review["0"] and review["0"]["status"] == DONE
    assert review["3"] == {"status": ACCEPTED}
    assert computed.entry(key)["summary"] == summarize(computed.version_path(key))

    computed.close(key)
    assert computed.session(key).episodes() == [0, 1, 2, 3]
    assert _box(computed.session(key), 0) == 5.0


def test_rebuild_replays_ops_and_review(computed, tmp_path):
    [deleted] = computed.apply("delete-episodes", params={"episodes": [0]}, review=True)
    OpSession(computed, deleted).keep(0, True)
    computed.close(deleted)
    [marked] = computed.apply("mark", deleted, online=True)
    session = OpSession(computed, marked)
    session.redo(1, {"value": 9})
    computed.accept(marked, 1)
    session.redo(2, {"value": 8})
    session.keep(2, False)
    computed.close(marked)

    rebuilt = computed.rebuild(tmp_path / "again")
    assert list(rebuilt.versions()) == [deleted, marked]
    assert rebuilt.entry(marked)["state"] == CLOSED
    assert rebuilt.entry(marked)["episodes"]["1"]["status"] == ACCEPTED
    assert rebuilt.entry(marked)["episodes"]["0"]["status"] == PENDING
    assert rebuilt.entry(marked)["episodes"]["2"]["keep"] is False
    assert rebuilt.session(deleted).episodes() == [0, 1, 2, 3]
    assert rebuilt.session(marked).episodes() == [0, 1, 2]
    assert _box(rebuilt.session(marked), 1) == 9.0
    assert _box(rebuilt.session(marked), 0) == _box(computed.session(marked), 0)


def test_batch_review_marks_episodes_done(computed):
    [key] = computed.apply("mark", params={"value": 3}, review=True)
    assert {e["status"] for e in computed.entry(key)["episodes"].values()} == {DONE}
    assert _box(computed.session(key), 0) == 3.0


def _cli(argv: list[str]) -> None:
    command = DatasetCommand()
    command.parse_args(argv)
    command.handle(None)


def test_cli_new_apply_log_export(computed, tmp_path, capsys):
    ref = str(computed.path)
    _cli(["new", str(computed.source), str(tmp_path / "other")])
    assert (tmp_path / "other" / "computed.json").is_file()
    _cli(["apply", ref, "delete-episodes", "--param", "episodes=[2]"])
    _cli(["apply", ref, "split-pick-place"])
    capsys.readouterr()
    _cli(["apply", f"{ref}@02-split-pick-place-pick", "mirror", "--progress-json"])
    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert events[0] == {"event": "versions", "keys": ["03-mirror"]}
    assert events[-1]["event"] == "result" and events[-1]["versions"][0]["key"] == "03-mirror"

    _cli(["log", ref])
    tree = capsys.readouterr().out
    assert "01-delete-episodes  closed, 3 episodes" in tree
    assert "03-mirror" in tree
    _cli(["export", f"{ref}@03-mirror", str(tmp_path / "flat")])
    assert Session(tmp_path / "flat").episodes() == computed.session("03-mirror").episodes()
    _cli(["drop", f"{ref}@02-split-pick-place-pick"])
    assert set(computed.versions()) == {"01-delete-episodes", "02-split-pick-place-place"}


def test_cli_session_speaks_json_lines(computed, monkeypatch, capsys):
    [key] = computed.apply("mark", online=True)
    requests = [{"id": "a", "episode": 1, "params": {"value": 4}}, {"episode": 9}, {"accept": 1}, {"close": True}]
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(json.dumps(r) + "\n" for r in requests)))
    _cli(["session", str(computed.path)])

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [e["event"] for e in events] == ["ready", "started", "result", "started", "error", "accepted", "closed"]
    assert events[2] == {"event": "result", "episode": 1, "changed": {"targets": {CAMERA: {"value": 4.0}}},
                         "id": "a"}
    assert "episodes 0-3" in events[4]["message"] and events[4]["episode"] == 9
    assert computed.entry(key)["state"] == CLOSED
    assert _box(computed.session(key), 1) == 4.0


def test_cli_redo_and_accept(computed, capsys):
    [key] = computed.apply("delete-episodes", params={"episodes": [0]}, review=True)
    _cli(["redo", str(computed.path), "0", "--keep"])
    _cli(["accept", str(computed.path), "2"])
    review = computed.entry(key)["episodes"]
    assert review["0"] == {"status": ACCEPTED, "keep": True}
    assert review["2"]["status"] == ACCEPTED
    computed.close(key)
    assert computed.session(key).episodes() == [0, 1, 2, 3]
    assert N == computed.session(key).episode_meta(0)["n_rows"]


def test_cli_session_previews_a_redo_until_it_is_stopped(computed, monkeypatch, capsys):
    [key] = computed.apply("slow-mark", online=True)
    SlowMarkOp.previewed.clear()

    def stdin():
        yield json.dumps({"id": "a", "episode": 1, "params": {"value": 0}}) + "\n"
        SlowMarkOp.previewed.wait(5)
        yield json.dumps({"id": "c", "cancel": 1}) + "\n"
        yield json.dumps({"id": "b", "episode": 1, "params": {"value": 6}}) + "\n"
        yield json.dumps({"close": True}) + "\n"

    monkeypatch.setattr(sys, "stdin", stdin())
    _cli(["session", str(computed.path)])

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    order = [(e["event"], e.get("id")) for e in events if e["event"] != "preview"]
    assert order == [("ready", None), ("started", "a"), ("cancelled", "a"), ("cancelled", "c"),
                     ("started", "b"), ("result", "b"), ("closed", None)]
    assert {"event": "preview", "episode": 1, "id": "a", "frame": 0} in events
    assert [e for e in events if e["event"] == "preview" and e["id"] == "b"] == [
        {"event": "preview", "episode": 1, "id": "b", "frame": 0}]
    assert _box(computed.session(key), 1) == 6.0


def test_cli_session_drop_stops_the_episodes_redo(computed, monkeypatch, capsys):
    [key] = computed.apply("slow-mark", online=True)
    SlowMarkOp.previewed.clear()

    def stdin():
        yield json.dumps({"id": "a", "episode": 1, "params": {"value": 0}}) + "\n"
        SlowMarkOp.previewed.wait(5)
        yield json.dumps({"id": "d", "episode": 1, "keep": False}) + "\n"
        yield json.dumps({"close": True}) + "\n"

    monkeypatch.setattr(sys, "stdin", stdin())
    _cli(["session", str(computed.path)])

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    order = [(e["event"], e.get("id")) for e in events if e["event"] != "preview"]
    assert order == [("ready", None), ("started", "a"), ("cancelled", "a"), ("kept", "d"), ("closed", None)]
    assert len(computed.session(key).episodes()) == 3
