"""Dataset ops: versions over a session, the built-in ops, and flattening for training."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from solarian.cli.commands.dataset import _parse_params
from solarian.core.recording.archive import (
    VERSION_FILE,
    Session,
    SessionWriter,
    column_stats,
    iter_zip,
    materialize,
    table_bytes,
    targets_bytes,
    video_member,
)
from solarian.datasets import Episode, run_op
from solarian.datasets.ops.mirror import reflect_rotvec
from solarian.datasets.video import VideoEncoder, decode

#: N odd so the temporal midpoint is the home frame, 15.
W, H, N, FPS = 32, 24, 31, 10
CAMERA = "camera0.rgb"
POSE = ("x", "y", "z", "rx", "ry", "rz")


def _frame(k: int) -> np.ndarray:
    """A white square drifting right on black: asymmetric, so a mirror shows."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[8:16, 2 + k % 10:10 + k % 10] = 255
    return frame


def make_session(path: Path, episodes: int = 3, extra_column: bool = False) -> Path:
    """Episodes of N frames whose height ``y`` dwells at home (0) around frame 15."""
    k = np.arange(N)
    y = 0.1 * np.abs(np.sin(np.pi * k / 15))
    y[12:19] = 0
    columns = {
        "t": pa.array(k / FPS, type=pa.float64()),
        "obs.robot0.x": pa.array(0.01 * k + 0.1, type=pa.float32()),
        "obs.robot0.y": pa.array(y, type=pa.float32()),
        "obs.robot0.z": pa.array(np.full(N, 0.2), type=pa.float32()),
        "obs.robot0.rx": pa.array(np.full(N, 0.5), type=pa.float32()),
        "obs.robot0.ry": pa.array(np.full(N, 0.25), type=pa.float32()),
        "obs.robot0.rz": pa.array(np.full(N, -0.125), type=pa.float32()),
        "obs.robot0.gripper_width": pa.array(np.full(N, 0.05), type=pa.float32()),
        "cam.camera0.rgb.frame_index": pa.array(k, type=pa.int32()),
    }
    if extra_column:
        columns["obs.robot0.extra"] = pa.array(np.zeros(N), type=pa.float32())
    table = pa.table(columns)
    schema = {name: {"dtype": np.dtype(table.schema.field(name).type.to_pandas_dtype()).name, "unit": None}
              for name in table.column_names}
    with SessionWriter(path) as writer:
        writer.write_json("session.json", {
            "fps": FPS, "cameras": {CAMERA: {"width": W, "height": H}}, "schema": schema,
            "robot": {"modules": {"robot0": {"kind": "gripper", "pose_frame": "tag"}}},
            "obs_fields": {"robot0": [*POSE, "gripper_width"] + (["extra"] if extra_column else [])},
        })
        for i in range(episodes):
            encoder = VideoEncoder(writer.scratch / f"{i}.mp4", FPS)
            for j in range(N):
                encoder.write(_frame(j))
            pose = np.stack([table.column(f"obs.robot0.{f}").to_numpy() for f in POSE], axis=-1)
            bbox = np.tile(np.array([[2, 8, 10, 16]], dtype=np.float32), (N, 1))
            bbox[3] = -1
            point = np.tile(np.array([[3.5, 9.0]], dtype=np.float32), (N, 1))
            point[4] = -1
            writer.add_episode(
                {"index": i, "n_rows": N, "start_ts": 100.0 * i, "end_ts": 100.0 * i + N / FPS,
                 "status": "ok", "n_frames": {CAMERA: N}, "stats": column_stats(table, {"obs.robot0.y": {}}),
                 "umi": {"robot0": {"demo_start_pose": pose[0].tolist(), "demo_end_pose": pose[-1].tolist()}}},
                files={
                    "episode.parquet": table_bytes(table),
                    f"video/{CAMERA}.mp4": encoder.close(),
                    f"tags/{CAMERA}.json": json.dumps([{"7": [[1, 2], [3, 2], [3, 4], [1, 4]]}] * N).encode(),
                    f"targets/{CAMERA}.npz": targets_bytes(bbox=bbox, point=point),
                },
            )
    return path


@pytest.fixture
def session(tmp_path) -> Path:
    return make_session(tmp_path / "session")


def _stored(path: Path) -> set[str]:
    return {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}


def test_delete_stores_only_metadata_and_reads_through(session, tmp_path):
    out = tmp_path / "deleted"
    [summary] = run_op("delete-episodes", [session], {"out": out}, {"episodes": [1]})

    assert summary.episodes == 2
    assert _stored(out) == {"version.json", "manifest.jsonl",
                            "episodes/000000/episode.json", "episodes/000001/episode.json"}
    version = Session(out)
    source = Session(session)
    assert version.episodes() == [0, 1]
    assert version.episode_meta(1)["index"] == 1
    assert version.episode_meta(1)["start_ts"] == source.episode_meta(2)["start_ts"]
    assert version.read_bytes(video_member(1, CAMERA)) == source.read_bytes(video_member(2, CAMERA))
    assert version.locate(video_member(1, CAMERA)) == source.path / video_member(2, CAMERA)
    assert version.lineage()[0]["op"]["name"] == "delete-episodes"


def test_unknown_episode_and_unknown_param_are_refused(session, tmp_path):
    with pytest.raises(ValueError, match="Not episodes"):
        run_op("delete-episodes", [session], {"out": tmp_path / "a"}, {"episodes": [9]})
    with pytest.raises(ValueError, match="episdoes"):
        run_op("delete-episodes", [session], {"out": tmp_path / "b"}, {"episdoes": [1]})
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_split_cuts_at_the_return_home_into_two_outputs(session, tmp_path):
    outputs = {"pick": tmp_path / "pick", "place": tmp_path / "place"}
    summaries = run_op("split-pick-place", [session], outputs, {})

    assert {s.name: s.episodes for s in summaries} == {"pick": 3, "place": 3}
    pick, place = Session(outputs["pick"]), Session(outputs["place"])
    for i in range(3):
        a, b = pick.read_table(i), place.read_table(i)
        assert (a.num_rows, b.num_rows) == (16, 16)
        # The home frame is the last of pick and the first of place.
        assert a.column("obs.robot0.x")[-1] == b.column("obs.robot0.x")[0]
        assert b.column("t")[0].as_py() == 0 and b.column("cam.camera0.rgb.frame_index")[0].as_py() == 0
        assert len(list(decode(place, video_member(i, CAMERA)))) == 16
        assert len(place.read_json(f"episodes/{i:06d}/tags/{CAMERA}.json")) == 16
        assert place.read_targets(i, CAMERA)["bbox"].shape == (16, 4)
        start = place.episode_meta(i)["umi"]["robot0"]["demo_start_pose"]
        assert start[0] == pytest.approx(0.01 * 15 + 0.1)


def test_split_failure_leaves_neither_output(session, tmp_path):
    outputs = {"pick": tmp_path / "pick", "place": tmp_path / "place"}
    with pytest.raises(ValueError, match="short half"):
        run_op("split-pick-place", [session], outputs, {"min_frames": 20, "strict": True})
    assert list(tmp_path.iterdir()) == [session]


def test_mirror_follows_each_episode_with_its_reflection(session, tmp_path):
    out = tmp_path / "mirrored"
    run_op("mirror", [session], {"out": out}, {"axis": "x"})

    mirrored = Session(out)
    assert mirrored.episodes() == list(range(6))
    assert not any(name.startswith("episodes/000000/") and name.endswith(".mp4") for name in _stored(out))
    original, mirror = mirrored.read_table(0), mirrored.read_table(1)
    assert np.allclose(mirror.column("obs.robot0.x").to_numpy(), -original.column("obs.robot0.x").to_numpy())
    rot = np.stack([original.column(f"obs.robot0.{f}").to_numpy() for f in ("rx", "ry", "rz")], axis=-1)
    assert np.allclose(np.stack([mirror.column(f"obs.robot0.{f}").to_numpy() for f in ("rx", "ry", "rz")],
                                axis=-1), reflect_rotvec(rot, 0))
    bbox = mirrored.read_targets(1, CAMERA)["bbox"]
    assert bbox[0].tolist() == [W - 10, 8, W - 2, 16]
    assert bbox[3].tolist() == [-1, -1, -1, -1]
    point = mirrored.read_targets(1, CAMERA)["point"]
    assert point[0].tolist() == [W - 3.5, 9.0]
    assert point[4].tolist() == [-1, -1]
    assert mirrored.read_json(f"episodes/000001/tags/{CAMERA}.json")[0]["7"][0] == [W - 1 - 1, 2]
    first = next(decode(mirrored, video_member(1, CAMERA))).astype(int)
    assert np.abs(first - _frame(0)[:, ::-1]).mean() < 10
    assert mirrored.episode_meta(1)["umi"]["robot0"]["demo_start_pose"][0] == pytest.approx(-0.1)


def test_merge_keeps_common_columns_and_rewrites_only_what_differs(session, tmp_path):
    other = make_session(tmp_path / "other", episodes=2, extra_column=True)
    out = tmp_path / "merged"
    run_op("merge", [session, other], {"out": out}, {})

    merged = Session(out)
    assert merged.episodes() == list(range(5))
    assert "obs.robot0.extra" not in merged.meta["schema"]
    assert "extra" not in merged.meta["obs_fields"]["robot0"]
    stored = _stored(out)
    assert "episodes/000000/episode.parquet" not in stored
    assert "episodes/000003/episode.parquet" in stored
    assert "obs.robot0.extra" not in merged.read_table(3).column_names


def test_a_version_chain_flattens_to_the_zip_training_reads(session, tmp_path):
    run_op("delete-episodes", [session], {"out": tmp_path / "v1"}, {"episodes": [0]})
    run_op("mirror", [tmp_path / "v1"], {"out": tmp_path / "v2"}, {})
    version = Session(tmp_path / "v2")

    flat = materialize(version, tmp_path / "flat.solarian.zip")
    with zipfile.ZipFile(flat) as zf:
        names = set(zf.namelist())
    assert VERSION_FILE not in names
    assert {"session.json", "manifest.jsonl", "lineage.json", video_member(3, CAMERA)} <= names
    lineage = json.loads(Session(flat).read_bytes("lineage.json"))
    assert [v["op"]["name"] for v in lineage] == ["mirror", "delete-episodes"]

    streamed = tmp_path / "streamed.solarian.zip"
    streamed.write_bytes(b"".join(iter_zip(version)))
    a, b = Session(flat), Session(streamed)
    assert a.names == b.names
    for name in a.names - {"manifest.jsonl", "lineage.json"}:
        assert a.read_bytes(name) == b.read_bytes(name), name


def test_targets_read_back_as_written(session):
    targets = Session(session).read_targets(0, CAMERA)
    assert targets["bbox"].dtype == np.float32 and targets["bbox"].shape == (N, 4)
    with np.load(io.BytesIO(targets_bytes(bbox=targets["bbox"]))) as npz:
        assert np.array_equal(npz["bbox"], targets["bbox"])


def test_derived_targets_keep_the_other_ops_arrays(session):
    episode = Episode(Session(session), 0)
    point = np.full((N, 2), 5.0, dtype=np.float32)
    out = episode.derive_targets(CAMERA, {"point": point}, replaces=("point",), coverage={"found": N, "frames": N})
    with np.load(io.BytesIO(out.files[f"targets/{CAMERA}.npz"])) as npz:
        assert set(npz.files) == {"bbox", "point"}
        assert np.array_equal(npz["bbox"], episode.targets(CAMERA)["bbox"])
    assert out.meta["targets"][CAMERA] == {"found": N, "frames": N}

    out = episode.derive_targets(CAMERA, {"point": point}, replaces=("bbox", "point"), coverage={})
    with np.load(io.BytesIO(out.files[f"targets/{CAMERA}.npz"])) as npz:
        assert set(npz.files) == {"point"}


def test_cli_params_read_values_as_yaml_and_dotted_keys_nest():
    params = _parse_params(["prompt.point=[640, 400]", "prompt.frame=-1", "axis=x"], '{"mask": true}')
    assert params == {"mask": True, "prompt": {"point": [640, 400], "frame": -1}, "axis": "x"}
