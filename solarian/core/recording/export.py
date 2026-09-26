"""Turn a recorded session into the format a trainer or a cloud wants.

A session on the robot is deliberately *not* a training set. It keeps the
frame the sensor produced, the signals at their own rates, and the calibration
as a reference rather than baked in — because all three of those decisions can
only be reversed by recording again. Export is where the irreversible choices
are finally made: a target rate, a policy resolution, and a warped image.

That makes export cheap to redo and safe to get wrong. Recalibrate the camera
and re-export; decide the policy wants 60 Hz and re-export. The session does
not change.

Only LeRobot v2.1 is implemented here. It is what PrimaryKey ingests, and it
is the format every off-the-shelf trainer reads.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from solarian.core.recording.session import (
    EPISODE_FILE,
    TABLE_FILE,
    SessionPaths,
)

logger = logging.getLogger(__name__)

#: LeRobot's on-disk contract version. v3 moved to one file per chunk rather
#: than one per episode; PrimaryKey and the trainers in use read v2.1.
CODEBASE_VERSION = "v2.1"

#: Episodes per chunk directory. LeRobot's own default, and what every reader
#: assumes when it globs ``chunk-000``.
CHUNK_SIZE = 1000

DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)


@dataclass(frozen=True)
class ExportStats:
    n_episodes: int
    n_frames: int
    n_videos: int
    path: Path


def export_lerobot(
    session: Path,
    out: Path,
    *,
    fps: float | None = None,
    episodes: list[int] | None = None,
    remap: bool = True,
    overwrite: bool = False,
) -> ExportStats:
    """Write *session* out as a LeRobot v2.1 dataset at *out*.

    Args:
        fps: Rate every signal is resampled onto. Defaults to the session's own
            video grid, which is the one rate nothing has to be interpolated to.
        episodes: Indices to include. ``None`` takes every sealed episode.
        remap: Apply each camera's optical remap and downscale to the
            observation resolution, i.e. produce the frames a policy trains on.
            ``False`` copies the archive through untouched, which is what a
            re-calibration or a higher-resolution experiment wants.
        overwrite: Replace *out* if it exists.
    """
    paths = SessionPaths(Path(session))
    meta = json.loads(paths.session_json.read_text())
    schema = meta["schema"]
    fps = float(fps if fps is not None else meta["fps"])

    out = Path(out)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists; pass overwrite to replace it")
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    selected = _episode_dirs(paths, episodes)
    if not selected:
        raise ValueError(f"no sealed episodes under {paths.episodes}")

    state_columns = [c for c in schema if c.startswith("obs.") and not c.endswith(".ts")]
    action_columns = [c for c in schema if c.startswith("act.")]
    cameras = meta["cameras"]

    tasks: dict[str, int] = {}
    episode_rows: list[dict] = []
    episode_stats: list[dict] = []
    n_frames = 0

    for index, (source_index, episode_dir) in enumerate(selected):
        episode_meta = json.loads((episode_dir / EPISODE_FILE).read_text())
        task = episode_meta.get("task") or ""
        task_index = tasks.setdefault(task, len(tasks))

        frame = _resample(
            pq.read_table(episode_dir / TABLE_FILE),
            fps=fps,
            state_columns=state_columns,
            action_columns=action_columns,
        )
        _write_episode_table(out, index, frame, task_index, n_frames, fps)

        for name in cameras:
            source = episode_dir / "video" / f"{name}.mp4"
            if not source.is_file():
                logger.warning("Episode %d has no video for %s", source_index, name)
                continue
            target = out / VIDEO_PATH.format(
                episode_chunk=index // CHUNK_SIZE,
                video_key=_video_key(name),
                episode_index=index,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            _export_video(
                source,
                target,
                frame_indices=frame["frame_index_by_camera"][name],
                camera=cameras[name],
                fps=fps,
                remap=remap,
            )

        episode_rows.append(
            {
                "episode_index": index,
                "tasks": [task],
                "length": frame["n_rows"],
            }
        )
        episode_stats.append(
            {"episode_index": index, "stats": _stats_for(frame, state_columns, action_columns)}
        )
        n_frames += frame["n_rows"]
        logger.info(
            "Exported episode %d -> %d (%d frames)", source_index, index, frame["n_rows"]
        )

    video_keys = [_video_key(name) for name in cameras]
    _write_jsonl(out / "meta" / "episodes.jsonl", episode_rows)
    _write_jsonl(out / "meta" / "episodes_stats.jsonl", episode_stats)
    _write_jsonl(
        out / "meta" / "tasks.jsonl",
        [{"task_index": i, "task": t} for t, i in sorted(tasks.items(), key=lambda kv: kv[1])],
    )
    (out / "meta" / "info.json").write_text(
        json.dumps(
            _info(
                meta=meta,
                fps=fps,
                n_episodes=len(episode_rows),
                n_frames=n_frames,
                n_tasks=len(tasks),
                state_columns=state_columns,
                action_columns=action_columns,
                cameras=cameras,
                remap=remap,
            ),
            indent=2,
        )
    )

    return ExportStats(
        n_episodes=len(episode_rows),
        n_frames=n_frames,
        n_videos=len(episode_rows) * len(video_keys),
        path=out,
    )


# -- the table ---------------------------------------------------------------


def _resample(table: pa.Table, *, fps: float, state_columns, action_columns) -> dict:
    """Put every signal on one constant-rate grid.

    The session table has a row per module update, so its rows are neither
    evenly spaced nor one-per-camera-frame. LeRobot's contract is the
    opposite: row *i* is the state at ``i / fps``. Each grid slot therefore
    takes the last row at or before it — a sample-and-hold, which is what the
    forward-filled columns already are and the only resampling that cannot
    invent a value the robot never reported.
    """
    t = table.column("t").to_numpy()
    n_rows = max(int(math.floor(t[-1] * fps)) + 1, 1) if len(t) else 0
    grid = np.arange(n_rows) / fps
    # searchsorted gives the first row strictly after the slot; the row before
    # it is the most recent sample, clamped so slot 0 uses the first row.
    rows = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, max(len(t) - 1, 0))

    def stack(columns):
        if not columns:
            return np.zeros((n_rows, 0), dtype=np.float32)
        return np.stack(
            [table.column(c).to_numpy()[rows] for c in columns], axis=1
        ).astype(np.float32)

    frame_index_by_camera = {}
    for column in table.schema.names:
        if column.startswith("cam.") and column.endswith(".frame_index"):
            name = column[len("cam.") : -len(".frame_index")]
            frame_index_by_camera[name] = table.column(column).to_numpy()[rows]

    return {
        "n_rows": n_rows,
        "timestamp": grid.astype(np.float32),
        "state": stack(state_columns),
        "action": stack(action_columns),
        "frame_index_by_camera": frame_index_by_camera,
    }


def _write_episode_table(
    out: Path, index: int, frame: dict, task_index: int, offset: int, fps: float
) -> None:
    n = frame["n_rows"]
    path = out / DATA_PATH.format(episode_chunk=index // CHUNK_SIZE, episode_index=index)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "observation.state": _vector_column(frame["state"]),
                "action": _vector_column(frame["action"]),
                "timestamp": pa.array(frame["timestamp"], pa.float32()),
                "frame_index": pa.array(np.arange(n), pa.int64()),
                "episode_index": pa.array(np.full(n, index), pa.int64()),
                "index": pa.array(np.arange(offset, offset + n), pa.int64()),
                "task_index": pa.array(np.full(n, task_index), pa.int64()),
            }
        ),
        path,
        compression="zstd",
    )


def _vector_column(values: np.ndarray) -> pa.Array:
    """One fixed-width float32 list per row, which is LeRobot's feature shape."""
    n_rows, width = values.shape
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), pa.float32()), max(width, 1)
    ) if width else pa.array([[]] * n_rows, pa.list_(pa.float32()))


def _stats_for(frame: dict, state_columns, action_columns) -> dict:
    stats = {}
    for key, values in (("observation.state", frame["state"]), ("action", frame["action"])):
        if values.shape[1] == 0:
            continue
        finite = np.where(np.isfinite(values), values, 0.0)
        stats[key] = {
            "min": finite.min(axis=0).tolist(),
            "max": finite.max(axis=0).tolist(),
            "mean": finite.mean(axis=0).tolist(),
            "std": finite.std(axis=0).tolist(),
            "count": [int(values.shape[0])],
        }
    return stats


# -- the video ---------------------------------------------------------------


def _export_video(
    source: Path,
    target: Path,
    *,
    frame_indices: np.ndarray,
    camera: dict,
    fps: float,
    remap: bool,
) -> None:
    """Re-time the archive onto the export grid, warping it on the way.

    One output frame per table row, taken from the archive frame that row
    points at. That is what makes ``frame_index`` in the exported table mean
    "row i is video frame i" — LeRobot's readers assume it, and the archive's
    own grid may be a different rate entirely.
    """
    import av

    processor = _frame_processor(camera) if remap else None

    with av.open(str(source)) as reader:
        decoded = [frame.to_ndarray(format="rgb24") for frame in reader.decode(video=0)]
    if not decoded:
        logger.warning("%s decoded to no frames", source)
        return

    with av.open(str(target), mode="w") as writer:
        stream = writer.add_stream("libx264", rate=Fraction(fps).limit_denominator(10000))
        first = processor(decoded[0]) if processor else decoded[0]
        stream.height, stream.width = first.shape[:2]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "veryfast", "g": "30"}

        # Cache of one: consecutive rows overwhelmingly point at the same
        # archive frame, and the warp is the expensive part.
        last_index, last_frame = -1, None
        for index in frame_indices:
            index = int(np.clip(index, 0, len(decoded) - 1))
            if index != last_index:
                source_frame = decoded[index]
                last_frame = processor(source_frame) if processor else source_frame
                last_index = index
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(last_frame, format="rgb24")
            ):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)


def _frame_processor(camera: dict):
    """The warp and downscale the robot deliberately did not apply.

    Built from ``session.json``, so a dataset re-exported after a
    recalibration picks up the new geometry without the archive changing.
    """
    import cv2

    from solarian.core.optical_remap import load_policy_remap

    archive_wh = (int(camera["width"]), int(camera["height"]))
    resize = camera.get("observation_resize")

    optical = load_policy_remap(
        camera.get("optical_remap"), archive_wh, out_wh=_remap_wh(camera)
    )

    def process(frame: np.ndarray) -> np.ndarray:
        if optical is not None:
            # Warps straight to the target plane, so nothing to resize after.
            frame = optical.apply(frame)
        elif resize:
            # No calibration to honour, so the observation was the capture
            # downscaled, and that is what the export should reproduce.
            width, height = int(resize[0]), int(resize[1])
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(
                    frame, (width, height), interpolation=cv2.INTER_AREA
                )
        return np.ascontiguousarray(frame)

    return process


def _remap_wh(camera: dict) -> tuple[int, int] | None:
    """GoPro plane to warp into; ``None`` for the one the calibration used.

    Datasets recorded before the observation went pre-remap carry the plane
    under ``observation_resize``, which back then was applied after the warp.
    """
    size = camera.get("remap_size") or camera.get("observation_resize")
    return (int(size[0]), int(size[1])) if size else None


# -- metadata ----------------------------------------------------------------


def _info(
    *, meta, fps, n_episodes, n_frames, n_tasks, state_columns, action_columns,
    cameras, remap,
) -> dict:
    features: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(state_columns)],
            "names": state_columns,
        },
        "action": {
            "dtype": "float32",
            "shape": [len(action_columns)],
            "names": action_columns,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for name, camera in cameras.items():
        height, width = _exported_frame_hw(camera, remap)
        features[_video_key(name)] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": fps,
                "video.height": height,
                "video.width": width,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    return {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": meta["robot"]["class"],
        "total_episodes": n_episodes,
        "total_frames": n_frames,
        "total_tasks": n_tasks,
        "total_videos": n_episodes * len(cameras),
        "total_chunks": max((n_episodes - 1) // CHUNK_SIZE + 1, 1),
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
        # Not LeRobot's, but the session this came out of and the choices made
        # on the way: without them a re-export cannot be reproduced.
        "solarian": {
            "session": meta,
            "optical_remap_applied": remap,
        },
    }


def _exported_frame_hw(camera: dict, remap: bool) -> tuple[int, int]:
    if not remap:
        return int(camera["height"]), int(camera["width"])
    if camera.get("optical_remap"):
        wh = _remap_wh(camera)
        if wh:
            return wh[1], wh[0]
        from solarian.core.optical_remap import peek_remap_out_size

        width, height = peek_remap_out_size(camera["optical_remap"])
        return height, width
    resize = camera.get("observation_resize")
    if resize:
        return int(resize[1]), int(resize[0])
    return int(camera["height"]), int(camera["width"])


def _video_key(camera_name: str) -> str:
    """``arm.wrist`` -> ``observation.images.arm.wrist``, LeRobot's namespace."""
    return f"observation.images.{camera_name}"


def _episode_dirs(paths: SessionPaths, episodes: list[int] | None):
    """Sealed episode directories, in index order, as ``(index, path)``."""
    wanted = set(episodes) if episodes is not None else None
    found = []
    for directory in sorted(paths.episodes.iterdir()):
        if not directory.is_dir() or not directory.name.isdigit():
            continue
        index = int(directory.name)
        if wanted is not None and index not in wanted:
            continue
        # An episode without its metadata was never sealed, so its table has
        # no final row group and its mp4 has no footer.
        meta_path = directory / EPISODE_FILE
        if not meta_path.is_file():
            continue
        status = json.loads(meta_path.read_text()).get("status")
        if status == "video_failed":
            logger.warning("Skipping episode %d: its video never sealed", index)
            continue
        found.append((index, directory))
    return found


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
