from __future__ import annotations

import json
from typing import Iterable, Literal

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, Field, field_validator

from solarian.core.recording.archive import TABLE_FILE, column_stats, table_bytes, targets_bytes
from solarian.datasets.op import DatasetOp, Episode, EpisodeOut, OpContext, OpKind
from solarian.datasets.video import VideoEncoder, decode, video_fps

HALVES = ("pick", "place")
POSE_FIELDS = ("x", "y", "z", "rx", "ry", "rz")


def median_filter_1d(y: np.ndarray, size: int) -> np.ndarray:
    """Centered median filter. Even ``size`` is rounded up so the window does not shift the valley."""
    y = np.asarray(y, dtype=np.float64)
    size = int(size)
    if size <= 1 or y.size == 0:
        return y
    if size % 2 == 0:
        size += 1
    if size > y.size:
        size = int(y.size) if int(y.size) % 2 == 1 else max(int(y.size) - 1, 1)
    if size <= 1:
        return y
    pad = size // 2
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(y, pad, mode="edge"), size)
    return np.median(windows, axis=-1)


def find_split(y: np.ndarray, margin_frac: float, tol: float, smooth: int) -> tuple[int, float]:
    """Frame of the return-to-home, and how far that valley sits above the episode minimum.

    The episode starts at home too, so a global argmin lands on the opening
    frames. Search the central window, take every frame within ``tol`` of
    that window's minimum, and keep the one closest to the temporal midpoint.
    Two dwells at the same height (the opening tail still inside the window,
    and the return-to-home) therefore resolve toward the middle.
    """
    y_s = median_filter_1d(y, smooth)
    n = int(y_s.shape[0])
    if n == 0:
        raise ValueError("empty episode")
    if n == 1:
        return 0, 0.0
    lo = min(max(int(np.floor(margin_frac * n)), 0), n - 1)
    hi = min(max(int(np.ceil((1.0 - margin_frac) * n)) - 1, 0), n - 1)
    if hi < lo:
        lo, hi = 0, n - 1
    window = y_s[lo:hi + 1]
    min_val = float(window.min())
    candidates = np.flatnonzero(window <= min_val + tol) + lo
    mid = 0.5 * (n - 1)
    split = int(candidates[np.argmin(np.abs(candidates.astype(np.float64) - mid))])
    return split, float(min_val - float(y_s.min()))


class SplitParams(BaseModel):
    module: str = Field(default="robot0", description="Module whose end-effector pose is searched.")
    axis: Literal["x", "y", "z"] = Field(default="y", description="Axis whose minimum is home.")
    tol: float = Field(default=0.01, ge=0, description="Meters. Frames this close to the window minimum belong to the valley.")
    margin_frac: float = Field(default=0.15, ge=0, lt=0.5, description="Fraction of the episode skipped at each end before searching.")
    smooth: int = Field(default=11, ge=1, description="Median-filter window in frames; even values are rounded up.")
    min_frames: int = Field(default=8, ge=1, description="Warn when either half is shorter than this.")
    strict: bool = Field(default=False, description="Fail instead of warning.")

    @field_validator("module")
    @classmethod
    def _module(cls, value: str) -> str:
        if not value or "." in value:
            raise ValueError("a module name, e.g. robot0")
        return value


def _half_table(table: pa.Table, start: int, length: int) -> pa.Table:
    """Rows ``[start, start + length)``, with ``t`` and ``*.frame_index`` restarting at 0."""
    part = table.slice(start, length)
    for i, name in enumerate(part.column_names):
        if name == "t" or name.endswith(".frame_index"):
            values = part.column(name).to_numpy()
            part = part.set_column(i, name, pa.array(values - values[0], type=part.schema.field(name).type))
    return part


def _half_meta(meta: dict, table: pa.Table, start: int, length: int, log) -> dict:
    """episode.json of a half: its rows, times, frame counts, stats and umi demo poses."""
    dt = (meta["end_ts"] - meta["start_ts"]) / meta["n_rows"]
    out = {
        **meta,
        "n_rows": length,
        "start_ts": meta["start_ts"] + start * dt,
        "end_ts": meta["start_ts"] + (start + length) * dt,
        "n_frames": {camera: length for camera in meta.get("n_frames", {})},
        "stats": column_stats(table, meta.get("stats", {})),
    }
    if "umi" in meta:
        umi = {}
        for module, poses in meta["umi"].items():
            columns = [f"obs.{module}.{f}" for f in POSE_FIELDS]
            if not all(c in table.column_names for c in columns):
                log(f"episode {meta['index']}: no {module} pose columns; keeping its demo poses")
                umi[module] = poses
                continue
            pose = np.stack([table.column(c).to_numpy() for c in columns], axis=-1).astype(np.float64)
            # The policy's *_wrt_start is taken against demo_start_pose.
            umi[module] = {**poses, "demo_start_pose": pose[0].tolist(), "demo_end_pose": pose[-1].tolist()}
        out["umi"] = umi
    return out


class SplitPickPlaceOp(DatasetOp):
    name = "split-pick-place"
    title = "Split into pick and place"
    description = """
    Cut every episode at the return-to-home between its two motions: the minimum
    of the end-effector's height axis nearest the temporal midpoint. Each episode
    becomes one episode of the pick output and one of the place output; the home
    frame is in both. Videos are re-encoded per half, tags and targets sliced,
    and each half's umi demo poses and stats recomputed.
    """
    kind = OpKind.EXPAND
    Params = SplitParams
    outputs = HALVES

    def setup(self, params: SplitParams, ctx: OpContext) -> None:
        column = f"obs.{params.module}.{params.axis}"
        if column not in ctx.session.meta.get("schema", {}):
            raise ValueError(f"{ctx.session.path.name} has no {column} column to split on")

    def expand(self, episode: Episode, params: SplitParams, ctx: OpContext) -> Iterable[tuple[str, EpisodeOut]]:
        table = episode.table()
        n = table.num_rows
        column = f"obs.{params.module}.{params.axis}"
        split, delta = find_split(table.column(column).to_numpy().astype(np.float64),
                                  params.margin_frac, params.tol, params.smooth)
        ranges = {"pick": (0, split + 1), "place": (split, n - split)}
        ctx.log(f"episode {episode.index}: split {split}/{n} ({100.0 * split / n:.1f}%), "
                f"{params.axis} valley {delta * 1000:.1f} mm above the minimum")
        warnings = []
        if delta > params.tol:
            warnings.append(f"valley is {delta * 1000:.1f} mm above the episode minimum")
        if split + 1 < params.min_frames or n - split < params.min_frames:
            warnings.append(f"short half (pick {split + 1} frames, place {n - split}, minimum {params.min_frames})")
        for warning in warnings:
            if params.strict:
                raise ValueError(f"episode {episode.index}: {warning}")
            ctx.log(f"WARNING episode {episode.index}: {warning}")

        files = {half: {} for half in HALVES}
        for camera in ctx.session.cameras:
            member = episode.video(camera)
            fps = video_fps(ctx.session, member)
            encoders = {half: VideoEncoder(ctx.scratch / f"{episode.index:06d}.{half}.{camera}.mp4", fps,
                                           ctx.session.meta.get("video")) for half in HALVES}
            for k, rgb in enumerate(decode(ctx.session, member, n)):
                for half, (start, length) in ranges.items():
                    if start <= k < start + length:
                        encoders[half].write(rgb)
            for half, encoder in encoders.items():
                files[half][f"video/{camera}.mp4"] = encoder.close()

        for rel in episode.members():
            if rel == TABLE_FILE or rel.startswith("video/"):
                continue
            for half, (start, length) in ranges.items():
                if rel.startswith("tags/"):
                    files[half][rel] = json.dumps(episode.read_json(rel)[start:start + length]).encode()
                elif rel.startswith("targets/"):
                    targets = episode.session.read_targets(episode.index, rel[len("targets/"):-len(".npz")])
                    files[half][rel] = targets_bytes(**{
                        key: value if value.ndim == 0 else value[start:start + length]
                        for key, value in targets.items()})
            if not rel.startswith(("tags/", "targets/")):
                ctx.log(f"WARNING episode {episode.index}: {rel} is not split; leaving it out")

        for half, (start, length) in ranges.items():
            part = _half_table(table, start, length)
            files[half][TABLE_FILE] = table_bytes(part)
            yield half, EpisodeOut(meta=_half_meta(episode.meta, part, start, length, ctx.log), files=files[half])
