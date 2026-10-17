"""Left-right mirroring, the math of UMI's ``zarr_mirror.py``.

Pick a world axis to reflect across (default x, the mirror plane being y-z)
and let M flip it. Positions become ``M @ pos``; a rotation R becomes
``M R M^-1``, which as an axis-angle vector v is ``-M v`` (since
``M [w]_x M^-1 = [-M w]_x`` for a reflection), i.e. every component but the
mirror axis negated. The default suits the UMI tabletop sessions, whose world
stores "up" along y with the gripper handle roughly along -y; mirroring y
would turn the handle upside down.
"""

from __future__ import annotations

import json
import re
from typing import Iterable, Literal

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, Field

from solarian.core.recording.archive import TABLE_FILE, column_stats, table_bytes, targets_bytes
from solarian.datasets.op import DatasetOp, Episode, EpisodeOut, OpContext, OpKind
from solarian.datasets.video import VideoEncoder, decode, video_fps

AXES = {"x": 0, "y": 1, "z": 2}
POSE_COLUMN = re.compile(r"^(obs|act)\.(robot\d+)\.(x|y|z|rx|ry|rz)$")
#: Unchanged by a left-right mirror.
INVARIANT_COLUMN = re.compile(r"^(t|wall_ts|.+\.ts|.+\.frame_index|.+\.capture_ts|.+\.gripper_width)$")


def reflect_pos(pos: np.ndarray, axis: int) -> np.ndarray:
    out = pos.copy()
    out[..., axis] = -out[..., axis]
    return out


def reflect_rotvec(rotvec: np.ndarray, axis: int) -> np.ndarray:
    out = -rotvec.copy()
    out[..., axis] = -out[..., axis]
    return out


def reflect_pose6(pose: np.ndarray, axis: int) -> np.ndarray:
    """A 6D pose ``[x, y, z, rx, ry, rz]``."""
    out = pose.copy()
    out[..., :3] = reflect_pos(pose[..., :3], axis)
    out[..., 3:] = reflect_rotvec(pose[..., 3:], axis)
    return out


def mirror_table(table: pa.Table, axis: int) -> tuple[pa.Table, list[str]]:
    """``table`` with every robot pose reflected, and the columns copied though unknown."""
    columns = {name: table.column(name) for name in table.column_names}
    prefixes = sorted({m.group(1, 2) for m in map(POSE_COLUMN.match, table.column_names) if m})
    for kind, module in prefixes:
        for fields, reflect in ((("x", "y", "z"), reflect_pos), (("rx", "ry", "rz"), reflect_rotvec)):
            names = [f"{kind}.{module}.{f}" for f in fields]
            if not all(n in columns for n in names):
                continue
            values = reflect(np.stack([columns[n].to_numpy() for n in names], axis=-1), axis)
            for k, name in enumerate(names):
                columns[name] = pa.array(values[:, k], type=table.schema.field(name).type)
    unknown = [n for n in table.column_names if not POSE_COLUMN.match(n) and not INVARIANT_COLUMN.match(n)]
    return pa.table(columns), unknown


def mirror_targets(targets: dict, width: int) -> dict:
    """bbox ``x0, x1 -> W - x1, W - x0`` (x1 exclusive, as cv2.boundingRect gives it); mask flipped."""
    out = dict(targets)
    if "bbox" in targets:
        bbox = targets["bbox"].copy()
        # A negative coordinate means "no box" and is kept.
        valid = (bbox[:, :4] >= 0).all(axis=1)
        x0 = bbox[valid, 0].copy()
        bbox[valid, 0] = width - bbox[valid, 2]
        bbox[valid, 2] = width - x0
        out["bbox"] = bbox
    if "mask" in targets:
        mask = np.unpackbits(targets["mask"], axis=-1, count=int(targets["mask_width"]))
        out["mask"] = np.packbits(mask[..., ::-1], axis=-1)
    return out


def mirror_tags(frames: list[dict], width: int) -> list[dict]:
    """Corner ``x -> W - 1 - x``: corners are pixel-centre coordinates."""
    return [{tag: [[width - 1 - x, y] for x, y in corners] for tag, corners in frame.items()} for frame in frames]


class MirrorParams(BaseModel):
    axis: Literal["x", "y", "z"] = Field(default="x", description="World axis to flip; the mirror plane is orthogonal to it.")
    keep_original: bool = Field(default=True, description="Follow each episode by its mirror; off writes only the mirrors.")


class MirrorOp(DatasetOp):
    name = "mirror"
    title = "Mirror left-right"
    description = """
    Augment by mirroring: every episode is followed by its left-right mirror
    (original 0, mirror 0, original 1, ...). Videos are flipped and re-encoded,
    robot poses reflected across the plane orthogonal to the axis, tags,
    targets and umi demo poses mirrored to match. The originals are not copied.
    """
    kind = OpKind.EXPAND
    Params = MirrorParams

    def expand(self, episode: Episode, params: MirrorParams, ctx: OpContext) -> Iterable[tuple[str, EpisodeOut]]:
        axis = AXES[params.axis]
        widths = {name: int(cam["width"]) for name, cam in ctx.session.meta["cameras"].items()}
        files = {}
        for camera in widths:
            member = episode.video(camera)
            encoder = VideoEncoder(ctx.scratch / f"{episode.index:06d}.mirror.{camera}.mp4",
                                   video_fps(ctx.session, member), ctx.session.meta.get("video"))
            for rgb in decode(ctx.session, member):
                encoder.write(np.ascontiguousarray(rgb[:, ::-1]))
            files[f"video/{camera}.mp4"] = encoder.close()

        table, unknown = mirror_table(episode.table(), axis)
        if unknown:
            ctx.log(f"WARNING episode {episode.index}: columns {unknown} are not mirrored; copied as they are")
        files[TABLE_FILE] = table_bytes(table)
        for rel in episode.members():
            if rel.startswith("tags/"):
                camera = rel[len("tags/"):-len(".json")]
                files[rel] = json.dumps(mirror_tags(episode.read_json(rel), widths[camera])).encode()
            elif rel.startswith("targets/"):
                camera = rel[len("targets/"):-len(".npz")]
                files[rel] = targets_bytes(**mirror_targets(episode.targets(camera), widths[camera]))
            elif rel not in files:
                ctx.log(f"WARNING episode {episode.index}: {rel} is not mirrored; copied as it is")

        meta = episode.meta
        mirrored = {**meta, "stats": column_stats(table, meta.get("stats", {})), "mirror": {"axis": params.axis}}
        if "umi" in meta:
            mirrored["umi"] = {
                module: {key: reflect_pose6(np.asarray(value, dtype=np.float64), axis).tolist()
                         if key in ("demo_start_pose", "demo_end_pose") else value
                         for key, value in poses.items()}
                for module, poses in meta["umi"].items()}
        if params.keep_original:
            yield "out", episode.derive()
        yield "out", EpisodeOut(meta=mirrored, files=files, source=episode)
