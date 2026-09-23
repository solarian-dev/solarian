"""Owns an episode: the table, the encoders' boundaries, and the ledger entry.

Lives in the robot's main process. One row is written whenever any module
publishes a new sample, so the table samples at the union of every module's
rate rather than at one grid chosen now — an arm at 90 Hz and a gripper at
120 Hz both keep their own cadence, and nothing is resampled until export,
when the target rate is finally known. Signals that did not advance carry
their previous value forward; the per-module ``ts`` column is what tells a
consumer how stale each one was.

Video never passes through here. The camera buses feed the encoder processes
directly at full resolution, and the table records only which grid slot each
capture landed in, derived from the same ``(start_time, fps)`` the encoder
uses.
"""

from __future__ import annotations

import logging
import math
import shutil
import threading
import time

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from solarian.core.action import Action, ActionType
from solarian.core.clock_sync import synced_time
from solarian.core.observation import Observation
from solarian.core.recording.schema import (
    NUMPY_BY_DTYPE,
    RecordingSchema,
    build_schema,
)
from solarian.core.recording.session import (
    TABLE_FILE,
    EpisodeMeta,
    SessionMeta,
    SessionPaths,
    append_manifest,
    directory_size,
)
from solarian.core.recording.timeline import frame_index_for
from solarian.core.recording.video_writer import VideoWriter

logger = logging.getLogger(__name__)

#: Rows buffered before a parquet row group is flushed. At ~240 rows/s this is
#: about 17 seconds, which bounds memory on a long episode and is a sensible
#: row-group size for a reader that wants to seek.
_ROWS_PER_CHUNK = 4096

_ARROW_BY_DTYPE = {
    "float64": pa.float64(),
    "float32": pa.float32(),
    "int32": pa.int32(),
    "bool": pa.bool_(),
}

#: What an unset column holds, per numpy kind. NaN says "never sampled" for a
#: float; ``-1`` is the equivalent for a frame index, since 0 is a real frame.
_UNSET = {"f": math.nan, "i": -1, "b": False}


class SessionRecorder:
    """One recording session: many episodes under one root, one schema."""

    def __init__(
        self,
        *,
        root: Path,
        session: str,
        fps: float,
        writers: dict[str, VideoWriter],
        robot_name: str,
        cameras: dict,
        video_options: dict,
        task: str | None = None,
        max_session_bytes: int | None = None,
        schema: RecordingSchema | None = None,
    ):
        self.fps = float(fps)
        self.writers = writers
        self.schema = schema if schema is not None else build_schema()
        self.task = task
        self.max_session_bytes = max_session_bytes

        self._root = Path(root)
        self._robot_name = robot_name
        self._cameras = cameras
        self._video_options = video_options

        self._lock = threading.Lock()
        self._recording = False
        self._episode_task: str | None = None
        self._episode_skill: str | None = None
        self._start_ts = 0.0
        self._writer: pq.ParquetWriter | None = None

        # Field name -> column name, per module, so a row is assembled by
        # walking what the observation actually carries instead of probing the
        # schema for fields this robot may compute elsewhere or not at all.
        self._obs_columns = {
            module: {name: f"obs.{module}.{name}" for name in fields}
            for module, fields in self.schema.obs_fields
        }
        self._act_columns = {
            module: {name: f"act.{module}.{name}" for name in fields}
            for module, fields in self.schema.act_fields
        }
        self._cameras_by_module: dict[str, dict] = {}
        for camera in self.schema.cameras:
            self._cameras_by_module.setdefault(camera.module, {})[camera.field] = camera

        self._arrow_schema = pa.schema(
            [(c.name, _ARROW_BY_DTYPE[c.dtype]) for c in self.schema.columns]
        )
        self._chunk = {
            c.name: np.empty(_ROWS_PER_CHUNK, dtype=NUMPY_BY_DTYPE[c.dtype])
            for c in self.schema.columns
        }
        self._unset = {
            name: _UNSET[buffer.dtype.kind] for name, buffer in self._chunk.items()
        }
        self._stat_columns = self.schema.numeric_columns()

        #: Most recent value of every column, so a row is complete even though
        #: only some modules advanced since the previous one.
        self._latest: dict[str, float] = {}
        self._chunk_rows = 0
        self._total_rows = 0
        self._stats: dict[str, dict] = {}
        self._module_updates: dict[str, int] = {}

        self._session = ""
        self.open_session(session)

    # -- session -------------------------------------------------------------

    def open_session(self, session: str) -> None:
        """Point recording at a fresh directory named for *session*.

        A recording run is one session, and the run starts from the
        controller — long after the robot booted and built this. Rolling here
        is what lets one running robot record ``cup_stacking`` and then
        ``plate_stacking`` without a restart.

        A timestamp is appended, so a rerun never mixes its episodes in with a
        previous one's: the schema and the calibration in ``session.json``
        describe the episodes beside them, and appending to a directory
        written by an older build would make that description a lie.
        """
        if session == self._session:
            return
        if self._recording:
            self.end_episode(status="interrupted")

        self._session = session
        stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        self.paths = SessionPaths.create(self._root / f"{session}__{stamp}")
        self._episode_index = self.paths.next_episode_index()
        self._session_bytes = 0
        self._over_budget = False

        SessionMeta(
            robot=self._robot_name,
            schema=self.schema,
            cameras=self._cameras,
            video=self._video_options,
            fps=self.fps,
            task=self.task,
        ).write(self.paths)
        logger.info("Recording to %s", self.paths.root)

    # -- state ---------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def episode_count(self) -> int:
        """Episodes sealed so far, which is also the next episode's index."""
        return self._episode_index

    # -- episode lifecycle ---------------------------------------------------

    def start_episode(self, task: str | None = None, skill: str | None = None) -> int:
        """Open an episode and hand every encoder the same grid origin."""
        if self._recording:
            logger.warning("Episode %d is already recording", self._episode_index)
            return self._episode_index
        if self._over_budget:
            logger.error("Session is over its size budget; not starting an episode")
            return -1

        index = self._episode_index
        (self.paths.episode_dir(index) / "video").mkdir(parents=True, exist_ok=True)

        self._episode_task = task if task is not None else self.task
        self._episode_skill = skill
        self._latest = {}
        self._chunk_rows = 0
        self._total_rows = 0
        self._stats = {}
        self._module_updates = {}
        self._start_ts = synced_time()
        self._writer = pq.ParquetWriter(
            self.paths.episode_dir(index) / TABLE_FILE,
            self._arrow_schema,
            compression="zstd",
        )
        for name, writer in self.writers.items():
            writer.start_episode(
                str(self.paths.video_path(index, name)), self._start_ts
            )

        with self._lock:
            self._recording = True
        logger.info("Recording episode %d (task=%r)", index, self._episode_task)
        return index

    def end_episode(self, status: str = "ok") -> EpisodeMeta | None:
        """Seal the table and every mp4, then append one ledger line."""
        meta = self._seal(status)
        if meta is None:
            return None

        episode_dir = self.paths.episode_dir(meta.index)
        meta.write(self.paths)
        n_bytes = directory_size(episode_dir)
        self._session_bytes += n_bytes
        append_manifest(
            self.paths,
            {
                "episode": meta.index,
                "sealed_at": time.time(),
                # meta's, not the argument's: sealing is where a camera that
                # failed to produce a video downgrades the take.
                "status": meta.status,
                "bytes": n_bytes,
                "n_rows": meta.n_rows,
                "duration": round(meta.duration, 3),
                "transfer": {"state": "pending"},
                "upload": {"state": "pending"},
            },
        )
        self._episode_index += 1
        logger.info(
            "Sealed episode %d: %d rows, %.1fs, %.1f MB",
            meta.index,
            meta.n_rows,
            meta.duration,
            n_bytes / 1e6,
        )
        self._check_budget()
        return meta

    def drop_episode(self) -> None:
        """Seal and delete, reusing the index. A bad take leaves nothing behind.

        Sealing first rather than killing the encoders mid-stream is what
        keeps the next episode clean: an encoder with a half-open mux would
        carry this take's tail into the next one's file.
        """
        meta = self._seal("dropped")
        if meta is None:
            return
        shutil.rmtree(self.paths.episode_dir(meta.index), ignore_errors=True)
        logger.info("Dropped episode %d", meta.index)

    def close(self) -> None:
        if self._recording:
            self.end_episode(status="interrupted")
        for writer in self.writers.values():
            writer.close()

    def _seal(self, status: str) -> EpisodeMeta | None:
        """Close the table and the encoders, and measure what came out.

        Recording is switched off before any of the blocking work, so the
        observation thread is never held behind an encoder flush.
        """
        with self._lock:
            if not self._recording:
                return None
            self._recording = False
            self._flush_chunk()
            total_rows = self._total_rows
            module_updates = dict(self._module_updates)
            stats = self._stats

        end_ts = synced_time()
        self._writer.close()
        self._writer = None

        n_frames: dict[str, int] = {}
        coverage: dict[str, float] = {}
        dropped: dict[str, int] = {}
        for name, writer in self.writers.items():
            video = writer.end_episode(end_ts)
            if video is None or not video.complete:
                # Nothing to report means nothing to trust: the mp4 is short or
                # absent. Only "ok" is downgraded — a take the operator already
                # dropped or interrupted keeps the reason it was dropped.
                logger.error(
                    "Episode %d: camera %s did not produce a usable video",
                    self._episode_index,
                    name,
                )
                status = "video_failed" if status == "ok" else status
                n_frames[name] = 0
                coverage[name] = 0.0
                continue
            n_frames[name] = video.n_frames
            coverage[name] = round(video.coverage, 4)
            dropped[name] = video.n_dropped
            if video.n_dropped:
                logger.warning(
                    "Episode %d: camera %s lost %d frames to a full encoder queue",
                    self._episode_index,
                    name,
                    video.n_dropped,
                )

        duration = max(end_ts - self._start_ts, 1e-9)
        return EpisodeMeta(
            index=self._episode_index,
            start_ts=self._start_ts,
            end_ts=end_ts,
            n_rows=total_rows,
            status=status,
            task=self._episode_task,
            skill=self._episode_skill,
            n_frames=n_frames,
            video_coverage=coverage,
            dropped_frames=dropped,
            measured_hz={
                module: round(count / duration, 1)
                for module, count in module_updates.items()
            },
            stats=stats,
        )

    # -- row assembly --------------------------------------------------------

    def note_action(self, action: Action) -> None:
        """Remember the action the robot actually received.

        The robot's view is the one worth training on: the action path is
        conflated end to end, so a command the controller sent but the robot
        never received moved nothing and teaches a mapping that never happened.
        """
        if action.action_type != ActionType.MODULE_ACTION:
            return
        with self._lock:
            if not self._recording:
                return
            for module_name, module_action in action.items():
                columns = self._act_columns.get(module_name)
                if columns is None:
                    continue
                for field_name, data in module_action.items():
                    column = columns.get(field_name)
                    if column is not None:
                        self._latest[column] = _as_float(data.value)

    def write_row(self, observation: Observation) -> None:
        """Append one row for whichever modules just published.

        Called from the robot's observation thread, which already sees only
        the modules whose buffers advanced — exactly the event this table is
        keyed on.
        """
        with self._lock:
            if not self._recording:
                return

            wrote_module = False
            for module_name, module_obs in observation.items():
                columns = self._obs_columns.get(module_name)
                if columns is None:
                    continue
                wrote_module = True
                self._module_updates[module_name] = (
                    self._module_updates.get(module_name, 0) + 1
                )
                cameras = self._cameras_by_module.get(module_name, {})
                latest_ts = -math.inf
                for field_name, data in module_obs.items():
                    latest_ts = max(latest_ts, data.timestamp)
                    camera = cameras.get(field_name)
                    if camera is not None:
                        # The encoder derived the same index from the same
                        # stamp, so no message has to carry it across.
                        self._latest[camera.capture_ts_column] = data.timestamp
                        self._latest[camera.frame_index_column] = frame_index_for(
                            data.timestamp, self._start_ts, self.fps
                        )
                        continue
                    column = columns.get(field_name)
                    if column is not None:
                        self._latest[column] = _as_float(data.value)
                if latest_ts > -math.inf:
                    self._latest[f"obs.{module_name}.ts"] = latest_ts

            if not wrote_module:
                return

            now = synced_time()
            self._latest["wall_ts"] = now
            self._latest["t"] = now - self._start_ts

            row = self._chunk_rows
            latest = self._latest
            for name, buffer in self._chunk.items():
                buffer[row] = latest.get(name, self._unset[name])
            self._chunk_rows += 1
            self._total_rows += 1

            if self._chunk_rows >= _ROWS_PER_CHUNK:
                self._flush_chunk()

    # -- internals -----------------------------------------------------------

    def _flush_chunk(self) -> None:
        """Write the buffered rows as one row group and fold them into stats."""
        if self._chunk_rows == 0 or self._writer is None:
            return
        n = self._chunk_rows
        self._writer.write_table(
            pa.table(
                {name: buffer[:n] for name, buffer in self._chunk.items()},
                schema=self._arrow_schema,
            )
        )

        for name in self._stat_columns:
            values = self._chunk[name][:n]
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            entry = self._stats.setdefault(
                name,
                {"min": math.inf, "max": -math.inf, "sum": 0.0, "sumsq": 0.0,
                 "count": 0},
            )
            entry["min"] = min(entry["min"], float(finite.min()))
            entry["max"] = max(entry["max"], float(finite.max()))
            entry["sum"] += float(finite.sum())
            entry["sumsq"] += float(np.square(finite, dtype=np.float64).sum())
            entry["count"] += int(finite.size)

        self._chunk_rows = 0

    def _check_budget(self) -> None:
        """Stop before an SD card fills, because recording is on by default."""
        if self.max_session_bytes is None or self._over_budget:
            return
        if self._session_bytes < self.max_session_bytes:
            return
        self._over_budget = True
        logger.error(
            "Session reached its %.1f GB budget (%.1f GB written); recording is off",
            self.max_session_bytes / 1e9,
            self._session_bytes / 1e9,
        )


def _as_float(value) -> float:
    """Collapse a ``Data`` payload to one number.

    Vector-valued fields are not tabulated — a column holds a scalar, and the
    schema already left them out — so anything unexpected lands as NaN rather
    than taking the observation thread down.
    """
    if isinstance(value, (bool, int, float, np.generic)):
        return float(value)
    if isinstance(value, np.ndarray) and value.size == 1:
        return float(value.reshape(-1)[0])
    return math.nan
