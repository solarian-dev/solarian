"""Episode recording: archive video, one table per episode, and the ledger.

Either side of the link can record, into the same layout. On the robot, a
camera bus fans each captured frame out to a
:class:`~solarian.core.recording.video_writer.VideoWriter` process before the
observation path downscales and remaps it, so the archive keeps the resolution
and the geometry the sensor produced rather than the ones today's policy
happens to want. Everything else a session needs is written by
:class:`~solarian.core.recording.recorder.SessionRecorder` in the robot's main
process. On the controller,
:class:`~solarian.core.recording.controller_recorder.ControllerRecorder` records
what arrived there instead.
"""

from solarian.core.recording.controller_recorder import ControllerRecorder
from solarian.core.recording.export import ExportStats, export_lerobot
from solarian.core.recording.recorder import SessionRecorder
from solarian.core.recording.schema import CameraKey, RecordingSchema, build_schema
from solarian.core.recording.session import (
    CameraMeta,
    EpisodeMeta,
    SessionMeta,
    SessionPaths,
    append_manifest,
    read_manifest,
)
from solarian.core.recording.structure import build_structure
from solarian.core.recording.timeline import frame_index_for
from solarian.core.recording.video_writer import ArchiveSink, VideoWriter

__all__ = [
    "ArchiveSink",
    "CameraKey",
    "CameraMeta",
    "ControllerRecorder",
    "EpisodeMeta",
    "ExportStats",
    "RecordingSchema",
    "SessionMeta",
    "SessionPaths",
    "SessionRecorder",
    "VideoWriter",
    "append_manifest",
    "build_schema",
    "build_structure",
    "export_lerobot",
    "frame_index_for",
    "read_manifest",
]
