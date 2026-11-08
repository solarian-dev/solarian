"""The controller's own dataset: what reached it, in the robot's layout.

The robot's recording is the high-fidelity one: every capture at archive
resolution, every command that reached a module. This one is the other end of
the link. Its rows are the observations as they arrived — JPEG-decoded, at the
preview rate, after the network — and its actions are the ones the controller
sent, whether or not the robot ever received them. ``obs.<module>.ts`` is still
the robot's capture stamp while ``wall_ts`` is the controller's receive time, so
the latency of every row is in the table.

The session layout is the robot's, so the exporter and the viewer read either
one. Beside each episode it keeps the episode's
:class:`~solarian.core.recording.episode_log.EpisodeLog` under
``episodes/<index>/logs/``, with the tracker overlays drawn over each frame.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Mapping, Sequence
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import cv2
import numpy as np

from solarian.core.action import ActionType
from solarian.core.data import MaskData, PointData
from solarian.core.observation import Observation
from solarian.core.recording.episode_log import LOGS_DIR, EpisodeLog
from solarian.core.recording.recorder import SessionRecorder
from solarian.core.recording.schema import CameraKey, build_schema
from solarian.core.recording.session import CameraMeta
from solarian.core.recording.video_writer import ArchiveSink, VideoWriter

logger = logging.getLogger(__name__)


class ControllerRecorder:
    """One :class:`SessionRecorder` fed from the controller's side of the link.

    Episode boundaries block: the first start spawns the encoders, and a seal
    waits for them to drain. The controller calls them from a thread of its
    own, never from the UI or control loop. :meth:`feed` and :meth:`note_sent`
    are cheap and safe from any thread.
    """

    def __init__(
        self,
        *,
        root: Path,
        cfg,
        fps: float,
        task: str | None,
        session: str,
        shm_manager: SharedMemoryManager,
        overlays: Mapping[tuple[str, str], Sequence[str]] | None = None,
    ):
        """
        Args:
            cfg: An :class:`~solarian.core.configs.ArchiveConfig` — encoder
                settings and the session size budget.
            fps: Grid the video is encoded onto.
            session: Name episodes are filed under until a caller names one.
            overlays: ``(module, image field)`` to the box, point and mask
                fields drawn over it, logged with each frame they mark.
        """
        self._root = Path(root)
        self._cfg = cfg
        self._fps = float(fps)
        self._task = task
        self._session_name = session
        self._shm_manager = shm_manager
        self._overlays = dict(overlays or {})
        self.schema = build_schema()

        #: Built at the first episode, once frames have shown what size the
        #: cameras arrive at.
        self._session: SessionRecorder | None = None
        self._sinks: dict[str, tuple[CameraKey, ArchiveSink, tuple[int, int]]] = {}
        #: The log of the latest episode, kept after its seal so what happens
        #: between takes is still written down; ``None`` once it is dropped.
        self.log: EpisodeLog | None = None
        #: Capture time of the last frame whose overlays were logged, per camera.
        self._overlay_ts: dict[str, float] = {}

    # -- state ---------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._session is not None and self._session.is_recording

    @property
    def session_dir_name(self) -> str | None:
        return self._session.session_dir_name if self._session is not None else None

    @property
    def episode_index(self) -> int:
        """The episode recording now, or the next one between takes."""
        return self._session.episode_count if self._session is not None else 0

    @property
    def episode_dir(self) -> Path | None:
        """The open episode's directory; ``None`` between takes."""
        if not self.is_recording:
            return None
        return self._session.paths.episode_dir(self._session.episode_count)

    # -- episode lifecycle ---------------------------------------------------

    def start(
        self,
        latest: Observation,
        *,
        session: str | None = None,
        task: str | None = None,
        skill: str | None = None,
        dir_name: str | None = None,
        index: int | None = None,
    ) -> int:
        """Open an episode; returns its index, or ``-1``.

        Args:
            latest: The newest observation per module, which sizes each
                camera's encoder on the first episode.
            session: Name to file it under. ``None`` keeps the current one.
            dir_name, index: The robot's session directory and episode index,
                adopted verbatim so the two recordings line up.
        """
        if session is not None:
            self._session_name = session
        if self._session is None:
            self._session = self._build(latest)
        self._session.open_session(self._session_name, dir_name=dir_name)
        task = task if task is not None else self._task
        index = self._session.start_episode(task=task, skill=skill, index=index)
        if index >= 0:
            self.log = EpisodeLog(self._session.paths.episode_dir(index) / LOGS_DIR)
            self.log.event("episode_started", episode=index, task=task)
        return index

    def stop(self) -> None:
        if self._session is not None:
            if self.is_recording and self.log is not None:
                self.log.event("episode_ended", episode=self._session.episode_count)
            self._session.end_episode()

    def drop(self) -> None:
        if self._session is not None:
            if self.is_recording:
                self.log = None
            self._session.drop_episode()

    def close(self) -> None:
        if self._session is not None:
            self._session.close()

    # -- data ----------------------------------------------------------------

    def feed(self, observation: Observation) -> None:
        """Record one received observation: its frames, then its row."""
        if not self.is_recording:
            return
        for camera, sink, (height, width) in self._sinks.values():
            module_obs = observation[camera.module]
            data = module_obs.get(camera.field) if module_obs is not None else None
            if data is None or not sink.accepting:
                continue
            image = data.value
            if not isinstance(image, np.ndarray) or image.ndim != 3:
                continue
            self._log_overlays(camera, module_obs, image.shape[:2], float(data.timestamp))
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            sink.write_frame(image, float(data.timestamp))
        self._session.write_row(observation)

    def _log_overlays(self, camera: CameraKey, module_obs, size: tuple[int, int], ts: float) -> None:
        """Log what the overlay fields mark on this frame, once per frame and only when they mark something.

        Boxes and points are in the pixels of the frame as it arrived, ``size``,
        which the video may have been scaled from.
        """
        fields = self._overlays.get((camera.module, camera.field))
        log = self.log
        if not fields or log is None or self._overlay_ts.get(camera.name) == ts:
            return
        self._overlay_ts[camera.name] = ts
        row = {}
        for name in fields:
            overlay = module_obs.get(name)
            value = overlay.value if overlay is not None else None
            if value is None or not np.any(value):
                continue
            if isinstance(overlay, MaskData):
                if np.shape(value) != size:
                    continue
                ok, png = cv2.imencode(".png", np.asarray(value, dtype=np.uint8))
                if ok:
                    row[name] = {"mask": base64.b64encode(png.tobytes()).decode("ascii")}
            elif isinstance(overlay, PointData):
                row[name] = {"point": np.asarray(value, dtype=float).round(1).tolist()}
            else:
                row[name] = {"box": np.asarray(overlay.overlay_box()).tolist()}
        if row:
            log.overlay(camera.name, {"ts": ts, "size": list(size), **row})

    def note_sent(self, action) -> None:
        """Tabulate a ``MODULE_ACTION`` the controller put on the link."""
        if not self.is_recording or int(action.action_type) != int(ActionType.MODULE_ACTION):
            return
        for module_name, module_action in action.items():
            self._session.note_module_action(module_name, module_action)

    # -- internals -----------------------------------------------------------

    def _build(self, latest: Observation) -> SessionRecorder:
        """One encoder per camera, sized from the last frame that arrived.

        The controller is never told a camera's resolution, only shown its
        frames. A camera that has not sent one yet has nothing to size an
        encoder from, and is left out of this recorder.
        """
        cfg = self._cfg
        writers: dict[str, VideoWriter] = {}
        cameras: dict[str, CameraMeta] = {}
        for camera in self.schema.cameras:
            module_obs = latest[camera.module]
            data = module_obs.get(camera.field) if module_obs is not None else None
            image = data.value if data is not None else None
            if not isinstance(image, np.ndarray) or image.ndim != 3:
                logger.warning(
                    "No frame from %s has reached the controller yet; the "
                    "controller recording has no video for it",
                    camera.name,
                )
                continue
            height, width = image.shape[:2]
            writer = VideoWriter(
                name=camera.name,
                fps=self._fps,
                shape=(height, width, 3),
                codec=cfg.codec,
                crf=cfg.crf,
                keyint=cfg.keyint,
                preset=cfg.preset,
                threads=cfg.threads,
                x264_options=cfg.x264_options,
                input_pix_fmt="rgb24",
                buffer_size=cfg.queue_size,
            )
            self._sinks[camera.name] = (camera, writer.open(self._shm_manager), (height, width))
            writers[camera.name] = writer
            cameras[camera.name] = CameraMeta(
                width=width, height=height, color_mode="rgb", source_fps=self._fps
            )

        return SessionRecorder(
            root=self._root,
            session=self._session_name,
            fps=self._fps,
            writers=writers,
            robot=None,
            cameras=cameras,
            video_options={
                "codec": cfg.codec,
                "crf": cfg.crf,
                "keyint": cfg.keyint,
                "preset": cfg.preset,
                "pix_fmt": "yuv420p",
            },
            task=self._task,
            max_session_bytes=(
                int(cfg.max_session_gb * 1e9) if cfg.max_session_gb else None
            ),
            schema=self.schema,
            source="controller",
        )
