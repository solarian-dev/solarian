"""One H.264 encoder process per camera.

The camera bus pushes every captured frame here and never waits: a full queue
drops the frame rather than stalling capture, which is the only acceptable
failure mode when the producer is a sensor feeding a control loop. Episode
boundaries arrive on a separate queue, so the camera side holds no episode
state — the encoder owns the constant-rate grid and decides which slot each
capture lands in.

Because the camera feeds this queue whether or not an episode is open, an
episode ends at a *timestamp*, never at the queue running dry: a queue fed at
30 fps and drained at 29 never runs dry, and an encoder waiting for that would
encode past the end of the take until something killed it.

Adapted from UMI's ``umi/real_world/video_recorder.py``.
"""

from __future__ import annotations

import enum
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass
from fractions import Fraction
from multiprocessing.managers import SharedMemoryManager
from queue import Empty, Full

import av
import numpy as np

from solarian.core.recording.timeline import repeats_for
from solarian.utils.shared_memory.shared_memory_queue import SharedMemoryQueue

logger = logging.getLogger(__name__)

#: How long ``end_episode`` waits for the encoder to drain its backlog and close
#: the mux. Generous because the flush encodes everything still queued.
_SEAL_TIMEOUT_S = 30.0

#: How often to check, mid-episode, whether the encoder is shedding frames. An
#: encoder that cannot keep up is a broken take, and the operator has to hear
#: about it while there is still time to stop, not once the episode is sealed.
_LAG_WARN_PERIOD_S = 5.0


class _Command(enum.IntEnum):
    START = 0
    STOP = 1
    SHUTDOWN = 2


class ArchiveSink:
    """The producer half of an encoder, and the only half a camera bus sees.

    Separate from :class:`VideoWriter` because the bus lives in the camera
    module's process: this object crosses that process boundary, and a started
    ``mp.Process`` cannot. It holds shared memory and a counter, both of which
    can.
    """

    def __init__(self, frames: SharedMemoryQueue, dropped):
        self._frames = frames
        self._dropped = dropped

    def write_frame(self, img: np.ndarray, frame_time: float) -> None:
        """Hand one capture to the encoder, or drop it.

        Called on every capture whether or not an episode is open; the encoder
        discards frames outside one. Dropping rather than blocking is
        deliberate — the alternative is a camera that stalls its module's loop
        because an encoder fell behind.
        """
        try:
            self._frames.put({"img": img, "frame_time": float(frame_time)})
        except Full:
            with self._dropped.get_lock():
                self._dropped.value += 1


@dataclass(frozen=True)
class EpisodeVideoStats:
    """What one sealed mp4 turned out to contain."""

    path: str
    #: Grid slots written, i.e. the mp4's frame count.
    n_frames: int
    #: Distinct captures behind those slots.
    n_unique: int
    #: Slots that repeated the previous capture because none had arrived.
    n_repeat: int
    #: Captures the camera offered while the queue was full.
    n_dropped: int
    #: False if the mux never closed cleanly, so the file is missing its tail.
    complete: bool = True

    @property
    def coverage(self) -> float:
        """Fraction of grid slots backed by a distinct capture."""
        return self.n_unique / self.n_frames if self.n_frames else 0.0


class VideoWriter(mp.Process):
    """Encodes one camera's archive stream, one mp4 per episode."""

    def __init__(
        self,
        *,
        name: str,
        fps: float,
        shape: tuple[int, int, int],
        codec: str = "libx264",
        crf: int = 20,
        keyint: int = 30,
        preset: str = "veryfast",
        input_pix_fmt: str = "rgb24",
        output_pix_fmt: str = "yuv420p",
        buffer_size: int = 64,
    ):
        super().__init__(name=f"VideoWriter-{name}", daemon=True)
        self.camera_name = name
        self.fps = float(fps)
        self.shape = tuple(shape)
        self.codec = codec
        self.crf = int(crf)
        self.keyint = int(keyint)
        self.preset = preset
        self.input_pix_fmt = input_pix_fmt
        self.output_pix_fmt = output_pix_fmt
        self.buffer_size = int(buffer_size)

        self._frames: SharedMemoryQueue | None = None
        self._commands: mp.Queue = mp.Queue()
        self._sealed: mp.Queue = mp.Queue()
        self._ready = mp.Event()
        #: Captures lost to a full queue. Shared because only the producer can
        #: see them happen and only the consumer reports them.
        self._dropped = mp.Value("i", 0)

    # -- control side, runs in the robot process ----------------------------

    def open(self, shm_manager: SharedMemoryManager) -> ArchiveSink:
        """Allocate the frame queue, start the encoder, return the producer half.

        The sink has to exist before the camera module forks, because that is
        how the module's bus comes to hold it.
        """
        self._frames = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples={
                "img": np.zeros(self.shape, dtype=np.uint8),
                "frame_time": 0.0,
            },
            buffer_size=self.buffer_size,
        )
        self.start()
        if not self._ready.wait(timeout=30.0):
            logger.warning("%s did not report ready; frames may be lost", self.name)
        return ArchiveSink(self._frames, self._dropped)

    def start_episode(self, video_path: str, start_time: float) -> None:
        with self._dropped.get_lock():
            self._dropped.value = 0
        self._commands.put((int(_Command.START), str(video_path), float(start_time)))

    def end_episode(self, stop_time: float) -> EpisodeVideoStats | None:
        """Close the mux and report what it contains.

        *stop_time* is the instant the episode ended, on the same clock the
        camera stamps captures with. Frames taken after it belong to no
        episode, and the encoder stops at the first one it sees.

        Blocks until the encoder has flushed: an episode is not safe to seal,
        transfer or upload until its video file is complete.
        """
        self._commands.put((int(_Command.STOP), "", float(stop_time)))
        try:
            return self._sealed.get(timeout=_SEAL_TIMEOUT_S)
        except Empty:
            logger.error(
                "%s did not seal within %.0fs; its mp4 is missing its tail",
                self.name,
                _SEAL_TIMEOUT_S,
            )
            return None

    def close(self) -> None:
        if not self.is_alive():
            return
        # Stop time 0: every capture is newer, so an episode still open is cut
        # at once rather than drained. Shutting down is not the path that cares
        # about the tail — recorder.close() seals properly before it gets here.
        self._commands.put((int(_Command.SHUTDOWN), "", 0.0))
        self.join(timeout=15.0)
        if self.is_alive():
            logger.warning("%s did not exit; terminating", self.name)
            self.terminate()

    # -- encoder side, runs in its own process ------------------------------

    def run(self) -> None:
        # Reused across every frame: at archive resolution a fresh buffer per
        # capture is hundreds of MB/s of allocation churn.
        scratch = {
            "img": np.empty(self.shape, dtype=np.uint8),
            "frame_time": np.empty((), dtype=np.float64),
        }
        self._ready.set()
        while True:
            episode = self._wait_for_episode(scratch)
            if episode is None:
                return
            if self._encode_episode(*episode, scratch):
                return

    def _wait_for_episode(self, scratch: dict) -> tuple[str, float] | None:
        """Drop frames until an episode starts. ``None`` means shut down."""
        while True:
            try:
                cmd, path, timestamp = self._commands.get(timeout=0.05)
            except Empty:
                self._drain(scratch)
                continue
            if cmd == _Command.SHUTDOWN:
                return None
            if cmd == _Command.START:
                return path, timestamp
            # A STOP with nothing open: answer anyway so the caller's
            # end_episode() returns instead of waiting out its timeout.
            self._sealed.put(EpisodeVideoStats("", 0, 0, 0, 0))

    def _drain(self, scratch: dict) -> None:
        while self._frames is not None and not self._frames.empty():
            self._frames.get(out=scratch)

    def _encode_episode(
        self, video_path: str, start_time: float, scratch: dict
    ) -> bool:
        """Encode until the episode's last frame. True if shutdown was asked."""
        container, stream = self._open_container(video_path)

        next_index = 0
        n_unique = 0
        n_repeat = 0
        shutdown = False
        stop_time: float | None = None
        next_lag_check = time.monotonic() + _LAG_WARN_PERIOD_S
        drops_reported = 0

        while True:
            try:
                cmd, _, timestamp = self._commands.get_nowait()
                if cmd == _Command.SHUTDOWN:
                    shutdown = True
                if cmd in (_Command.SHUTDOWN, _Command.STOP):
                    stop_time = timestamp
            except Empty:
                pass

            try:
                self._frames.get(out=scratch)
            except Empty:
                if stop_time is not None:
                    break
                # Poll well inside a frame interval so the backlog stays short.
                time.sleep(0.1 / self.fps)
                continue

            frame_time = float(scratch["frame_time"])
            # Captures are FIFO and stamped in order, so the first one past the
            # end of the episode means every frame that belongs to it is muxed.
            if stop_time is not None and frame_time > stop_time:
                break

            if time.monotonic() >= next_lag_check:
                next_lag_check = time.monotonic() + _LAG_WARN_PERIOD_S
                with self._dropped.get_lock():
                    drops = self._dropped.value
                if drops > drops_reported:
                    logger.warning(
                        "%s cannot keep up: %d captures dropped so far this "
                        "episode; the video will repeat frames",
                        self.name,
                        drops,
                    )
                    drops_reported = drops

            if container is None:
                continue

            n_repeats, next_index = repeats_for(
                frame_time, start_time, self.fps, next_index
            )
            if n_repeats == 0:
                continue
            n_unique += 1
            n_repeat += n_repeats - 1

            frame = av.VideoFrame.from_ndarray(scratch["img"], format=self.input_pix_fmt)
            for _ in range(n_repeats):
                for packet in stream.encode(frame):
                    container.mux(packet)

        complete = True
        if container is not None:
            try:
                for packet in stream.encode():
                    container.mux(packet)
            except Exception:
                logger.exception("%s failed to flush %s", self.name, video_path)
                complete = False
            finally:
                container.close()

        with self._dropped.get_lock():
            n_dropped = self._dropped.value
        self._sealed.put(
            EpisodeVideoStats(
                path=video_path if container is not None else "",
                n_frames=next_index,
                n_unique=n_unique,
                n_repeat=n_repeat,
                n_dropped=n_dropped,
                complete=complete and container is not None,
            )
        )
        return shutdown

    def _open_container(self, video_path: str):
        """Returns ``(container, stream)``, or ``(None, None)`` if it failed.

        A camera that cannot encode must not take the session down with it; the
        episode records zero frames for this camera and the rest continues.
        """
        try:
            # A plain mp4 keeps its index in a trailer, so one that never
            # closed is not a short video, it is no video at all; fragmenting
            # it makes a killed episode decode up to where it stopped. This is
            # a muxer option and is silently ignored in stream.options below.
            container = av.open(
                video_path,
                mode="w",
                options={"movflags": "frag_keyframe+empty_moov+default_base_moof"},
            )
            # PyAV wants an exact rational frame rate, not a float, so that
            # the container's time base is exact rather than rounded.
            stream = container.add_stream(
                self.codec, rate=Fraction(self.fps).limit_denominator(10000)
            )
            height, width, _ = self.shape
            stream.width = width
            stream.height = height
            stream.pix_fmt = self.output_pix_fmt
            # PyAV defaults to slice threading, which splits each frame across
            # the cores and scales poorly; frame threading is worth about 1.8x
            # at archive resolution. The latency it adds does not matter to a
            # file nothing reads until the episode is over.
            stream.codec_context.thread_type = "FRAME"
            # A short GOP costs a few percent of size and buys cheap random
            # access, which is all training-time frame sampling ever does.
            stream.options = {
                "crf": str(self.crf),
                "preset": self.preset,
                "g": str(self.keyint),
            }
            return container, stream
        except Exception:
            logger.exception("%s could not open %s", self.name, video_path)
            return None, None
