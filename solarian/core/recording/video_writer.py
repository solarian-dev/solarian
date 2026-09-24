"""One H.264 encoder process per camera.

The camera bus pushes a frame only while an episode is accepting, and never
waits: a full queue drops the frame rather than stalling capture, which is
the only acceptable failure mode when the producer is a sensor feeding a
control loop. Episode boundaries arrive on a separate queue. The encoder
owns the constant-rate grid and decides which slot each capture lands in.

An episode ends at a timestamp. Once stop is asked the camera stops pushing;
the encoder drains what is already queued and finishes when the queue is
empty or a capture is newer than the stop time. Waiting for a dry queue
while the camera was still feeding would encode past the end of the take.

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
    ``mp.Process`` cannot. It holds shared memory, a counter, and the
    accepting flag, all of which can.
    """

    def __init__(self, frames: SharedMemoryQueue, dropped, accepting):
        self._frames = frames
        self._dropped = dropped
        #: Shared with the robot process. False between episodes, so the
        #: camera bus can skip the archive resize until a take is open.
        self._accepting = accepting

    @property
    def accepting(self) -> bool:
        return bool(self._accepting.value)

    def write_frame(self, img: np.ndarray, frame_time: float) -> None:
        """Hand one capture to the encoder, or drop it.

        The bus calls this only while :attr:`accepting` is set. Dropping
        rather than blocking is deliberate — the alternative is a camera
        that stalls its module's loop because an encoder fell behind.
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
        threads: int = 1,
        input_pix_fmt: str = "rgb24",
        output_pix_fmt: str = "yuv420p",
        buffer_size: int = 64,
    ):
        super().__init__(name=f"VideoWriter-{name}", daemon=True)
        self.camera_name = name
        self.fps = float(fps)
        #: The grid rate as an exact rational. Frame timestamps are counted in
        #: its units, and PyAV wants a rational rather than a rounded float.
        self._rate = Fraction(self.fps).limit_denominator(10000)
        self.shape = tuple(shape)
        self.codec = codec
        self.crf = int(crf)
        self.keyint = int(keyint)
        self.preset = preset
        self.threads = int(threads)
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
        #: The camera process reads this; the robot process sets it around
        #: episode boundaries. Starts closed so idle captures are not resized.
        self._accepting = mp.Value("b", False)

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
        return ArchiveSink(self._frames, self._dropped, self._accepting)

    def start_episode(self, video_path: str, start_time: float) -> None:
        with self._dropped.get_lock():
            self._dropped.value = 0
        # Before the command, so a capture that lands during the handoff is
        # queued rather than resized away. Frames older than start_time are
        # skipped by the encoder.
        self._accepting.value = True
        self._commands.put((int(_Command.START), str(video_path), float(start_time)))

    def end_episode(self, stop_time: float) -> EpisodeVideoStats | None:
        """Close the mux and report what it contains.

        *stop_time* is the instant the episode ended, on the same clock the
        camera stamps captures with. Frames taken after it belong to no
        episode, and the encoder stops at the first one it sees.

        Blocks until the encoder has flushed: an episode is not safe to seal,
        transfer or upload until its video file is complete.
        """
        # Stop the resize before the command. Frames already queued still
        # encode; the encoder then sees an empty queue with stop_time set.
        self._accepting.value = False
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
        self._accepting.value = False
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
        import cv2

        container, stream = self._open_container(video_path)
        # x264 wants yuv420p either way. Handing it rgb24 makes swscale do the
        # conversion inside ``encode``, which costs about twice what OpenCV
        # does at archive resolution.
        #
        # Do not call cv2.setNumThreads here. This process is forked from one
        # that is itself resizing frames, and resizing OpenCV's pool in the
        # child deadlocks on a mutex the fork copied mid-flight. Measured, it
        # would buy nothing anyway: the convert is the same cost either way.
        to_yuv = (
            cv2.COLOR_BGR2YUV_I420
            if self.input_pix_fmt == "bgr24"
            else cv2.COLOR_RGB2YUV_I420
        )

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

            # Converted once even when the slot repeats: a repeat is the same
            # capture written twice, not a second capture.
            frame = av.VideoFrame.from_ndarray(
                cv2.cvtColor(scratch["img"], to_yuv), format="yuv420p"
            )
            frame.time_base = 1 / self._rate
            for slot in range(next_index - n_repeats, next_index):
                # Stamped per copy, not left to the encoder. This frame is
                # already in the stream's own format, so nothing reformats it
                # and ``encode`` is handed this very object each time round;
                # an inferred timestamp would repeat with it, and a repeated
                # one is not a worse video but a file the muxer refuses.
                frame.pts = slot
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
            stream = container.add_stream(self.codec, rate=self._rate)
            height, width, _ = self.shape
            stream.width = width
            stream.height = height
            stream.pix_fmt = self.output_pix_fmt
            # Inert at one thread, but the right kind to grow into: slice
            # threading splits a single frame across cores and scales poorly.
            # The latency frame threading adds does not matter to a file
            # nothing reads until the episode is over.
            stream.codec_context.thread_type = "FRAME"
            # Pinned rather than left to count the cores. Threading here buys
            # wall-clock at the cost of total CPU, and on a board shared with
            # a control loop the frame interval is already met without it.
            stream.codec_context.thread_count = self.threads
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
