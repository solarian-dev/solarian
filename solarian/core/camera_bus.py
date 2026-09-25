"""Camera buses, and the archive tap that only they need.

Kept apart from :mod:`solarian.core.component_bus` because none of it means
anything to a servo. Archive sinks, capture ages and colour order are camera
concerns, and a motor bus that inherits them inherits questions it can never
answer.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, TYPE_CHECKING

from solarian.core.clock_sync import synced_time
from solarian.core.sensor_bus import SensorBus

if TYPE_CHECKING:
    import numpy as np

    from solarian.core.recording.video_writer import ArchiveSink


class ColorMode(str, Enum):
    """Channel order an observation carries. Sensors deliver BGR either way."""

    RGB = "rgb"
    BGR = "bgr"


class ArchivesFrames:
    """The archive tap, for any bus whose reads are camera frames.

    A mixin rather than a base class because two unrelated kinds of bus need
    it: the camera buses below, and the proxy in :mod:`solarian.core.bus_server`
    that stands in for a bus living in another process. It is deliberately not
    on :class:`~solarian.core.component_bus.ComponentBus`, so that asking a
    servo bus where its archive goes is a mistake the reader can see.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #: Where archive frames go, by component name. Empty unless the robot
        #: wired an encoder to this bus, which it only does for cameras.
        self.archive_sinks: Dict[str, "ArchiveSink"] = {}
        #: When each component's most recent sample was taken. The module reads
        #: this instead of stamping on return, so the observation and the
        #: archive carry the same instant.
        self.capture_timestamps: Dict[str, float] = {}
        #: Calibrated delay between the physical sample and this bus seeing it,
        #: for hardware that cannot say. A constant, and therefore a guess at
        #: something that varies.
        self.capture_latency: float = 0.0
        #: Age of the capture just read, where the hardware timestamps its own
        #: frames. Takes precedence over :attr:`capture_latency`, which is the
        #: same quantity estimated rather than measured.
        self.measured_capture_age: float | None = None

    def archive_and_finish(self, name: str, frame: "np.ndarray", cfg):
        """Feed the archive, and return the frame the observation carries.

        Takes a rotated capture still in the sensor's own channel order, and
        owns the last two steps every camera bus performs: the colour swap and
        the downscale. Owning both is the point, because their order is what
        they cost. Downscaling first makes the swap a quarter of the work and
        lets the archive reuse the result instead of resampling the full frame
        a second time: off a 2028x1520 sensor that is 13 ms a frame against 3.
        Nothing is given up for it — area averaging resamples each channel on
        its own, so swapping two of them before or after is the same image.

        The two downscales are only interchangeable while ``archive_*`` and
        ``resize_*`` agree. An archive deliberately kept larger than the
        observation has to come off the full-resolution capture, and pays
        full price for both.
        """
        import cv2

        from solarian.core.optical_remap import finish_camera_frame

        shared = (cfg.archive_width, cfg.archive_height) == (
            cfg.resize_width,
            cfg.resize_height,
        )
        if shared:
            frame = finish_camera_frame(frame, cfg)
        if cfg.color_mode == ColorMode.RGB.value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.archive_capture(name, frame, cfg)
        return frame if shared else finish_camera_frame(frame, cfg)

    def archive_capture(self, name: str, frame: "np.ndarray", cfg) -> float:
        """Send *frame* to the archive encoder and return its capture stamp.

        A camera bus calls this after rotation and before the optical remap.
        The remap is a calibration artifact rather than data: kept out of the
        archive, a later recalibration re-derives every dataset already
        recorded; baked in, it could never be undone.

        The stamp returned is also the one the observation must carry. The
        encoder and the recorder each place the frame on the episode's grid by
        computing from it independently, and they can only agree because there
        is exactly one of it.

        How old the frame is matters more than it looks. The grid has a slot
        per frame interval and keeps only the first capture to land in each,
        so a stamp that jitters by half an interval puts two captures in one
        slot and leaves the next empty — the second is discarded, the gap is
        filled with a repeat, and the take loses coverage although the camera
        missed nothing. Measured on a loaded Pi 5, stamping on arrival cost
        18% of captures that way. Hence the preference for an age the hardware
        measured over one the config guessed.
        """
        age = (
            self.capture_latency
            if self.measured_capture_age is None
            else self.measured_capture_age
        )
        captured_at = synced_time() - age
        self.capture_timestamps[name] = captured_at

        sink = self.archive_sinks.get(name)
        # A sink with no ``accepting`` flag is always fed. The real encoder
        # sets the flag only while an episode is open, so the archive
        # resize and the shared-memory copy do not run between takes.
        if sink is not None and getattr(sink, "accepting", True):
            sink.write_frame(archive_frame(frame, cfg), captured_at)
        return captured_at


class CameraBus(ArchivesFrames, SensorBus):
    """Base class for buses whose components are cameras.

    Subclass this rather than :class:`~solarian.core.sensor_bus.SensorBus`
    when the hardware produces frames: it is what brings the archive tap, and
    the recorder has no other way to reach the sensor's own output.
    """


def archive_frame(frame: "np.ndarray", cfg) -> "np.ndarray":
    """Downscale a capture to the archive resolution, if one is configured.

    Unset archives at whatever resolution the sensor produced. This is the
    knob for encoder CPU on a Raspberry Pi, and it deliberately does not touch
    the sensor mode: selecting a smaller mode would crop the field of view and
    invalidate the camera's calibration.
    """
    if cfg.archive_width is None or cfg.archive_height is None:
        return frame
    if frame.shape[:2] == (int(cfg.archive_height), int(cfg.archive_width)):
        return frame

    import cv2

    return cv2.resize(
        frame,
        (int(cfg.archive_width), int(cfg.archive_height)),
        interpolation=cv2.INTER_AREA,
    )
