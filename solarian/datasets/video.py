"""Decoding an episode's mp4 and encoding a derived one like the session's own."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import BinaryIO, Iterator, Mapping

import av
import numpy as np

from solarian.core.recording.archive import Session

#: solarian's recording VideoWriter defaults, for sessions whose session.json lacks "video".
VIDEO_OPTIONS = {"codec": "libx264", "crf": 20, "keyint": 30, "preset": "veryfast", "pix_fmt": "yuv420p"}


def decode(session: Session, member: str, n: int | None = None, fmt: str = "rgb24") -> Iterator[np.ndarray]:
    """The frames (at most ``n``) of the mp4 ``member`` as ``fmt`` arrays."""
    with session.open(member) as fh:
        yield from decode_file(fh, n, fmt)


def decode_file(fh: BinaryIO, n: int | None = None, fmt: str = "rgb24") -> Iterator[np.ndarray]:
    """The frames (at most ``n``) of the mp4 open as ``fh`` as ``fmt`` arrays."""
    with av.open(fh) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for k, frame in enumerate(container.decode(stream)):
            if n is not None and k >= n:
                return
            yield frame.to_ndarray(format=fmt)


def video_fps(session: Session, member: str) -> Fraction:
    with session.open(member) as fh, av.open(fh) as container:
        return container.streams.video[0].average_rate


class VideoEncoder:
    """An mp4 at ``path`` encoded like the session's own videos (``session.json["video"]``)."""

    def __init__(self, path: Path, fps, options: Mapping | None = None):
        self.options = {**VIDEO_OPTIONS, **(options or {})}
        self.path = Path(path)
        self.fps = Fraction(fps).limit_denominator(10000)
        self.container = None
        self.stream = None

    def write(self, rgb: np.ndarray) -> None:
        if self.container is None:
            self.container = av.open(str(self.path), mode="w")
            self.stream = self.container.add_stream(self.options["codec"], rate=self.fps)
            self.stream.height, self.stream.width = rgb.shape[:2]
            self.stream.pix_fmt = self.options["pix_fmt"]
            self.stream.options = {"crf": str(self.options["crf"]), "preset": self.options["preset"],
                                   "g": str(self.options["keyint"])}
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
            self.container.mux(packet)

    def close(self) -> Path:
        if self.container is None:
            raise ValueError(f"{self.path}: no frames were written")
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        return self.path
