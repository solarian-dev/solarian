"""The constant-rate grid an episode's video is encoded onto.

Two processes have to agree on which grid slot a captured frame belongs to: the
encoder, which repeats a frame when the next one is late so the mp4 stays
constant-rate, and the recorder, which writes the resulting frame index into the
episode table. They agree by both deriving it from ``(start_time, fps)`` rather
than by exchanging indices, so a dropped message cannot desynchronise the video
from the table.

Ported from UMI's ``get_accumulate_timestamp_idxs``.
"""

from __future__ import annotations

import math

#: Nudge so a timestamp landing exactly on ``start_time + k / fps`` lands in slot
#: k rather than k-1 on the wrong side of a float rounding error.
_EPS = 1e-5


def frame_index_for(timestamp: float, start_time: float, fps: float) -> int:
    """Grid slot owning *timestamp*. Negative before the episode started."""
    return math.floor((timestamp - start_time) * fps + _EPS)


def repeats_for(
    timestamp: float, start_time: float, fps: float, next_index: int
) -> tuple[int, int]:
    """How many times to encode a frame captured at *timestamp*, and the new cursor.

    One capture fills every grid slot from the cursor up to its own, so a frame
    arriving two slots late is written three times and the video stays in step
    with wall clock. A frame arriving inside a slot that is already filled is
    written zero times and dropped.
    """
    index = frame_index_for(timestamp, start_time, fps)
    if index < 0:
        return 0, next_index
    n_repeats = max(0, index - next_index + 1)
    return n_repeats, next_index + n_repeats
