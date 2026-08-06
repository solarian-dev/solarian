"""Application-level clock sync against the Solarian relay server.

OS clocks on controller / robot / server often disagree by tens or hundreds of
milliseconds. SCHEDULE_WAYPOINT stamps are wall-clock values; the robot converts
them to monotonic time with ``monotonic - wall + target_wall``. If the machines
disagree on wall time, waypoints land in the past and are silently dropped.

This module estimates ``offset`` such that ``time.time() + offset ≈ server_time``
via a short Cristian / NTP-style exchange over ZMQ REQ/REP. Controllers stamp
waypoints with :func:`synced_time`; modules convert with the same helper so both
sides share the server's time base without needing root or NTP.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass

import msgpack
import zmq

logger = logging.getLogger(__name__)

# Process-wide offset: local_wall + offset ≈ server_wall.
_offset_s: float = 0.0
_rtt_s: float = 0.0
_synced: bool = False


@dataclass(frozen=True)
class ClockSyncResult:
    """Outcome of a sync exchange with the relay server."""

    offset_s: float
    rtt_s: float
    samples: int
    discarded: int = 0


def get_offset() -> float:
    """Seconds to add to ``time.time()`` to approximate server wall time."""
    return _offset_s


def get_rtt() -> float:
    """Round-trip time of the last successful sync (seconds)."""
    return _rtt_s


def is_synced() -> bool:
    return _synced


def synced_time() -> float:
    """Wall time on the relay server's clock (``time.time() + offset``)."""
    return time.time() + _offset_s


def set_offset(offset_s: float, rtt_s: float = 0.0) -> None:
    """Install an offset (mainly for tests)."""
    global _offset_s, _rtt_s, _synced
    _offset_s = float(offset_s)
    _rtt_s = float(rtt_s)
    _synced = True


def reset_offset() -> None:
    """Clear sync state (tests / disconnect)."""
    global _offset_s, _rtt_s, _synced
    _offset_s = 0.0
    _rtt_s = 0.0
    _synced = False


def estimate_offset_from_samples(
    samples: list[tuple[float, float, float]],
) -> ClockSyncResult:
    """Estimate offset from ``(t0, t_server, t1)`` samples.

    Uses the lowest-RTT half of samples (NTP-style) and takes the median offset:
    ``offset = t_server - (t0 + t1) / 2``.
    """
    if not samples:
        raise ValueError("need at least one clock-sync sample")

    measured = []
    for t0, t_server, t1 in samples:
        rtt = float(t1) - float(t0)
        if rtt < 0:
            continue
        offset = float(t_server) - 0.5 * (float(t0) + float(t1))
        measured.append((rtt, offset))

    if not measured:
        raise ValueError("all clock-sync samples had negative RTT")

    measured.sort(key=lambda x: x[0])
    keep = max(1, len(measured) // 2)
    best = measured[:keep]
    offset = float(statistics.median(o for _, o in best))
    rtt = float(statistics.median(r for r, _ in best))
    return ClockSyncResult(
        offset_s=offset,
        rtt_s=rtt,
        samples=len(measured),
        discarded=len(samples) - len(measured),
    )


def sync_with_server(
    ip: str,
    port: int,
    *,
    samples: int = 8,
    timeout_ms: int = 2000,
    context: zmq.Context | None = None,
) -> ClockSyncResult:
    """Exchange timestamps with the relay and install the process offset.

    Raises on transport failure so callers can decide whether to soft-fail.
    """
    global _offset_s, _rtt_s, _synced

    own_context = context is None
    ctx = context or zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
    socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
    socket.connect(f"tcp://{ip}:{port}")

    collected: list[tuple[float, float, float]] = []
    try:
        for _ in range(max(1, int(samples))):
            t0 = time.time()
            socket.send(msgpack.packb({"t0": t0}, use_bin_type=True))
            raw = socket.recv()
            t1 = time.time()
            reply = msgpack.unpackb(raw, raw=False)
            t_server = float(reply["t_server"])
            collected.append((t0, t_server, t1))
            # Small gap so samples are not identical under coarse clocks.
            time.sleep(0.002)
    finally:
        socket.close(linger=0)
        if own_context:
            ctx.term()

    result = estimate_offset_from_samples(collected)
    _offset_s = result.offset_s
    _rtt_s = result.rtt_s
    _synced = True
    logger.info(
        "Clock synced to relay %s:%s — offset=%+.1f ms, rtt=%.1f ms (%d samples)",
        ip,
        port,
        result.offset_s * 1000.0,
        result.rtt_s * 1000.0,
        result.samples,
    )
    return result


def serve_clock_sync_reply(socket: zmq.Socket) -> None:
    """Handle one REQ on a REP socket (blocking recv/send)."""
    raw = socket.recv()
    try:
        req = msgpack.unpackb(raw, raw=False)
        t0 = req.get("t0")
    except Exception:  # noqa: BLE001
        t0 = None
    reply = {"t0": t0, "t_server": time.time()}
    socket.send(msgpack.packb(reply, use_bin_type=True))
