"""Unit tests for application-level clock sync helpers."""

from __future__ import annotations

import threading
import time

import pytest
import zmq

from solarian.core.clock_sync import (
    estimate_offset_from_samples,
    reset_offset,
    serve_clock_sync_reply,
    sync_with_server,
    synced_time,
)


def test_estimate_offset_median_of_best_rtt_half():
    # Two fast samples with +100 ms offset, two slow with junk offset.
    samples = [
        (0.0, 0.100, 0.010),  # rtt 10ms, offset 0.095
        (1.0, 1.100, 1.012),  # rtt 12ms, offset 0.094
        (2.0, 2.500, 2.200),  # rtt 200ms, offset 0.400
        (3.0, 3.500, 3.250),  # rtt 250ms, offset 0.375
    ]
    result = estimate_offset_from_samples(samples)
    assert result.samples == 4
    assert abs(result.offset_s - 0.0945) < 1e-6
    assert abs(result.rtt_s - 0.011) < 1e-6


def test_sync_with_server_roundtrip():
    reset_offset()
    ctx = zmq.Context()
    port = ctx.socket(zmq.REP)
    port.bind("tcp://127.0.0.1:*")
    endpoint = port.getsockopt_string(zmq.LAST_ENDPOINT)
    listen_port = int(endpoint.rsplit(":", 1)[-1])

    stop = threading.Event()

    def _serve():
        poller = zmq.Poller()
        poller.register(port, zmq.POLLIN)
        while not stop.is_set():
            if poller.poll(50):
                serve_clock_sync_reply(port)

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    try:
        # Inject a fixed server skew by monkeypatching time on the reply path:
        # serve uses time.time(); instead verify sync converges near 0 for
        # same-process client/server.
        result = sync_with_server("127.0.0.1", listen_port, samples=4, timeout_ms=1000)
        assert abs(result.offset_s) < 0.05
        assert result.rtt_s >= 0.0
        # Two separate clock reads, so the tolerance has to cover the time
        # between them: at 1e-6 this compares against the cost of the syscall.
        assert abs(synced_time() - time.time() - result.offset_s) < 1e-3
    finally:
        stop.set()
        thread.join(timeout=1.0)
        port.close(linger=0)
        ctx.term()
        reset_offset()
