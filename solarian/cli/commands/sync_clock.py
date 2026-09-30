"""Measure and install the local wall-clock offset vs the relay server."""

from __future__ import annotations

import time

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.clock_sync import sync_with_server, synced_time
from solarian.core.configs import ServerConfig


@CommandRegistry.register("sync-clock", "server")
class SyncClockCommand(BaseCommand):
    category = "Utilities"
    help = "Sync Solarian wall-clock offset to the relay server (does not change OS time)."
    description = """
    Takes 12 timing samples against the relay's clock-sync port and reports the
    offset and round trip. Use it to check that two machines agree on a timebase
    before trusting anything that subtracts their timestamps, such as
    calibrate-camera-latency across a laptop and a robot.

    The relay must already be running (`solarian run-server`). The offset applies
    to this invocation only; run-robot and run-controller sync themselves on
    connect, and the OS clock is never touched.
    """
    examples = (
        "solarian sync-clock",
        "solarian sync-clock -d pi",
    )

    def get_config_class(self):
        return ServerConfig

    def handle(self, cfg: DictConfig) -> None:
        conn = cfg.connection
        print(
            f"Syncing to relay at {conn.ip}:{conn.clock_sync_port} "
            f"(server must be running)..."
        )
        try:
            result = sync_with_server(
                conn.ip, int(conn.clock_sync_port), samples=12, connection=conn
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Clock sync failed: {exc}")
            raise SystemExit(1) from exc

        local = time.time()
        print(f"  samples:     {result.samples}")
        print(f"  offset:      {result.offset_s * 1000.0:+.2f} ms  (local + offset ≈ server)")
        print(f"  rtt:         {result.rtt_s * 1000.0:.2f} ms")
        print(f"  local time:  {local:.6f}")
        print(f"  synced time: {synced_time():.6f}")
        print(
            "\nInstalled process offset for this CLI invocation only. "
            "run-controller / run-robot sync automatically on connect."
        )
