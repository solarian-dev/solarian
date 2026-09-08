"""Measure and install the local wall-clock offset vs the relay server."""

from __future__ import annotations

import time

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.clock_sync import sync_with_server, synced_time
from solarian.core.configs import ServerConfig


@CommandRegistry.register("sync-clock", "server")
class SyncClockCommand(BaseCommand):
    help = (
        "Sync this machine's Solarian clock offset to the relay server "
        "(Cristian / NTP-style; does not change the OS clock)"
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
