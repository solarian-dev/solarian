"""A teleop device sampled somewhere else, published here as observations.

Reading a device and consuming what it reports are already separate concerns:
:class:`~solarian.core.skill.Skill` code reads
``observations["leader_arm"]``, never the serial bus. That separation is what
makes a remote device tractable — if the sampling happens in a browser tab and
the samples arrive over a WebSocket, the skill cannot tell, and by design should
not be able to.

What genuinely has to sit next to the hardware is the request/response polling:
one servo read per joint per tick, round-tripped over the internet, would be
fatal. Sampling locally at 30-60 Hz and streaming the samples is not.
"""

from __future__ import annotations

import logging

from solarian.core.clock_sync import synced_time
from solarian.core.configs import RemoteInputConfig
from solarian.core.execution import ExecutionMode
from solarian.core.module import Module

logger = logging.getLogger("module")


class RemoteInputModule(Module):
    """Stands in for an input device this process cannot reach.

    Inline, like the local module it replaces: the host's tick is its tick, so
    every skill watching the device sees one sample per tick at one instant.
    ``update_observations`` takes the newest sample off the UI stream instead of
    off a bus, and publishes it into the same named
    :class:`~solarian.utils.shared_memory.observation_buffer.ObservationBuffer`.
    """

    kind: str = "input"

    execution_mode = ExecutionMode.INLINE
    config_class = RemoteInputConfig

    #: Nothing to describe in YAML — the device is not on this machine — so a
    #: project need not ship a config for a module it only runs in the cloud.
    default_config = {"frequency": 60.0}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #: ``(readings, arrival_s)``, swapped as one tuple because the UI thread
        #: writes it and the control thread reads it: a single attribute
        #: assignment cannot be observed half-applied, where field-by-field
        #: mutation could hand a consumer a pose from two different instants.
        self._latest: tuple[dict, float] | None = None
        #: Newest sender stamp accepted so far. Touched only by the UI thread.
        self._accepted_stamp = 0.0
        self._stale = False

    # ========= sampling ============
    def accept_sample(self, fields: dict) -> None:
        """Take one reading of the device off the UI stream.

        Called from whichever thread owns the UI, so it does the minimum: reject
        anything not newer than what has already been accepted, then swap it in.
        Reordering is ordinary on a lossy link, and for teleop the newest sample
        is the only one worth having.

        The sender's stamps order the stream but do not date it — a browser's
        clock is not synced to the relay's, and could be minutes out. Readings go
        out stamped with their arrival in this process's synced time base, which
        is what a locally-read module reports and what a consumer means by "when
        this was true".
        """
        readings = {}
        newest_stamp = 0.0
        arrival_s = synced_time()
        for name, field in fields.items():
            # Off the wire: a sampler can send a field shaped any way at all.
            if not isinstance(field, dict) or "value" not in field:
                continue
            try:
                readings[name] = {
                    "value": float(field["value"]),
                    "timestamp": arrival_s,
                }
            except (TypeError, ValueError):
                logger.debug("%s: ignored non-numeric field '%s'", self.name, name)
                continue
            newest_stamp = max(newest_stamp, float(field.get("timestamp") or 0.0))

        if not readings:
            return
        if newest_stamp and newest_stamp <= self._accepted_stamp:
            logger.debug(
                "%s: dropped an out-of-order sample (%.6f <= %.6f)",
                self.name,
                newest_stamp,
                self._accepted_stamp,
            )
            return

        self._accepted_stamp = newest_stamp
        self._latest = (readings, arrival_s)

    # ========= observations ============
    def update_observations(self):
        """Publish the newest sample, or an empty one once the link goes quiet.

        Publishing nothing rather than re-publishing the last pose is the whole
        safety argument: a consumer that sees no joints emits no waypoint, so a
        dropped input link leaves the follower where it is instead of holding a
        target the operator stopped commanding. The robot-side watchdog is the
        backstop for the link that carries the actions; this is the one for the
        link that carries the intent.
        """
        if self.observation_buffer is None:
            return

        latest = self._latest
        if latest is None:
            return
        readings, arrival_s = latest

        age_s = synced_time() - arrival_s
        if age_s > self.cfg.stale_after_s:
            if not self._stale:
                logger.warning(
                    "%s: no input sample for %.0f ms; publishing nothing until "
                    "a fresh one arrives",
                    self.name,
                    age_s * 1000.0,
                )
                self._stale = True
            self.observation_buffer.put({})
            return

        if self._stale:
            logger.info("%s: input link recovered", self.name)
            self._stale = False
        self.observation_buffer.put(readings)

    def control_step(self):
        """Input-only: there is nothing here to drive."""

    # ========= hardware hooks ============
    # There is no bus: the device is on the sampler's machine. These say so,
    # rather than leaving the base class to reach for components that were never
    # created.
    def connect(self, calibration: dict | None = None):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def is_calibrated(self) -> bool:
        """The sampler calibrates what it samples, and normalizes before sending."""
        return True
