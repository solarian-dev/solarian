"""No front end at all: for a controller driven entirely by code."""

from __future__ import annotations

import time

from solarian.core.ui.base import UICommand, UISnapshot


class NullUI:
    """Draws nothing and reads nothing.

    For a controller nobody is watching — a scripted recording run, a test — so
    the loop still paces itself but no terminal is put in cbreak and no window
    is opened. The status board keeps rendering; it is not this UI's to own.
    """

    captures_keys = True

    def __init__(self):
        self._last_tick = time.perf_counter()

    def poll(self) -> list[UICommand]:
        return []

    def render(self, snapshot: UISnapshot) -> None:
        pass

    def tick(self, fps: int) -> None:
        period = (1.0 / fps) if fps else 0.01
        sleep_s = period - (time.perf_counter() - self._last_tick)
        if sleep_s > 0:
            time.sleep(sleep_s)
        self._last_tick = time.perf_counter()

    def close(self) -> None:
        pass
