"""Console front end: the Rich status board plus keys read off the terminal."""

from __future__ import annotations

import logging
import select
import sys
import time

from solarian.core.ui.base import UICommand, UICommandType, UISnapshot
from solarian.utils import StatusBoard

logger = logging.getLogger(__name__)


class ConsoleUI:
    """Default UI: telemetry on the status board, keys from stdin.

    The board is a process-wide singleton that every component already writes
    to, so this publishes only what the snapshot alone knows — which skills are
    running and what the keys do — instead of re-sending the observation rows
    ``get_state`` has already put there.

    Keys arrive in cbreak mode, which reports characters rather than
    transitions, so each one becomes a down *and* an up. Holding a key to jog an
    arm therefore needs a window (``ui=pygame``) or a browser, not a terminal.
    """

    captures_keys = True

    def __init__(self):
        self._last_tick = time.perf_counter()
        self._termios = None
        self._saved_termios = None

        if sys.stdin is None or not sys.stdin.isatty():
            # A runner, a pipe or a systemd unit has no terminal to put in
            # cbreak; the board still renders, there is just nothing to type on.
            logger.info("stdin is not a tty; console UI runs without key input")
            return

        import termios
        import tty

        self._termios = termios
        self._saved_termios = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        logger.info("Console UI ready — press 'q' to quit")

    def poll(self) -> list[UICommand]:
        if self._saved_termios is None:
            return []
        commands: list[UICommand] = []
        while select.select([sys.stdin], [], [], 0)[0]:
            char = sys.stdin.read(1).lower()
            if char == "q":
                commands.append(UICommand(UICommandType.QUIT))
                break
            commands.append(UICommand(UICommandType.KEY_DOWN, {"key": char}))
            commands.append(UICommand(UICommandType.KEY_UP, {"key": char}))
        return commands

    def render(self, snapshot: UISnapshot) -> None:
        running = snapshot.skills.get("running") or ()
        StatusBoard.set("Skills", ", ".join(running) if running else "(none)")
        if snapshot.key_help:
            StatusBoard.set(
                "Keys",
                "  ".join(f"{keys}:{desc}" for keys, desc in snapshot.key_help),
            )

    def tick(self, fps: int) -> None:
        period = (1.0 / fps) if fps else 0.01
        sleep_s = period - (time.perf_counter() - self._last_tick)
        if sleep_s > 0:
            time.sleep(sleep_s)
        self._last_tick = time.perf_counter()

    def close(self) -> None:
        if self._saved_termios is None:
            return
        self._termios.tcsetattr(sys.stdin, self._termios.TCSADRAIN, self._saved_termios)
        self._saved_termios = None
