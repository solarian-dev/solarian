"""What the console front end makes of the bytes a terminal sends.

Esc is the interesting one: it is both the quit key and the first byte of what
the arrows and forward delete send, so the reader has to tell them apart.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from solarian.core.ui import console
from solarian.core.ui.base import UICommandType


class _FakeStdin:
    def __init__(self, data: str):
        self._pending = list(data)

    def read(self, n: int) -> str:
        return "".join(self._pending.pop(0) for _ in range(n))

    def readable(self) -> bool:
        return bool(self._pending)


@pytest.fixture
def typed(monkeypatch):
    """Return a factory that runs one ``poll`` over the given keystrokes."""

    def run(data: str) -> list:
        stdin = _FakeStdin(data)
        monkeypatch.setattr(console, "sys", SimpleNamespace(stdin=stdin))
        monkeypatch.setattr(
            console,
            "select",
            SimpleNamespace(
                select=lambda r, w, x, t: ([stdin] if stdin.readable() else [], [], [])
            ),
        )
        ui = object.__new__(console.ConsoleUI)
        ui._saved_termios = object()
        return ui.poll()

    return run


def _keys(commands) -> list[str]:
    return [
        c.data["key"] for c in commands if c.type == UICommandType.KEY_DOWN
    ]


def test_a_character_is_reported_as_itself(typed):
    assert _keys(typed("w")) == ["w"]


def test_each_character_is_both_a_press_and_a_release(typed):
    """cbreak reports characters, not transitions, so the up is synthesised."""
    types = [c.type for c in typed("w")]

    assert types == [UICommandType.KEY_DOWN, UICommandType.KEY_UP]


def test_the_delete_key_a_laptop_has_arrives_as_backspace(typed):
    assert _keys(typed("\x7f")) == ["backspace"]


def test_forward_delete_arrives_as_delete(typed):
    assert _keys(typed("\x1b[3~")) == ["delete"]


def test_esc_on_its_own_quits(typed):
    assert [c.type for c in typed("\x1b")] == [UICommandType.QUIT]


def test_an_arrow_key_is_read_and_dropped_rather_than_quitting(typed):
    """Treating the ESC that starts a sequence as a quit would end the session."""
    assert typed("\x1b[A") == []


def test_a_key_typed_behind_an_arrow_survives(typed):
    """Draining to the end of the burst would eat it."""
    assert _keys(typed("\x1b[Aw")) == ["w"]


def test_q_no_longer_quits(typed):
    """It jogs the arm now."""
    assert _keys(typed("q")) == ["q"]
