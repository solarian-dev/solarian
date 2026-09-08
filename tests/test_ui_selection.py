"""Which front end ``ui:`` and the flags it replaced select.

The mapping matters more than it looks: it is what decides whether a project
that has not been touched since ``use_gui`` still starts, and what a stale
``headless: true`` in a conf file now means.
"""

from __future__ import annotations

import pytest

from solarian.core.ui import UI_NAMES, _resolve_name


class Cfg:
    """Just the fields the resolver reads."""

    def __init__(self, ui: str | None = "console", headless: bool = False):
        self.ui = ui
        self.headless = headless


def resolve(cfg: Cfg, *, use_gui: bool = True, declared: bool = False) -> str:
    return _resolve_name(cfg, use_gui=use_gui, use_gui_declared=declared)


@pytest.mark.parametrize("name", UI_NAMES)
def test_an_explicit_ui_is_taken_as_written(name):
    assert resolve(Cfg(ui=name)) == name


def test_an_explicit_ui_wins_over_the_deprecated_flags():
    cfg = Cfg(ui="external", headless=True)

    assert resolve(cfg, use_gui=False, declared=True) == "external"


def test_the_default_is_the_console():
    assert resolve(Cfg(ui=None)) == "console"


def test_an_unknown_name_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="Unknown ui 'consle'"):
        resolve(Cfg(ui="consle"))


def test_headless_maps_to_the_console_it_always_was(caplog):
    """The old headless loop was the status board plus cbreak stdin."""
    assert resolve(Cfg(headless=True)) == "console"
    assert "deprecated" in caplog.text


def test_opting_out_of_the_gui_maps_to_the_opencv_windows(caplog):
    assert resolve(Cfg(), use_gui=False, declared=True) == "opencv"
    assert "ui=opencv" in caplog.text


def test_a_project_that_asked_for_the_window_is_told_it_moved(caplog):
    assert resolve(Cfg(), use_gui=True, declared=True) == "console"
    assert "ui=pygame" in caplog.text


def test_a_project_that_never_set_use_gui_is_not_warned(caplog):
    assert resolve(Cfg(), use_gui=True, declared=False) == "console"
    assert caplog.text == ""
