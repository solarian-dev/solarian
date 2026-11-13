"""Which front end ``ui:`` selects."""

from __future__ import annotations

import pytest

from solarian.core.configs import ControllerConfig
from solarian.core.ui import UI_NAMES, _resolve_name


class Cfg:
    """Just the field the resolver reads."""

    def __init__(self, ui: str):
        self.ui = ui


@pytest.mark.parametrize("name", UI_NAMES)
def test_an_explicit_ui_is_taken_as_written(name):
    assert _resolve_name(Cfg(ui=name)) == name


def test_the_default_is_the_gui():
    assert ControllerConfig.model_fields["ui"].default == "gui"


def test_an_unknown_name_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="Unknown ui 'consle'"):
        _resolve_name(Cfg(ui="consle"))
