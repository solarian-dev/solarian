from enum import Enum
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

import solarian.core.configs as core_configs
from solarian.utils.configs import (
    CONFIG_DOCS_URL,
    ConfigError,
    docs_url,
    field_help,
    parse_with_friendly_errors,
)


class Color(str, Enum):
    RED = "red"
    BLUE = "blue"


class Inner(BaseModel):
    """A nested section.

    Docs: https://example.com/inner
    """
    #: Port the inner thing listens on.
    port: int


class Outer(BaseModel):
    """The top-level config.

    Docs: https://example.com/outer
    """
    model_config = ConfigDict(extra="forbid")

    #: Where the checkpoint lives.
    #: Second line of help.
    policy_path: str
    #: Compute device.
    device: Literal["cpu", "cuda"] = "cpu"
    color: Color = Color.RED
    inner: Inner | None = None
    sizes: list[int] = []
    named: dict[str, Inner] = {}


class Undocumented(BaseModel):
    value: int


def _error(data, **kwargs) -> ConfigError:
    with pytest.raises(ConfigError) as info:
        parse_with_friendly_errors(Outer, data, **kwargs)
    return info.value


def test_missing_field_lists_type_and_help():
    text = str(_error({}, config_path="/tmp/outer.yaml"))
    assert "Configuration error in /tmp/outer.yaml (Outer)" in text
    assert "Missing required fields:" in text
    assert "policy_path: str" in text
    assert "Where the checkpoint lives. Second line of help." in text
    assert "input_value=" not in text
    assert "Traceback" not in text


def test_command_config_suggests_a_cli_override(monkeypatch):
    monkeypatch.setenv("SOLARIAN_CURRENT_CONFIG", "/tmp/controller.yaml")
    text = str(_error({}))
    assert "/tmp/controller.yaml" in text
    assert "policy_path=<value>" in text


def test_module_config_does_not_suggest_a_cli_override():
    text = str(_error({}, config_path="/tmp/arm.yaml"))
    assert "command line" not in text


def test_wrong_literal_shows_input_allowed_values_and_default():
    text = str(_error({"policy_path": "p", "device": "gpu"}))
    assert "Invalid values:" in text
    assert "device: got 'gpu'" in text
    assert "'cpu'" in text and "'cuda'" in text
    assert "Default: 'cpu'" in text
    assert "Compute device." in text


def test_wrong_enum_value():
    text = str(_error({"policy_path": "p", "color": "green"}))
    assert "color: got 'green'" in text
    assert "'red'" in text and "'blue'" in text


def test_nested_error_uses_dotted_path_and_its_own_docs():
    text = str(_error({"policy_path": "p", "inner": {"port": "x"}}))
    assert "inner.port: got 'x'" in text
    assert "Port the inner thing listens on." in text
    assert "Docs: https://example.com/inner" in text
    assert text.rstrip().endswith("Docs: https://example.com/outer")


def test_missing_nested_field():
    text = str(_error({"policy_path": "p", "inner": {}}))
    assert "inner.port: int" in text


def test_dict_of_models_path():
    text = str(_error({"policy_path": "p", "named": {"a": {"port": "x"}}}))
    assert "named.a.port: got 'x'" in text


def test_list_index_path():
    text = str(_error({"policy_path": "p", "sizes": [1, "two"]}))
    assert "sizes[1]: got 'two'" in text


def test_unknown_field_suggests_close_match():
    text = str(_error({"policy_path": "p", "polcy_path": "q"}))
    assert "Unknown fields:" in text
    assert "polcy_path: not a field of Outer (did you mean 'policy_path'?)" in text


def test_long_inputs_are_truncated():
    text = str(_error({"policy_path": "p", "device": "x" * 500}))
    assert "x" * 100 not in text
    assert "..." in text


def test_class_without_docs_falls_back():
    with pytest.raises(ConfigError) as info:
        parse_with_friendly_errors(Undocumented, {})
    assert f"Docs: {CONFIG_DOCS_URL}" in str(info.value)


class PluginInner(Inner):
    extra_port: int = 0


def test_docs_url_is_inherited():
    assert docs_url(PluginInner) == "https://example.com/inner"


def test_field_help_reads_inherited_comments():
    help_ = field_help(core_configs.CameraConfig)
    assert help_["fps"]
    assert help_["port"]


def _core_config_classes():
    return [
        cls
        for cls in vars(core_configs).values()
        if isinstance(cls, type)
        and issubclass(cls, BaseModel)
        and cls.__module__ == core_configs.__name__
    ]


@pytest.mark.parametrize("cls", _core_config_classes(), ids=lambda c: c.__name__)
def test_every_core_config_field_has_help(cls):
    missing = [name for name, text in field_help(cls).items() if not text]
    assert not missing, f"{cls.__name__} fields without a #: comment: {missing}"


@pytest.mark.parametrize("cls", _core_config_classes(), ids=lambda c: c.__name__)
def test_every_core_config_has_a_docs_link(cls):
    assert docs_url(cls), f"{cls.__name__} has no 'Docs:' line in its docstring"
