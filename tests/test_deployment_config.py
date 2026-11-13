"""Selecting a deployment, and inheriting one from another."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from solarian.utils import configs


@pytest.fixture
def conf_dir(tmp_path, monkeypatch):
    """A conf/ tree with an empty deployment cache pointed at it."""
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    monkeypatch.setenv("SOLARIAN_CONF_PATH", str(tmp_path))
    monkeypatch.setattr(configs, "_deployment_cache", {})
    return deployment


def write(directory, name: str, body: dict) -> None:
    (directory / f"{name}.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(body)))


def test_the_named_deployment_is_the_one_loaded(conf_dir, monkeypatch):
    write(conf_dir, "localhost", {"connection": {"ip": "127.0.0.1"}})
    write(conf_dir, "pi", {"connection": {"ip": "pi.local"}})
    monkeypatch.setenv("SOLARIAN_DEPLOYMENT", "pi")

    assert configs.load_deployment_config().connection.ip == "pi.local"


def test_extending_inherits_everything_it_does_not_restate(conf_dir, monkeypatch):
    write(conf_dir, "pi", {
        "connection": {"ip": "pi.local", "action_sender_port": 5555},
        "actuator_port_right_arm": "/dev/ttyACM0",
    })
    write(conf_dir, "pi_relay_mac", {
        "extends": "pi",
        "connection": {"ip": "mac.local"},
    })
    monkeypatch.setenv("SOLARIAN_DEPLOYMENT", "pi_relay_mac")

    cfg = configs.load_deployment_config()

    assert cfg.connection.ip == "mac.local", "the value it restated"
    assert cfg.connection.action_sender_port == 5555, "merged, not replaced"
    assert cfg.actuator_port_right_arm == "/dev/ttyACM0"
    assert "extends" not in cfg, "a directive, not config the project can read"


def test_extends_chains(conf_dir, monkeypatch):
    write(conf_dir, "base", {"a": 1, "b": 1, "c": 1})
    write(conf_dir, "middle", {"extends": "base", "b": 2, "c": 2})
    write(conf_dir, "leaf", {"extends": "middle", "c": 3})
    monkeypatch.setenv("SOLARIAN_DEPLOYMENT", "leaf")

    cfg = configs.load_deployment_config()

    assert (cfg.a, cfg.b, cfg.c) == (1, 2, 3)


def test_a_cycle_is_reported_instead_of_recursing(conf_dir, monkeypatch):
    write(conf_dir, "one", {"extends": "two"})
    write(conf_dir, "two", {"extends": "one"})
    monkeypatch.setenv("SOLARIAN_DEPLOYMENT", "one")

    with pytest.raises(ValueError, match="one -> two -> one"):
        configs.load_deployment_config()


def test_extending_something_that_is_not_there_says_who_asked(conf_dir, monkeypatch):
    write(conf_dir, "solarian_base", {"extends": "lapotp", "connection": {"ip": "relay"}})
    monkeypatch.setenv("SOLARIAN_DEPLOYMENT", "solarian_base")

    with pytest.raises(ValueError, match="'solarian_base' extends 'lapotp'"):
        configs.load_deployment_config()


def test_an_unknown_deployment_lists_the_ones_there_are(conf_dir, monkeypatch):
    write(conf_dir, "localhost", {"connection": {"ip": "127.0.0.1"}})
    monkeypatch.setenv("SOLARIAN_DEPLOYMENT", "nope")

    with pytest.raises(ValueError, match="Available deployments: localhost"):
        configs.load_deployment_config()


def test_a_missing_deployment_key_names_the_file_and_the_port(conf_dir, tmp_path, monkeypatch):
    write(conf_dir, "localhost", {"connection": {"ip": "127.0.0.1"}})
    module = tmp_path / "arm.yaml"
    module.write_text("actuator_port: ${deployment.actuator_port_arm}\n")
    monkeypatch.delenv("SOLARIAN_DEPLOYMENT", raising=False)

    with pytest.raises(configs.ConfigError, match="actuator_port_arm") as exc:
        configs.load_config_yaml(str(module), dict)

    message = str(exc.value)
    assert "arm.yaml" in message
    assert "actuator_port requires ${deployment.actuator_port_arm}" in message
    assert "deployment/localhost.yaml" in message


def test_a_missing_key_of_the_file_itself_names_that_file(conf_dir, tmp_path):
    write(conf_dir, "localhost", {})
    module = tmp_path / "arm.yaml"
    module.write_text("actuators:\n  a:\n    port: ${actuator_port}\n")

    with pytest.raises(configs.ConfigError) as exc:
        configs.load_config_yaml(str(module), dict)

    message = str(exc.value)
    assert "actuators.a.port requires ${actuator_port}" in message
    assert "deployment" not in message
    assert message.endswith("arm.yaml.")


def test_a_yaml_syntax_error_names_the_file_and_line(tmp_path):
    module = tmp_path / "arm.yaml"
    module.write_text("frequency: 50\nactuators:\n  - a\n  b: 1\n")

    with pytest.raises(configs.ConfigError, match=r"arm\.yaml is not valid YAML, line 4"):
        configs.load_yaml(module)


def test_a_config_error_survives_a_child_traceback(conf_dir, tmp_path):
    import traceback

    from solarian.core.skill import failure_reason

    write(conf_dir, "localhost", {})
    module = tmp_path / "arm.yaml"
    module.write_text("actuator_port: ${deployment.actuator_port_arm}\n")
    try:
        configs.load_config_yaml(str(module), dict)
    except configs.ConfigError as exc:
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        original = str(exc)

    recovered = configs.config_error_in(tb_str)
    assert recovered is not None and str(recovered) == original
    assert failure_reason(tb_str) == original
    assert configs.config_error_in("Traceback...\nValueError: nope") is None


def test_a_project_with_no_deployment_directory_gets_an_empty_config(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SOLARIAN_CONF_PATH", str(tmp_path))
    monkeypatch.setattr(configs, "_deployment_cache", {})

    assert configs.load_deployment_config() == OmegaConf.create({})
