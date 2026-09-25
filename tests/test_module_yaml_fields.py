"""Reading and writing single fields in a module yaml.

``calibrate-arm`` / ``calibrate-gripper`` write their measurements straight back
into ``conf/<module>.yaml``, and a skill in another process reads the action
latency out of the same file. Both sides go through here, so a module yaml is
edited by hand and by a calibration in turn and has to survive both.
"""

from __future__ import annotations

import pytest

from solarian.core.module_yaml import (
    module_yaml_path,
    patch_top_level_value,
    read_module_field,
)

_YAML = """\
actuator_port: ${deployment.actuator_port_right_gripper}
frequency: 120
# Max aperture speed (mm/s).
max_speed: 150.0  # measured 2026-09
receive_latency: 0.001073
actuators:
    gripper:
        id: 7
        max_speed: 9.0
"""


@pytest.fixture
def module_conf(tmp_path, monkeypatch):
    conf = tmp_path / "conf"
    conf.mkdir()
    (conf / "right_gripper.yaml").write_text(_YAML)
    monkeypatch.setenv("SOLARIAN_CONF_PATH", str(conf))
    return conf / "right_gripper.yaml"


def test_an_existing_field_is_replaced_in_place(module_conf):
    patch_top_level_value(str(module_conf), "max_speed", 295.0)
    lines = module_conf.read_text().splitlines()
    assert lines[3] == "max_speed: 295  # measured 2026-09"


def test_the_comment_above_a_field_survives_the_write(module_conf):
    """A calibration overwrites numbers, never the reasoning next to them."""
    patch_top_level_value(str(module_conf), "max_speed", 295.0)
    assert "# Max aperture speed (mm/s)." in module_conf.read_text()


def test_a_nested_field_of_the_same_name_is_left_alone(module_conf):
    """An actuator may declare its own max_speed; only column zero is the module's."""
    patch_top_level_value(str(module_conf), "max_speed", 295.0)
    assert "        max_speed: 9.0" in module_conf.read_text()


def test_a_new_field_lands_above_the_hardware_map(module_conf):
    patch_top_level_value(str(module_conf), "action_latency", 0.042)
    lines = module_conf.read_text().splitlines()
    assert lines[lines.index("actuators:") - 1] == "action_latency: 0.042"


def test_a_commented_out_field_is_not_mistaken_for_the_field(tmp_path):
    path = tmp_path / "right_arm.yaml"
    path.write_text("# max_speed: [0.25, 0.6]\nmax_speed: [0.5, 1.5]\nactuators:\n")
    patch_top_level_value(str(path), "max_speed", [0.31, 1.5])
    assert path.read_text().splitlines()[:2] == [
        "# max_speed: [0.25, 0.6]",
        "max_speed: [0.31, 1.5]",
    ]


def test_a_list_is_written_as_a_flow_sequence(tmp_path):
    """An arm's max_speed is (translation, rotation), and only the first is measured."""
    path = tmp_path / "right_arm.yaml"
    path.write_text("frequency: 90\nactuators:\n")
    patch_top_level_value(str(path), "max_speed", [0.31, 1.5])
    assert "max_speed: [0.31, 1.5]" in path.read_text()


def test_a_written_field_reads_back(module_conf):
    patch_top_level_value(str(module_conf), "action_latency", 0.042)
    assert read_module_field("right_gripper", "action_latency", 0.0) == 0.042


def test_reading_ignores_interpolations_elsewhere_in_the_file(module_conf):
    """The skill reads one number without composing a deployment tree.

    ``actuator_port`` is a ``${deployment.*}`` reference that would not resolve
    here, and selecting a different key must never touch it.
    """
    assert read_module_field("right_gripper", "receive_latency", 0.0) == 0.001073


def test_a_missing_field_falls_back_to_the_default(module_conf):
    assert read_module_field("right_gripper", "action_latency", 0.0) == 0.0


def test_a_module_with_no_yaml_falls_back_to_the_default(module_conf):
    """A skill names modules that may live on another machine's conf."""
    assert module_yaml_path("left_gripper") is None
    assert read_module_field("left_gripper", "action_latency", 0.11) == 0.11
