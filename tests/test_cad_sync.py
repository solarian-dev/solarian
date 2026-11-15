"""Unit tests for MJCF → YAML / config.json joint-limit sync (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from solarian.sim.cad import (
    clear_joint_property_limits,
    read_joint_ranges,
    sync_joint_properties,
    sync_limits_from_mjcf,
    sync_module_yaml,
)

SAMPLE_MJCF = """\
<?xml version='1.0' encoding='utf-8'?>
<mujoco model="test">
  <worldbody>
    <body name="base">
      <joint name="shoulder_pan" type="hinge" range="-2.10999 2.10999" />
      <body name="link2">
        <joint name="shoulder_lift" type="hinge" range="-1.8099 1.9949" />
        <joint name="no_range" type="hinge" />
      </body>
    </body>
  </worldbody>
</mujoco>
"""

SAMPLE_YAML = """\
# module conf — keep this comment
frequency: 90
actuators:
    shoulder_pan:
        id: 1
        mode: range_m100_100
        sim_joint: "right/shoulder_pan"
        sim_range_min: -1.0  # old value
        sim_range_max: 1.0
    shoulder_lift:
        id: 2
        mode: range_m100_100
        sim_joint: "right/shoulder_lift"
        sim_range_min: 0.0
        sim_range_max: 0.5
    orphan_joint:
        id: 3
        mode: range_m100_100
        sim_joint: "right/orphan_joint"
        sim_range_min: -0.1
        sim_range_max: 0.1
"""

SAMPLE_CONFIG = {
    "robot_name": "test",
    "joint_properties": {
        "shoulder_pan": {"limits": [-1.0, 1.0], "range": True},
        "shoulder_lift": {"limits": [0.0, 0.5], "range": True},
        "extra_unused": {"limits": [-9.0, 9.0], "range": True},
    },
    "solarian": {"dof_mates": ["shoulder_pan", "shoulder_lift"]},
}


@pytest.fixture
def mjcf_path(tmp_path: Path) -> Path:
    path = tmp_path / "model.xml"
    path.write_text(SAMPLE_MJCF, encoding="utf-8")
    return path


@pytest.fixture
def yaml_path(tmp_path: Path) -> Path:
    path = tmp_path / "right_arm.yaml"
    path.write_text(SAMPLE_YAML, encoding="utf-8")
    return path


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(SAMPLE_CONFIG, indent=4) + "\n", encoding="utf-8")
    return path


def test_read_joint_ranges(mjcf_path: Path) -> None:
    ranges = read_joint_ranges(str(mjcf_path))
    assert ranges == {
        "shoulder_pan": (-2.10999, 2.10999),
        "shoulder_lift": (-1.8099, 1.9949),
    }
    assert "no_range" not in ranges


def test_sync_module_yaml_preserves_comments_and_updates_ranges(
    yaml_path: Path, mjcf_path: Path
) -> None:
    ranges = read_joint_ranges(str(mjcf_path))
    updated = sync_module_yaml(str(yaml_path), ranges)

    assert updated == ["shoulder_pan", "shoulder_lift"]
    text = yaml_path.read_text(encoding="utf-8")

    assert "# module conf — keep this comment" in text
    assert "sim_range_min: -1.0  # old value" not in text
    assert "sim_range_min: -2.10999" in text
    assert "# old value" in text
    assert "sim_range_max: 2.10999" in text
    assert "sim_range_min: -1.8099" in text
    assert "sim_range_max: 1.9949" in text
    # Unmatched sim actuator left alone
    assert "orphan_joint:" in text
    assert "sim_range_min: -0.1" in text
    assert 'sim_joint: "right/shoulder_pan"' in text


def test_sync_module_yaml_matches_by_actuator_key(tmp_path: Path) -> None:
    """When sim_joint is absent, the actuator key itself is the joint name."""
    yaml_text = """\
actuators:
    shoulder_pan:
        id: 1
        sim_range_min: 0
        sim_range_max: 1
"""
    path = tmp_path / "arm.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    updated = sync_module_yaml(str(path), {"shoulder_pan": (-2.5, 2.5)})
    assert updated == ["shoulder_pan"]
    text = path.read_text(encoding="utf-8")
    assert "sim_range_min: -2.5" in text
    assert "sim_range_max: 2.5" in text


def test_sync_module_yaml_skips_actuators_without_sim(tmp_path: Path) -> None:
    yaml_text = """\
actuators:
    shoulder_pan:
        id: 1
        port: /dev/ttyUSB0
"""
    path = tmp_path / "arm.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    updated = sync_module_yaml(str(path), {"shoulder_pan": (-2.5, 2.5)})
    assert updated == []
    assert path.read_text(encoding="utf-8") == yaml_text


def test_sync_joint_properties(config_path: Path, mjcf_path: Path) -> None:
    ranges = read_joint_ranges(str(mjcf_path))
    updated = sync_joint_properties(str(config_path), ranges)

    assert set(updated) == {"shoulder_pan", "shoulder_lift"}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["joint_properties"]["shoulder_pan"]["limits"] == [-2.10999, 2.10999]
    assert data["joint_properties"]["shoulder_lift"]["limits"] == [-1.8099, 1.9949]
    # Existing entry with no MJCF joint is left alone
    assert data["joint_properties"]["extra_unused"]["limits"] == [-9.0, 9.0]
    assert data["joint_properties"]["shoulder_pan"]["range"] is True


def test_sync_limits_from_mjcf(
    mjcf_path: Path, yaml_path: Path, config_path: Path
) -> None:
    result = sync_limits_from_mjcf(
        str(mjcf_path), yaml_path=str(yaml_path), config_path=str(config_path)
    )
    assert result["yaml"] == ["shoulder_pan", "shoulder_lift"]
    assert set(result["config"]) == {"shoulder_pan", "shoulder_lift"}
    assert result["unmatched_mjcf"] == []
    assert result["unmatched_yaml"] == ["orphan_joint"]


def test_sync_limits_from_mjcf_config_only(mjcf_path: Path, config_path: Path) -> None:
    """Without a yaml path, only config.json is refreshed from the MJCF."""
    result = sync_limits_from_mjcf(str(mjcf_path), config_path=str(config_path))
    assert result["yaml"] == []
    assert set(result["config"]) == {"shoulder_pan", "shoulder_lift"}
    assert result["unmatched_mjcf"] == []
    assert result["unmatched_yaml"] == []
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["joint_properties"]["shoulder_pan"]["limits"] == [-2.10999, 2.10999]


def test_clear_joint_property_limits() -> None:
    jp = {
        "wrist_flex": {"limits": [-2.6, 0.4], "range": True},
        "shoulder_pan": {"range": True},
        "default": {"limits": [-3.0, 3.0], "max_effort": 10.0},
    }
    cleared = clear_joint_property_limits(jp)
    assert set(cleared) == {"wrist_flex", "default"}
    assert "limits" not in jp["wrist_flex"]
    assert jp["wrist_flex"]["range"] is True
    assert "limits" not in jp["default"]
    assert jp["default"]["max_effort"] == 10.0
    assert jp["shoulder_pan"] == {"range": True}


def test_sync_module_yaml_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        sync_module_yaml("/nonexistent/path.yaml", {"a": (0.0, 1.0)})
