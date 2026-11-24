"""Unit tests for MJCF → config.json joint-limit sync (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from solarian.sim.cad import (
    clear_joint_property_limits,
    read_joint_ranges,
    sync_joint_properties,
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
