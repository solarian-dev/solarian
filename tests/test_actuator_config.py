"""Actuator yaml flatten, class-default merge, and surgical yaml patch."""

from __future__ import annotations

from pathlib import Path

from solarian.core.actuator import Actuator, merge_actuator_config
from solarian.core.configs import ActuatorConfig
from solarian.core.ik.actuator import CartesianIKActuator
from solarian.core.module_yaml import patch_actuator_fields, patch_top_level_mapping


def test_actuator_config_flattens_legacy_sim_block() -> None:
    cfg = ActuatorConfig.model_validate(
        {
            "id": 1,
            "port": "/dev/ttyUSB0",
            "type": "feetech",
            "model": "sts3215",
            "sim": {
                "joint": "right/shoulder_pan",
                "actuator": "right/shoulder_pan",
                "range_min": -1.5,
                "range_max": 1.5,
                "drive_mode": 1,
            },
        }
    )
    assert cfg.model == "sts3215"
    assert cfg.sim_joint == "right/shoulder_pan"
    assert cfg.sim_drive_mode == 1
    assert not hasattr(cfg, "type")


def test_actuator_config_reads_sim_followers() -> None:
    cfg = ActuatorConfig.model_validate(
        {
            "id": 1,
            "sim_joint": "gripper/left_manifold_to_rail",
            "sim_followers": [
                {
                    "joint": "gripper/right_manifold_to_rail",
                    "range_min": -0.0545,
                    "range_max": 0,
                }
            ],
        }
    )
    assert cfg.sim_followers[0].joint == "gripper/right_manifold_to_rail"
    assert cfg.sim_followers[0].drive_mode == 0


def test_joint_spec_carries_followers() -> None:
    from types import SimpleNamespace

    from solarian.sim.runtime import _joint_spec

    cfg = ActuatorConfig.model_validate(
        {
            "id": 1,
            "mode": "range_m100_100",
            "sim_joint": "gripper/left_manifold_to_rail",
            "sim_drive_mode": 1,
            "sim_followers": [
                {"joint": "gripper/right_manifold_to_rail"}
            ],
        }
    )
    spec = _joint_spec(SimpleNamespace(cfg=cfg))
    assert spec["joint"] == "gripper/left_manifold_to_rail"
    assert spec["followers"] == [
        {
            "joint": "gripper/right_manifold_to_rail",
            "actuator": "gripper/right_manifold_to_rail",
            "drive_mode": 0,
        }
    ]


def test_actuator_config_reads_legacy_homing_offset() -> None:
    cfg = ActuatorConfig.model_validate({"id": 1, "homing_offset": -776})
    assert cfg.center_offset == -776


def test_merge_uses_class_defaults_when_yaml_omits_keys() -> None:
    marker = Actuator(model="sts3250", is_continuous=True, mode="range_m100_100")
    yaml_cfg = ActuatorConfig.model_validate({"id": 4, "port": "/dev/ttyACM0"})
    merged = merge_actuator_config(marker, "wrist_flex", yaml_cfg, "/dev/ttyACM0")
    assert merged.model == "sts3250"
    assert merged.mode == "range_m100_100"
    assert merged.id == 4
    assert merged.port == "/dev/ttyACM0"


def test_merge_yaml_model_overrides_class_default() -> None:
    marker = Actuator(model="sts3215")
    yaml_cfg = ActuatorConfig.model_validate(
        {"id": 7, "port": "/dev/ttyACM1", "model": "sts3250"}
    )
    merged = merge_actuator_config(marker, "gripper", yaml_cfg, "/dev/ttyACM1")
    assert merged.model == "sts3250"


def test_merge_cartesian_without_yaml_uses_axis_id_and_hardware_port() -> None:
    marker = CartesianIKActuator()
    merged = merge_actuator_config(marker, "rz", None, "/dev/ttyACM0")
    assert merged.id == 5
    assert merged.mode == "raw"
    assert merged.port == "/dev/ttyACM0"


def test_patch_actuator_fields_replaces_and_appends(tmp_path: Path) -> None:
    path = tmp_path / "arm.yaml"
    path.write_text(
        """\
# keep this comment
actuators:
    shoulder_pan:
        id: 1
        port: /dev/ttyUSB0
        drive_mode: 1  # old value
    wrist_roll:
        id: 6
        port: /dev/ttyUSB0
""",
        encoding="utf-8",
    )
    updated = patch_actuator_fields(
        str(path),
        {
            "shoulder_pan": {
                "real_range_min": 452,
                "real_range_max": 3255,
                "center_offset": 0,
                "drive_mode": 0,
            },
            "wrist_roll": {
                "real_range_min": 0,
                "real_range_max": 4094,
                "center_offset": -1480,
                "drive_mode": 0,
            },
        },
    )
    assert updated == ["shoulder_pan", "wrist_roll"]
    text = path.read_text(encoding="utf-8")
    assert "# keep this comment" in text
    assert "drive_mode: 0" in text
    assert "# old value" in text
    assert "real_range_min: 452" in text
    assert "real_range_max: 3255" in text
    assert "center_offset: -1480" in text
    assert "wrist_roll:" in text


def test_patch_migrates_homing_offset_key(tmp_path: Path) -> None:
    path = tmp_path / "arm.yaml"
    path.write_text(
        """\
actuators:
    wrist_roll:
        id: 6
        homing_offset: 0
""",
        encoding="utf-8",
    )
    patch_actuator_fields(str(path), {"wrist_roll": {"center_offset": -400}})
    text = path.read_text(encoding="utf-8")
    assert "center_offset: -400" in text
    assert "homing_offset:" not in text


def test_patch_top_level_mapping_inserts_home_before_actuators(tmp_path: Path) -> None:
    path = tmp_path / "arm.yaml"
    path.write_text(
        """\
# keep this comment
frequency: 90
actuators:
    shoulder_pan:
        id: 1
""",
        encoding="utf-8",
    )
    patch_top_level_mapping(
        str(path),
        "home",
        {"shoulder_pan": 0.0, "wrist_roll": 1.5},
    )
    text = path.read_text(encoding="utf-8")
    assert "# keep this comment" in text
    assert "home:" in text
    assert "shoulder_pan: 0" in text
    assert "wrist_roll: 1.5" in text
    assert text.index("home:") < text.index("actuators:")

    patch_top_level_mapping(str(path), "home", {"shoulder_pan": 12.0})
    text = path.read_text(encoding="utf-8")
    assert "shoulder_pan: 12" in text
    assert "wrist_roll:" not in text.split("actuators:")[0]
