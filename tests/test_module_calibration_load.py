"""load_calibration reads yaml ranges; missing data stays None and must be required."""

import pytest

from solarian.core.actuator import Actuator
from solarian.core.actuator_module import ActuatorModule
from solarian.core.configs import ActuatorConfig, ModuleConfig


class _OneJoint(ActuatorModule):
    joint = Actuator()


def _module(act: dict, name: str = "leader_arm") -> _OneJoint:
    return _OneJoint(
        ModuleConfig(frequency=30.0, actuators={"joint": ActuatorConfig.model_validate(act)}),
        name=name,
    )


def test_load_calibration_from_yaml_ranges() -> None:
    module = _module(
        {
            "id": 1,
            "port": "/dev/null",
            "real_range_min": 1023,
            "real_range_max": 3070,
            "center_offset": -15,
            "drive_mode": 1,
        }
    )
    module.load_calibration()
    assert module.calibration is not None
    cal = module.calibration["joint"]
    assert cal.range_min == 1023
    assert cal.range_max == 3070
    assert cal.homing_offset == -15
    assert cal.drive_mode == 1
    grouped = module.require_calibration()
    assert "joint" in grouped["/dev/null"]


def test_load_calibration_missing_ranges_is_none_and_require_raises() -> None:
    module = _module({"id": 1, "port": "/dev/null"})
    module.load_calibration()
    assert module.calibration is None
    with pytest.raises(RuntimeError, match="No calibration for module 'leader_arm'"):
        module.require_calibration()
