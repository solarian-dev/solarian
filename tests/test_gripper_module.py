"""Unit tests for GripperModule scalar helpers used by the calibrate commands."""

from types import SimpleNamespace

import numpy as np
import pytest

from solarian.cli.commands.calibrate_gripper_ranges import _aperture_ruler_steps
from solarian.core.gripper_module import GripperModule


def test_aperture_ruler_starts_at_closed_when_closed_is_negative():
    steps = _aperture_ruler_steps(-0.8, 8.5)
    assert steps[0] == -0.8
    assert 0.0 in steps
    assert steps[-1] == 8.5


def test_negative_closed_cm_maps_to_closed_norm():
    from solarian_umi_slim.aperture import aperture_to_norm, norm_to_aperture

    assert aperture_to_norm(-0.8, -0.8, 8.5) == -100.0
    assert aperture_to_norm(8.5, -0.8, 8.5) == 100.0
    zero = aperture_to_norm(0.0, -0.8, 8.5)
    assert -90.0 < zero < -70.0
    assert abs(norm_to_aperture(-100.0, -0.8, 8.5) - (-0.8)) < 1e-9


def test_aperture_ruler_starts_at_zero_when_closed_is_zero():
    steps = _aperture_ruler_steps(0.0, 8.5)
    assert steps[0] == 0.0
    assert steps[-1] == 8.5


def test_scalar_limits_from_aperture_cfg():
    module = GripperModule.__new__(GripperModule)
    module.cfg = SimpleNamespace(aperture_closed_cm=0.0, aperture_open_cm=8.5)
    module.open_value = None
    assert module.scalar_limits() == (0.0, 8.5)


def test_scalar_limits_fall_back_to_the_declared_open_value():
    module = GripperModule.__new__(GripperModule)
    module.cfg = SimpleNamespace()
    module.open_value = 1.0
    assert module.scalar_limits() == (0.0, 1.0)


def test_aperture_unit_suffix_when_open_cm_set():
    module = GripperModule.__new__(GripperModule)
    module.cfg = SimpleNamespace(aperture_open_cm=8.5)
    assert module.aperture_unit_suffix() == " (cm)"
    module.cfg = SimpleNamespace()
    assert module.aperture_unit_suffix() == ""


def test_schedule_j_records_speed_limited_arrival(monkeypatch):
    import solarian.core.actuator_module as am
    from solarian.core.actuator_module import ActuatorModule
    from solarian.utils.trajectory_interpolator import ScalarTrajectoryInterpolator

    t0 = 10.0
    wall = 1_000_000.0
    monkeypatch.setattr(am.time, "monotonic", lambda: t0)
    monkeypatch.setattr(am, "synced_time", lambda: wall)

    module = ActuatorModule.__new__(ActuatorModule)
    module.interp = ScalarTrajectoryInterpolator(
        np.array([t0]), np.array([8.5], dtype=np.float64)
    )
    module.last_waypoint_time = t0
    module.frequency = 120.0
    module._get_max_speed_kwargs = lambda: {"max_speed": 15.0}
    module._resolve_waypoint_time = lambda target, now: (
        target if target > now else now + 1.0 / 120.0
    )

    # Request 0.3 s; 8.5 → -0.8 at 15 cm/s needs 0.62 s.
    module._schedule_scalar_waypoint(-0.8, wall + 0.3, t_now=t0)
    assert module.last_waypoint_time == pytest.approx(t0 + (8.5 - -0.8) / 15.0)


def test_schedule_j_never_drops_timestamps():
    module = GripperModule.__new__(GripperModule)
    captured: dict = {}

    def fake(value, target_time_wall, t_now):
        captured["value"] = value
        captured["wall"] = target_time_wall
        captured["t_now"] = t_now

    module._schedule_scalar_waypoint = fake  # type: ignore[method-assign]
    module.scheduleJ(4.2, 123.0)
    assert captured["value"] == 4.2
    assert captured["wall"] == 123.0


def test_read_scalar_delegates_to_current():
    module = GripperModule.__new__(GripperModule)
    module._read_current_scalar = lambda: 3.25  # type: ignore[method-assign]
    assert module.read_scalar() == 3.25


def test_flip_drive_modes_toggles_yaml_and_bus_cache():
    from solarian.core.actuator_module import ActuatorModule
    from solarian.core.calibration import MotorCalibration

    module = ActuatorModule.__new__(ActuatorModule)
    entry = MotorCalibration(
        id=7, drive_mode=1, homing_offset=589, range_min=3059, range_max=4095
    )
    module.calibration = {"gripper": entry}
    wrote = []
    module._save_calibration = lambda cal: wrote.append(cal["gripper"].drive_mode)
    bus = SimpleNamespace(calibration={})
    module._hardware_actuators_by_port = lambda: {"port": ["gripper"]}
    module.module_bus = SimpleNamespace(component_bus_by_port={"port": bus})
    module._flip_drive_modes(["gripper"])
    assert entry.drive_mode == 0
    assert wrote == [0]
    assert bus.calibration["gripper"].drive_mode == 0
    module._flip_drive_modes(["gripper"])
    assert entry.drive_mode == 1
