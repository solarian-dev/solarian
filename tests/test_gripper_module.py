"""Unit tests for GripperModule aperture conversion and scalar helpers."""

from types import SimpleNamespace

import numpy as np
import pytest

from solarian.core.aperture_table import ApertureTable
from solarian.core.gripper_module import GripperModule


def _module(**cfg) -> GripperModule:
    cfg.setdefault("aperture_table", None)
    module = GripperModule.__new__(GripperModule)
    module.cfg = SimpleNamespace(**cfg)
    return module


def test_scalar_limits_from_aperture_cfg():
    module = _module(aperture_closed_mm=0.0, aperture_open_mm=85.0)
    assert module.scalar_limits() == (0.0, 85.0)


def test_negative_closed_mm_maps_to_closed_norm():
    module = _module(aperture_closed_mm=-8.0, aperture_open_mm=85.0)

    assert module._scalar_to_actuator(-8.0) == -100.0
    assert module._scalar_to_actuator(85.0) == 100.0
    # Zero is a fraction of the way along a travel that starts below it.
    assert -90.0 < module._scalar_to_actuator(0.0) < -70.0
    assert module._actuator_to_scalar(-100.0) == pytest.approx(-8.0)
    assert module._actuator_to_scalar(100.0) == pytest.approx(85.0)


def test_linear_map_saturates_past_the_ends():
    module = _module(aperture_closed_mm=0.0, aperture_open_mm=85.0)
    assert module._scalar_to_actuator(-50.0) == -100.0
    assert module._scalar_to_actuator(999.0) == 100.0


def test_measured_table_supersedes_the_linear_map():
    # Compressed at the closed end, which is what the linkage actually does.
    table = ApertureTable(
        norm=[-100.0, -50.0, 0.0, 50.0, 100.0],
        aperture_mm=[0.0, 8.0, 25.0, 55.0, 85.0],
    )
    module = _module(
        aperture_closed_mm=0.0, aperture_open_mm=85.0, aperture_table=table
    )

    linear = _module(aperture_closed_mm=0.0, aperture_open_mm=85.0)
    assert module._scalar_to_actuator(25.0) == pytest.approx(0.0, abs=1e-6)
    assert linear._scalar_to_actuator(25.0) == pytest.approx(-41.18, abs=0.01)
    assert module._actuator_to_scalar(0.0) == pytest.approx(25.0)


def test_scalar_limits_come_from_the_table_when_present():
    table = ApertureTable(norm=[-100.0, 100.0], aperture_mm=[2.0, 70.0])
    module = _module(
        aperture_closed_mm=0.0, aperture_open_mm=85.0, aperture_table=table
    )
    assert module.scalar_limits() == (2.0, 70.0)


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
        np.array([t0]), np.array([85.0], dtype=np.float64)
    )
    module.last_waypoint_time = t0
    module.frequency = 120.0
    module._get_max_speed_kwargs = lambda: {"max_speed": 150.0}
    module._resolve_waypoint_time = lambda target, now: (
        target if target > now else now + 1.0 / 120.0
    )

    # Request 0.3 s; 85 → -8 at 150 mm/s needs 0.62 s.
    module._schedule_scalar_waypoint(-8.0, wall + 0.3, t_now=t0)
    assert module.last_waypoint_time == pytest.approx(t0 + (85.0 - -8.0) / 150.0)


def test_schedule_j_never_drops_timestamps():
    module = GripperModule.__new__(GripperModule)
    captured: dict = {}

    def fake(value, target_time_wall, t_now):
        captured["value"] = value
        captured["wall"] = target_time_wall
        captured["t_now"] = t_now

    module._schedule_scalar_waypoint = fake  # type: ignore[method-assign]
    module.scheduleJ(42.0, 123.0)
    assert captured["value"] == 42.0
    assert captured["wall"] == 123.0


def test_read_aperture_delegates_to_current():
    module = GripperModule.__new__(GripperModule)
    module._read_current_scalar = lambda: 32.5  # type: ignore[method-assign]
    assert module.read_aperture() == 32.5


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
