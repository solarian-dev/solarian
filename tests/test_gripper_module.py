"""Unit tests for GripperModule scalar helpers used by calibrate-gripper."""

from types import SimpleNamespace

from solarian.core.gripper_module import GripperModule


def test_scalar_limits_from_aperture_cfg():
    module = GripperModule.__new__(GripperModule)
    module.cfg = SimpleNamespace(aperture_closed_cm=0.0, aperture_open_cm=8.5)
    module.home_value = None
    assert module.scalar_limits() == (0.0, 8.5)


def test_scalar_limits_fallback_home():
    module = GripperModule.__new__(GripperModule)
    module.cfg = SimpleNamespace()
    module.home_value = 1.0
    assert module.scalar_limits() == (0.0, 1.0)


def test_aperture_unit_suffix_when_open_cm_set():
    module = GripperModule.__new__(GripperModule)
    module.cfg = SimpleNamespace(aperture_open_cm=8.5)
    assert module._aperture_status_suffix() == " (cm)"
    module.cfg = SimpleNamespace()
    assert module._aperture_status_suffix() == ""


def test_schedule_scalar_never_drops_timestamps():
    module = GripperModule.__new__(GripperModule)
    captured: dict = {}

    def fake(value, target_time_wall, t_now, drop_timestamps=False):
        captured["value"] = value
        captured["wall"] = target_time_wall
        captured["drop"] = drop_timestamps
        captured["t_now"] = t_now

    module._schedule_scalar_waypoint = fake  # type: ignore[method-assign]
    module.schedule_scalar(4.2, 123.0)
    assert captured["value"] == 4.2
    assert captured["wall"] == 123.0
    assert captured["drop"] is False


def test_read_scalar_delegates_to_current():
    module = GripperModule.__new__(GripperModule)
    module._read_current_scalar = lambda: 3.25  # type: ignore[method-assign]
    assert module.read_scalar() == 3.25
