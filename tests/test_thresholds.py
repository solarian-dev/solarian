"""Tests for observation value-threshold bands (GUI warn/high coloring)."""

from solarian.core.data import TemperatureData, threshold_exceeded, threshold_level
from solarian.core.observation import SensorField


def test_one_sided_band():
    assert not threshold_exceeded(49.9, 50.0)
    assert threshold_exceeded(50.0, 50.0)
    assert threshold_exceeded(70.0, 50.0)
    assert not threshold_exceeded(0.0, None)


def test_two_sided_band():
    assert not threshold_exceeded(0.0, (-90.0, 90.0))
    assert threshold_exceeded(-90.0, (-90.0, 90.0))
    assert threshold_exceeded(90.0, (-90.0, 90.0))
    assert not threshold_exceeded(89.9, (-90.0, 90.0))


def test_high_wins_over_mid():
    assert threshold_level(55.0, 50.0, 60.0) == "mid"
    assert threshold_level(60.0, 50.0, 60.0) == "high"
    assert threshold_level(40.0, 50.0, 60.0) is None
    assert threshold_level(99.0, (-90.0, 90.0), (-98.0, 98.0)) == "high"
    assert threshold_level(95.0, (-90.0, 90.0), (-98.0, 98.0)) == "mid"


def test_temperature_data_defaults():
    assert TemperatureData.mid == 50.0
    assert TemperatureData.high == 60.0
    assert threshold_level(55.0, TemperatureData.mid, TemperatureData.high) == "mid"


def test_sensor_field_inherits_and_overrides_data_thresholds():
    src = TemperatureData()
    src.path = ("right_arm", "shoulder_lift", "temperature")
    inherited = SensorField(src)
    assert inherited.mid == 50.0
    assert inherited.high == 60.0
    assert inherited.threshold_level(55.0) == "mid"
    assert inherited.threshold_level(61.0) == "high"

    override = SensorField(src, mid=40.0, high=70.0)
    assert override.mid == 40.0
    assert override.high == 70.0
    assert override.threshold_level(55.0) == "mid"
    assert override.threshold_level(61.0) == "mid"
    assert override.threshold_level(70.0) == "high"
