"""Unit tests for the measured aperture <-> normalized servo interpolation."""

import pytest
from pydantic import ValidationError

from solarian.core.aperture_table import ApertureTable

# Compressed at the closed end: the shape a parallel gripper's linkage produces,
# and the reason a straight line between the endpoints is not good enough.
_MEASURED = ApertureTable(
    norm=[-100.0, -50.0, 0.0, 50.0, 100.0],
    aperture_mm=[0.0, 8.0, 25.0, 55.0, 85.0],
)


@pytest.mark.parametrize("mm", [0.0, 8.0, 25.0, 55.0, 85.0])
def test_measurements_round_trip_exactly(mm):
    assert _MEASURED.to_aperture(_MEASURED.to_norm(mm)) == pytest.approx(mm)


@pytest.mark.parametrize("mm", [3.0, 17.0, 40.0, 70.0])
def test_interpolated_points_round_trip(mm):
    assert _MEASURED.to_aperture(_MEASURED.to_norm(mm)) == pytest.approx(mm, abs=1e-6)


def test_interpolation_follows_the_measurements_not_a_straight_line():
    # 40 mm sits near the open end of a curve that is stretched there, so the
    # servo command is far from where linear interpolation would put it.
    linear = (40.0 / 85.0) * 200.0 - 100.0
    assert _MEASURED.to_norm(40.0) == pytest.approx(26.61, abs=0.05)
    assert abs(_MEASURED.to_norm(40.0) - linear) > 30.0


def test_pchip_stays_monotone_between_knots():
    previous = -101.0
    for step in range(0, 86):
        norm = _MEASURED.to_norm(float(step))
        assert norm > previous
        previous = norm


def test_commands_past_the_measured_travel_saturate():
    assert _MEASURED.to_norm(-25.0) == pytest.approx(-100.0)
    assert _MEASURED.to_norm(1_000.0) == pytest.approx(100.0)
    assert _MEASURED.to_aperture(-1_000.0) == pytest.approx(0.0)
    assert _MEASURED.to_aperture(1_000.0) == pytest.approx(85.0)


def test_limits_report_the_measured_travel():
    assert _MEASURED.limits == (0.0, 85.0)


def test_descending_measurements_are_accepted():
    """A gripper whose norm opens as the aperture closes sweeps downwards."""
    table = ApertureTable(norm=[-100.0, 0.0, 100.0], aperture_mm=[85.0, 25.0, 0.0])
    assert table.to_norm(85.0) == pytest.approx(-100.0)
    assert table.to_norm(0.0) == pytest.approx(100.0)
    assert table.limits == (0.0, 85.0)


def test_aperture_that_turns_back_is_rejected():
    with pytest.raises(ValidationError, match="strictly monotone"):
        ApertureTable(norm=[-100.0, 0.0, 100.0], aperture_mm=[0.0, 40.0, 30.0])


def test_repeated_aperture_is_rejected():
    with pytest.raises(ValidationError, match="strictly monotone"):
        ApertureTable(norm=[-100.0, 0.0, 100.0], aperture_mm=[0.0, 40.0, 40.0])


def test_a_single_measurement_is_rejected():
    with pytest.raises(ValidationError, match="at least 2"):
        ApertureTable(norm=[-100.0], aperture_mm=[0.0])


def test_mismatched_series_lengths_are_rejected():
    with pytest.raises(ValidationError, match="one aperture per servo command"):
        ApertureTable(norm=[-100.0, 0.0], aperture_mm=[0.0])
