"""Tests for I/O-free IK alignment math."""

import math

from solarian.core.ik.alignment import composed_rad, continuous_range_window


CPR = 4096
PI = math.pi


class TestContinuousRangeWindow:
    def test_zero_centered_at_mid_fits(self):
        win = continuous_range_window(CPR / 2, -PI, PI, CPR)
        assert win.needs_rehome is False
        assert win.range_min == 0
        assert win.range_max == CPR - 1

    def test_zero_centered_off_mid_asks_rehome(self):
        win = continuous_range_window(100, -PI, PI, CPR)
        assert win.needs_rehome is True
        assert win.range_min == 0
        assert win.range_max == CPR - 1

    def test_endpoint_zero_never_saves_half_span(self):
        # CAD wrist_roll [-2π, 0]: model 0 is at the high end. A naive
        # center-minus-span window overflows and used to clamp to 0..2047.
        for present in (100, CPR / 2, 3900):
            win = continuous_range_window(present, -2 * PI, 0.0, CPR)
            assert win.needs_rehome is True
            assert win.range_min == 0
            assert win.range_max == CPR - 1
            span = win.range_max - win.range_min
            assert span / (CPR - 1) > 0.98

    def test_home_maps_to_model_zero_when_centered(self):
        win = continuous_range_window(CPR / 2, -PI, PI, CPR)
        rad = composed_rad(CPR / 2, win.range_min, win.range_max, -PI, PI)
        assert abs(rad) < 0.01
