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


class TestOffsetTicks:
    """Shifting a calibration by offset_ticks adds the asked radians to every read."""

    def test_hard_stop_joint_reads_delta_more_after_shifting_its_range(self):
        from solarian.core.calibration import MotorCalibration
        from solarian.core.ik.alignment import offset_ticks

        lo, hi = -1.7987712336903257, 1.7802358370342863
        for drive_mode in (0, 1):
            cal = MotorCalibration(id=2, drive_mode=drive_mode, homing_offset=0,
                                   range_min=1011, range_max=3359)
            per_tick = (hi - lo) / (cal.range_max - cal.range_min)
            for delta in (0.035, -0.021):
                ticks, applied = offset_ticks(cal, lo, hi, delta)
                assert abs(applied - delta) <= per_tick / 2
                for raw in (1500, 2200, 3000):
                    before = composed_rad(raw, cal.range_min, cal.range_max, lo, hi, drive_mode)
                    after = composed_rad(raw, cal.range_min + ticks, cal.range_max + ticks,
                                         lo, hi, drive_mode)
                    assert abs(after - before - applied) < 1e-9

    def test_continuous_joint_reads_delta_more_after_shifting_its_offset(self):
        from solarian.core.calibration import MotorCalibration
        from solarian.core.ik.alignment import offset_ticks

        lo, hi = -PI, PI

        def read(raw, cal):
            present = (raw - cal.homing_offset) % CPR
            return composed_rad(present, cal.range_min, cal.range_max, lo, hi, cal.drive_mode)

        for drive_mode in (0, 1):
            cal = MotorCalibration(id=4, drive_mode=drive_mode, homing_offset=-414,
                                   range_min=0, range_max=CPR - 1)
            for delta in (0.035, -0.021):
                ticks, applied = offset_ticks(cal, lo, hi, delta)
                assert abs(applied - delta) <= PI / CPR
                shifted = MotorCalibration(id=4, drive_mode=drive_mode,
                                           homing_offset=cal.homing_offset + ticks,
                                           range_min=0, range_max=CPR - 1)
                for raw in (1200, 1650, 2500):
                    assert abs(read(raw, shifted) - read(raw, cal) - applied) < 1e-9
