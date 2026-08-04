"""Tests for generic joint calibration helpers."""

import pytest

from solarian.core.joint_calibration import (
    CalibrationSafetyError,
    EncoderRangeTracker,
    MAX_SPAN_FRACTION_OF_CPR,
    calibration_from_raw_range,
    derive_homing_from_raw_range,
    drive_mode_from_low_end_pose,
    low_command_label,
    validate_contiguous_raw_range,
)


CPR = 4096


class TestCalibrationFromRawRange:
    def test_keeps_homing_offset_zero(self):
        d = calibration_from_raw_range("j", 2700, 3700, CPR)
        assert d.homing_offset == 0
        assert d.range_min == 2700
        assert d.range_max == 3700
        assert d.mid_raw == 3200

    def test_near_top_of_encoder_ok(self):
        # Travel near CPR without remounting — offset stays 0.
        d = calibration_from_raw_range("gripper", 3500, 4000, CPR)
        assert d.homing_offset == 0
        assert d.range_min == 3500
        assert d.range_max == 4000

    def test_near_zero_ok(self):
        d = calibration_from_raw_range("gripper", 50, 1050, CPR)
        assert d.homing_offset == 0

    def test_rejects_huge_span(self):
        raw_min, raw_max = 100, 100 + int(MAX_SPAN_FRACTION_OF_CPR * CPR) + 10
        with pytest.raises(CalibrationSafetyError, match="wrap"):
            validate_contiguous_raw_range("j", raw_min, raw_max, CPR)

    def test_rejects_inverted_range(self):
        with pytest.raises(CalibrationSafetyError, match="invalid raw range"):
            calibration_from_raw_range("j", 2000, 1000, CPR)

    def test_legacy_alias_still_works(self):
        d = derive_homing_from_raw_range("j", 2700, 3700, CPR, max_homing_abs=100)
        assert d.homing_offset == 0


class TestEncoderRangeTracker:
    def test_ignores_opposite_side_glitch(self):
        t = EncoderRangeTracker(cpr=CPR)
        for p in (3900, 3850, 3800, 3750):
            assert t.update(p)
        # Spurious read at 0 would make span look like a full wrap — reject it.
        assert t.update(0) is False
        assert t.rejected == 1
        lo, hi = t.result()
        assert lo == 3750
        assert hi == 3900

    def test_accepts_normal_sweep_near_top(self):
        t = EncoderRangeTracker(cpr=CPR)
        for p in range(3500, 4010, 20):
            assert t.update(p)
        lo, hi = t.result()
        assert lo == 3500
        assert hi == 4000


class TestDriveMode:
    def test_pose_near_min_is_drive_mode_0(self):
        assert drive_mode_from_low_end_pose(1510, 1500, 2500) == 0

    def test_pose_near_max_is_drive_mode_1(self):
        assert drive_mode_from_low_end_pose(2490, 1500, 2500) == 1

    def test_tie_prefers_min(self):
        mid = (1500 + 2500) / 2
        assert drive_mode_from_low_end_pose(mid, 1500, 2500) == 0


class TestLabels:
    def test_range_0_100(self):
        assert "0" in low_command_label("range_0_100")

    def test_range_m100_100(self):
        assert low_command_label("range_m100_100") == "-100"
