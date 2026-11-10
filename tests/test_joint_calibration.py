"""Tests for generic joint calibration helpers."""

import pytest

from solarian.core.joint_calibration import (
    CalibrationSafetyError,
    EncoderRangeTracker,
    SeamCrossingError,
    calibration_from_circular_arc,
    calibration_from_raw_range,
    circular_distance,
    derive_homing_from_raw_range,
    drive_mode_from_low_end_pose,
    low_command_label,
    shortest_path_crosses_wrap,
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

    def test_accepts_large_span(self):
        # Sweep defines the limits — no span-vs-CPR cap.
        d = calibration_from_raw_range("j", 100, 100 + int(0.80 * CPR), CPR)
        assert d.range_min == 100
        assert d.range_max == 100 + int(0.80 * CPR)

    def test_rejects_inverted_range(self):
        with pytest.raises(CalibrationSafetyError, match="invalid raw range"):
            calibration_from_raw_range("j", 2000, 1000, CPR)

    def test_rejects_out_of_bounds(self):
        with pytest.raises(CalibrationSafetyError, match="outside encoder"):
            validate_contiguous_raw_range("j", -1, 1000, CPR)

    def test_legacy_alias_still_works(self):
        d = derive_homing_from_raw_range("j", 2700, 3700, CPR, max_homing_abs=100)
        assert d.homing_offset == 0


class TestCircularHelpers:
    def test_circular_distance_across_wrap(self):
        assert circular_distance(4000, 50, CPR) == (50 - 4000) % CPR

    def test_shortest_path_crosses_wrap(self):
        assert shortest_path_crosses_wrap(3900, 0, CPR) is True
        assert shortest_path_crosses_wrap(3900, 3800, CPR) is False
        assert shortest_path_crosses_wrap(100, 200, CPR) is False
        assert shortest_path_crosses_wrap(50, 4000, CPR) is True


class TestEncoderRangeTracker:
    def test_ignores_opposite_side_glitch(self):
        t = EncoderRangeTracker(cpr=CPR)
        for p in (3900, 3850, 3800, 3750):
            assert t.update(p)
        assert t.update(0) is True
        d = t.derived("j")
        assert d.center_offset == 0
        assert d.range_min == 3750
        assert d.range_max == 3900
        assert t.wrap_rejected >= 1

    def test_ignores_far_teleport(self):
        t = EncoderRangeTracker(cpr=CPR)
        assert t.update(2000)
        # Jump halfway around the circle in one sample.
        assert t.update(2000 + int(0.5 * CPR)) is False
        assert t.rejected == 1
        assert t.raw_min == t.raw_max == 2000

    def test_accepts_normal_sweep_near_top(self):
        t = EncoderRangeTracker(cpr=CPR)
        for p in range(3500, 4010, 20):
            assert t.update(p)
        lo, hi = t.result()
        assert lo == 3500
        assert hi == 4000

    def test_accepts_large_contiguous_travel(self):
        # Solarian Arm 1-class arms need large fractions of a turn (e.g. pan ≈ 69%).
        t = EncoderRangeTracker(cpr=CPR)
        lo, hi = 500, 500 + int(0.70 * CPR)
        for p in list(range(lo, hi, 50)) + [hi]:
            assert t.update(p), f"rejected contiguous sample {p} in {lo}..{hi}"
        got_lo, got_hi = t.result()
        assert got_lo == lo
        assert got_hi == hi

    def test_single_wrap_sample_does_not_unwrap(self):
        t = EncoderRangeTracker(cpr=CPR)
        for p in range(4000, 4090, 10):
            assert t.update(p)
        assert t.update(20) is True
        d = t.derived("j")
        assert d.center_offset == 0
        assert d.range_min == 4000
        assert d.range_max == 4080
        assert t.wrap_rejected >= 1

    def test_wrapping_sweep_is_rejected(self):
        t = EncoderRangeTracker(cpr=CPR)
        path = [3900, 3950, 4000, 4050, 10, 50, 100, 150, 200]
        for p in path:
            assert t.update(p), f"rejected {p}"
        with pytest.raises(SeamCrossingError) as excinfo:
            t.derived("j")
        exc = excinfo.value
        assert exc.span == 200 + CPR - 3900
        assert exc.mid_raw == (3900 + 200 + CPR) // 2 % CPR
        assert exc.cpr == CPR

    def test_live_window_pos_stays_between_min_max_across_seam(self):
        t = EncoderRangeTracker(cpr=CPR)
        path = [3900, 3950, 4000, 4050, 10, 50, 100, 150, 200]
        for p in path:
            assert t.update(p), f"rejected {p}"
            window = t.live_window()
            assert window is not None
            lo, pos, hi = window
            assert lo <= pos <= hi, f"raw {p} → window {lo}..{pos}..{hi}"

    def test_live_window_contiguous_matches_raw(self):
        t = EncoderRangeTracker(cpr=CPR)
        for p in range(3500, 4010, 20):
            assert t.update(p)
            lo, pos, hi = t.live_window()
            assert lo == 3500
            assert pos == p
            assert hi == p

    def test_glitch_at_zero_does_not_unwrap(self):
        t = EncoderRangeTracker(cpr=CPR)
        for p in (3900, 3850, 3800, 3750):
            assert t.update(p)
        assert t.update(0) is True
        d = t.derived("j")
        assert d.center_offset == 0
        assert d.range_min == 3750
        assert d.range_max == 3900


class TestCalibrationFromCircularArc:
    def test_already_linear_keeps_offset_zero(self):
        d = calibration_from_circular_arc("j", 1000, 2000, CPR)
        assert d.center_offset == 0
        assert d.range_min == 1000
        assert d.range_max == 2000

    def test_wrapping_arc_raises_seam_crossing(self):
        with pytest.raises(SeamCrossingError) as excinfo:
            calibration_from_circular_arc("j", 3900, 3900 + 400, CPR)
        exc = excinfo.value
        assert exc.span == 400
        assert exc.mid_raw == (3900 + 200) % CPR
        assert "encoder seam" in str(exc)

    def test_seam_crossing_is_a_safety_error(self):
        # calibrate()'s outer handler still aborts on it if nobody recovers.
        assert issubclass(SeamCrossingError, CalibrationSafetyError)

    def test_full_turn_raises(self):
        with pytest.raises(CalibrationSafetyError, match="full turn"):
            calibration_from_circular_arc("j", 0, CPR, CPR)


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
