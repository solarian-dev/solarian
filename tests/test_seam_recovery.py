"""Seam-crossing sweeps are rejected, then recovered by centering the shaft."""

import builtins
from types import SimpleNamespace

import pytest

from solarian.core.actuator_module import ActuatorModule
from solarian.core.joint_calibration import (
    CalibrationSafetyError,
    EncoderRangeTracker,
    SeamCrossingError,
)

CPR = 4096


class _Bus:
    def __init__(self):
        self.centered: list[list[str]] = []

    def drive_to_encoder_center(self, component_names):
        self.centered.append(list(component_names))
        return {n: (CPR - 1) // 2 for n in component_names}


def _module(buses: dict):
    module = ActuatorModule.__new__(ActuatorModule)
    module.module_bus = SimpleNamespace(component_bus_by_port=buses)
    return module


def _answers(monkeypatch, replies):
    queue = list(replies)

    def _input(prompt=""):
        del prompt
        assert queue, "flow asked for more input than expected"
        return queue.pop(0)

    monkeypatch.setattr(builtins, "input", _input)


def _seam_sweep(cpr: int = CPR) -> EncoderRangeTracker:
    """A gripper sweep like the arm1 right_gripper: 3645 → 4095 → 0 → 598."""
    tracker = EncoderRangeTracker(cpr=cpr)
    for pos in list(range(3645, 4096, 50)) + list(range(0, 599, 50)):
        tracker.update(pos % cpr)
    return tracker


class TestRejection:
    def test_seam_sweep_reports_span_and_mid_travel(self):
        with pytest.raises(SeamCrossingError) as excinfo:
            _seam_sweep().derived("gripper")
        exc = excinfo.value
        # Travel 3645..4095..0..550 unwrapped, so the span stays under a turn
        # and mid-travel sits near the seam rather than near 2048.
        assert 900 < exc.span < CPR
        assert exc.mid_raw > 3800 or exc.mid_raw < 300
        assert exc.name == "gripper"

    def test_contiguous_sweep_next_to_the_seam_is_accepted(self):
        tracker = EncoderRangeTracker(cpr=CPR)
        for pos in range(3000, 4096, 50):
            tracker.update(pos)
        d = tracker.derived("gripper")
        assert d.center_offset == 0
        assert (d.range_min, d.range_max) == (3000, 4050)


class TestRecovery:
    def test_centers_only_the_offending_joints(self, monkeypatch):
        bus = _Bus()
        module = _module({"/dev/a": bus})
        errors = {"gripper": SeamCrossingError("gripper", 1049, 4030, CPR)}
        _answers(monkeypatch, ["y", ""])

        assert module._recover_seam_crossing(
            errors, {"/dev/a": ["shoulder", "gripper"]}
        )
        assert bus.centered == [["gripper"]]

    def test_declining_aborts_without_touching_the_bus(self, monkeypatch):
        bus = _Bus()
        module = _module({"/dev/a": bus})
        errors = {"gripper": SeamCrossingError("gripper", 1049, 4030, CPR)}
        _answers(monkeypatch, [""])

        assert not module._recover_seam_crossing(errors, {"/dev/a": ["gripper"]})
        assert bus.centered == []

    def test_each_port_is_centered_in_one_pass(self, monkeypatch):
        left, right = _Bus(), _Bus()
        module = _module({"/dev/left": left, "/dev/right": right})
        errors = {
            "elbow": SeamCrossingError("elbow", 900, 10, CPR),
            "wrist": SeamCrossingError("wrist", 900, 10, CPR),
        }
        _answers(monkeypatch, ["y", ""])

        assert module._recover_seam_crossing(
            errors, {"/dev/left": ["elbow"], "/dev/right": ["wrist", "pan"]}
        )
        assert left.centered == [["elbow"]]
        assert right.centered == [["wrist"]]


def test_seam_error_is_catchable_as_a_safety_error():
    # calibrate()'s outer handler must still abort if recovery is not wired up.
    with pytest.raises(CalibrationSafetyError):
        raise SeamCrossingError("j", 1049, 4030, CPR)
