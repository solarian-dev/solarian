"""The hand-positioning sequence shared by the calibrate-arm-* commands.

These are ordering tests. A servo arm falls the moment torque is released, so
*when* each step happens is the whole correctness question.
"""

from __future__ import annotations

import numpy as np
import pytest

from solarian.cli.commands import _calibration


class _StubArm:
    frequency = 90.0
    last_waypoint_time = 0.0

    def __init__(self):
        self.log: list[str] = []
        self.pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])

    def _record(self, name):
        self.log.append(name)

    def read_pose(self):
        self._record("read_pose")
        return self.pose

    def start_calibration_mode(self):
        self._record("torque_off")

    def hold_measured_pose(self):
        self._record("latch")
        return self.pose

    def end_calibration_mode(self):
        self._record("end_calibration_mode")

    def enable_torque(self):
        self._record("torque_on")

    def enable_torque_holding(self):
        self.hold_measured_pose()
        self.enable_torque()
        return self.pose

    def moveL(self, pose, speed):
        self._record(f"moveL({pose[0]:.1f})")

    def control_step(self):
        pass


@pytest.fixture
def arm(monkeypatch):
    monkeypatch.setattr(_calibration.time, "sleep", lambda _s: None)
    return _StubArm()


def _answer_prompts(monkeypatch, arm):
    """Log each prompt in the same stream as the hardware calls."""

    def fake_input(prompt=""):
        arm.log.append(f"prompt:{prompt.split('.')[0][:20]}")
        return ""

    monkeypatch.setattr("builtins.input", fake_input)


def test_torque_is_released_only_after_the_user_confirms_a_grip(monkeypatch, arm):
    """The original sequence cut torque and *then* printed the warning, so the
    arm had already dropped by the time it could be read."""
    _answer_prompts(monkeypatch, arm)

    _calibration.free_drive_pose(arm, "place it somewhere safe")

    assert arm.log.index("torque_off") > 0
    assert arm.log[0].startswith("prompt:")


def test_free_drive_latches_the_pose_before_energising(monkeypatch, arm):
    _answer_prompts(monkeypatch, arm)

    _calibration.free_drive_pose(arm, "place it somewhere safe")

    hardware = [step for step in arm.log if not step.startswith("prompt:")]
    assert hardware == ["torque_off", "end_calibration_mode", "latch", "torque_on", "read_pose"]


def test_free_drive_reads_the_pose_after_the_user_lets_go(monkeypatch, arm):
    """Held, the pose includes whatever the user was supporting; released, it is
    the pose the servos hold against gravity."""
    _answer_prompts(monkeypatch, arm)

    _calibration.free_drive_pose(arm, "place it somewhere safe")

    assert arm.log.index("read_pose") > len([s for s in arm.log if s.startswith("prompt:")])


def test_returns_to_start_drives_back_to_the_recorded_pose(arm):
    with _calibration.returns_to_start(arm, speed=0.2) as start:
        assert start.tolist() == arm.pose.tolist()
        arm.pose = np.array([9.9, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert arm.log == ["read_pose", "moveL(0.1)"]


def test_returns_to_start_also_unwinds_after_a_failure(arm):
    with pytest.raises(RuntimeError):
        with _calibration.returns_to_start(arm, speed=0.2):
            raise RuntimeError("calibration aborted")

    assert "moveL(0.1)" in arm.log


def test_returns_to_start_does_not_mask_the_original_error(arm):
    """A bus that already dropped will fail the drive home; the reason the
    calibration was unwinding matters more than that failure."""

    def broken(pose, speed):
        raise OSError("port closed")

    arm.moveL = broken
    with pytest.raises(RuntimeError, match="calibration aborted"):
        with _calibration.returns_to_start(arm, speed=0.2):
            raise RuntimeError("calibration aborted")
