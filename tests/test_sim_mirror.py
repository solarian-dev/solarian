"""Live-mirror helpers used by ``solarian calibrate-ik``."""

from solarian.sim.mirror import joints_present


class _FakeModule:
    def __init__(self, calibration, readable):
        self.calibration = calibration
        self._readable = readable
        self.last_names = None

    def read_sensors(self, data_type, names, normalize=True):
        del data_type, normalize
        self.last_names = list(names)
        return {n: {"value": float(self._readable[n])} for n in names if n in self._readable}


def test_joints_present_skips_uncalibrated_continuous_joint():
    """wrist_roll is skipped by calibrate-module; reading it must not abort the mirror."""
    module = _FakeModule(
        calibration={"shoulder_pan": object(), "elbow_flex": object()},
        readable={"shoulder_pan": -10.0, "elbow_flex": 20.0, "wrist_roll": 0.0},
    )
    out = joints_present(module, ["shoulder_pan", "elbow_flex", "wrist_roll"])
    assert out == {"shoulder_pan": -10.0, "elbow_flex": 20.0}
    assert "wrist_roll" not in module.last_names


def test_joints_present_empty_when_nothing_calibrated():
    module = _FakeModule(calibration={}, readable={"wrist_roll": 1.0})
    assert joints_present(module, ["wrist_roll"]) == {}
    assert module.last_names is None
