"""Live-mirror helpers used by ``solarian calibrate``."""

from solarian.core.data import PosData
from solarian.sim.mirror import SimMirror, joints_present


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
    """wrist_roll is skipped by the hard-stop sweep; reading it must not abort the mirror."""
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


class _RecordingClient:
    def __init__(self):
        self.calls = []

    def write(self, values, normalize=True, snap=False):
        self.calls.append((values, normalize, snap))


class _PortBus:
    def __init__(self, components):
        self.components_by_port = {"usb": components}


def test_show_model_zero_writes_unnormalized_zeros():
    client = _RecordingClient()
    mirror = SimMirror(robot=None)
    mirror.available = True
    module = type("M", (), {})()
    module.module_bus = _PortBus({"shoulder_pan": object(), "wrist_roll": object()})
    mirror.robot = type("R", (), {"modules": {"right_arm": module}})()
    mirror._clients = {("right_arm", "usb"): client}

    mirror.show_model_zero("right_arm", ["shoulder_pan", "wrist_roll"])

    assert len(client.calls) == 1
    values, normalize, snap = client.calls[0]
    assert normalize is False
    assert snap is True
    assert set(values) == {"shoulder_pan", "wrist_roll"}
    assert all(isinstance(v, PosData) and v.value == 0.0 for v in values.values())


class _ReadClient:
    def __init__(self, radians):
        self.radians = radians

    def read(self, data_type, names, normalize=True):
        del data_type, normalize
        return {n: self.radians[n] for n in names if n in self.radians}


class _HudPipe:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)


def test_update_hud_sends_real_and_sim_rows():
    client = _ReadClient({"shoulder_pan": 0.12, "wrist_roll": -0.4})
    mirror = SimMirror(robot=None)
    mirror.available = True
    module = type("M", (), {})()
    module.module_bus = _PortBus({"shoulder_pan": object(), "wrist_roll": object()})
    mirror.robot = type("R", (), {"modules": {"right_arm": module}})()
    mirror._clients = {("right_arm", "usb"): client}
    mirror._prompt_parent = _HudPipe()
    mirror._active_joint = "wrist_roll"

    mirror.update_hud("right_arm", ["shoulder_pan", "wrist_roll"], {"shoulder_pan": -2.6})

    assert len(mirror._prompt_parent.sent) == 1
    kind, payload = mirror._prompt_parent.sent[0]
    assert kind == "hud"
    assert payload["joints"] == ["shoulder_pan", "wrist_roll"]
    assert payload["real"] == {"shoulder_pan": -2.6}
    assert payload["sim"] == {"shoulder_pan": 0.12, "wrist_roll": -0.4}
    assert payload["active"] == "wrist_roll"
