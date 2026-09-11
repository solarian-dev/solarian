"""Software center_offset is applied in normalize; hardware buses skip it."""

from solarian.core.actuator_bus import ActuatorBus
from solarian.core.calibration import MotorCalibration
from solarian.core.configs import ActuatorConfig
class _FakeAct:
    def __init__(self, mode="range_m100_100"):
        self.cfg = ActuatorConfig(id=1, mode=mode)


class _SoftBus(ActuatorBus):
    @property
    def is_connected(self) -> bool:
        return True

    def connect(self):
        pass

    def disconnect(self):
        pass

    def read(self, data_type, component_names, normalize=True):
        del data_type, component_names, normalize
        return {}

    def write(self, values, normalize=True):
        pass

    def is_calibrated(self) -> bool:
        return True

    def reset_calibration(self, component_names):
        pass

    def read_calibration(self):
        return self.calibration or {}

    def set_half_turn_homings(self, actuator_names):
        return {n: 0 for n in actuator_names}

    def counts_per_revolution(self, name):
        return 4096


class _HwBus(_SoftBus):
    def center_offset_in_hardware(self, name):
        return True


def _bus(cls, offset: int) -> ActuatorBus:
    cal = {
        "j": MotorCalibration(
            id=1, drive_mode=0, homing_offset=offset, range_min=0, range_max=4095
        )
    }
    return cls(port="x", components={"j": _FakeAct()}, calibration=cal)


def test_software_offset_maps_home_to_zero_norm():
    bus = _bus(_SoftBus, offset=1000)
    # raw 1000 → present 0 → norm -100; mid present 2047.5 ≈ raw 3047.5
    assert bus._normalize({"j": 1000})["j"] == -100.0
    mid_raw = (0 + 1000) % 4096  # present 0 is raw 1000
    assert bus._present_ticks("j", 1000) == 0


def test_software_unnormalize_roundtrip():
    bus = _bus(_SoftBus, offset=800)
    raw = 1200.0
    norm = bus._normalize({"j": raw})
    back = bus._unnormalize(norm)
    assert abs(back["j"] - raw) < 1.0 or abs(back["j"] - raw + 4096) < 1.0


def test_hardware_flag_does_not_subtract_again():
    bus = _bus(_HwBus, offset=1000)
    # Present is already shifted; raw 0 is range_min → -100
    assert bus._normalize({"j": 0})["j"] == -100.0


def test_software_offset_wraps_raw_below_offset():
    bus = _bus(_SoftBus, offset=100)
    assert bus._present_ticks("j", 50) == 4046
    # present 4046 of 0..4095 is near +100
    assert bus._normalize({"j": 50})["j"] > 97.0
