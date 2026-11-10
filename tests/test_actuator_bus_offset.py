"""Software center_offset is applied in normalize; hardware buses skip it."""

import pytest

from solarian.core.actuator_bus import ActuatorBus
from solarian.core.calibration import MotorCalibration
from solarian.core.configs import ActuatorConfig

CPR = 4096


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

    def drive_to_encoder_center(self, component_names):
        return {n: (CPR - 1) // 2 for n in component_names}

    def counts_per_revolution(self, name):
        return CPR


class _HwBus(_SoftBus):
    def center_offset_in_hardware(self, name):
        return True

    def max_homing_offset_abs(self, name):
        del name
        return 2047


class _SoftBusWithHwCap(_SoftBus):
    """Software offset, but still advertises a leftover Homing_Offset register cap."""

    def max_homing_offset_abs(self, name):
        del name
        return 2047


def _bus(cls, offset: int, *, range_min: int = 0, range_max: int = 4095) -> ActuatorBus:
    cal = {
        "j": MotorCalibration(
            id=1,
            drive_mode=0,
            homing_offset=offset,
            range_min=range_min,
            range_max=range_max,
        )
    }
    return cls(port="x", components={"j": _FakeAct()}, calibration=cal)


def test_software_offset_maps_home_to_zero_norm():
    bus = _bus(_SoftBus, offset=1000)
    # raw 1000 → present 0 → norm -100; mid present 2047.5 ≈ raw 3047.5
    assert bus._normalize({"j": 1000})["j"] == -100.0
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


def test_gripper_wrap_ends_stay_in_present_window():
    """arm1 right_gripper: seam in the middle of jaw travel."""
    bus = _bus(_SoftBus, offset=-1967, range_min=1530, range_max=2565)
    for raw in (3659, 598):
        present = bus._present_ticks("j", raw)
        assert 1530 <= present <= 2565, f"raw {raw} → present {present}"
    closed = bus._normalize({"j": 3659})["j"]
    opened = bus._normalize({"j": 598})["j"]
    assert closed == -100.0
    assert opened == 100.0
    assert abs(bus._unnormalize({"j": closed})["j"] - 3659) < 1.0
    assert abs(bus._unnormalize({"j": opened})["j"] - 598) < 1.0


def test_software_cap_is_cpr_minus_one_even_with_hw_register_cap():
    bus = _bus(_SoftBusWithHwCap, offset=0)
    assert bus.max_homing_offset_abs("j") == 2047
    assert bus.max_center_offset_abs("j") == CPR - 1
    hw = _bus(_HwBus, offset=0)
    assert hw.max_center_offset_abs("j") == 2047


class _Gripper:
    def __init__(self):
        self.cfg = ActuatorConfig(id=7, model="sts3250", mode="range_m100_100")


def _feetech_gripper_bus(homing_offset: int, reads: dict[str, int] | None = None):
    """A FeetechBus whose inner bus records register access instead of doing I/O.

    Provisioning has to work on servos that are not declared components, so the
    register writes go through addresses rather than names; the capture maps them
    back via the STS/SMS control table.
    """
    from solarian_feetech.bus import FeetechBus
    from solarian_feetech.driver.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE

    cal = {
        "gripper": MotorCalibration(
            id=7, drive_mode=1, homing_offset=homing_offset, range_min=1530, range_max=2565
        )
    }
    bus = FeetechBus(port="x", components={"gripper": _Gripper()}, calibration=dict(cal))

    name_of = {addr: name for name, (addr, _len) in STS_SMS_SERIES_CONTROL_TABLE.items()}
    reads = reads or {}
    writes: list[tuple[str, int]] = []

    def _write(addr, length, motor_id, value, **_kwargs):
        del length, motor_id
        writes.append((name_of[addr], int(value)))

    def _read(addr, length, motor_id, **_kwargs):
        del length, motor_id
        return int(reads.get(name_of[addr], 0)), 0, 0

    bus.bus._write = _write
    bus.bus._read = _read
    return bus, cal, writes


def test_feetech_write_calibration_keeps_homing_offset_at_zero():
    pytest.importorskip("solarian_feetech")
    bus, cal, writes = _feetech_gripper_bus(-1967)
    bus.write_calibration(cal)

    assert bus.center_offset_in_hardware("gripper") is False
    assert bus.max_center_offset_abs("gripper") == CPR - 1
    assert bus.calibration["gripper"].homing_offset == -1967
    assert bus.read_calibration()["gripper"].homing_offset == -1967
    homing_writes = [value for name, value in writes if name == "Homing_Offset"]
    assert homing_writes
    assert all(value == 0 for value in homing_writes)
    assert ("Lock", 0) in writes
    assert ("Min_Position_Limit", 0) in writes
    assert ("Max_Position_Limit", CPR - 1) in writes


def test_feetech_write_calibration_raises_if_homing_offset_not_cleared():
    pytest.importorskip("solarian_feetech")
    bus, cal, _ = _feetech_gripper_bus(-1967, reads={"Homing_Offset": 1967})
    with pytest.raises(RuntimeError, match="Homing_Offset"):
        bus.write_calibration(cal)


def test_feetech_drive_to_encoder_center_commands_mid_and_leaves_torque_off():
    pytest.importorskip("solarian_feetech")
    bus, _, writes = _feetech_gripper_bus(0, reads={"Present_Position": (CPR - 1) // 2})

    assert bus.drive_to_encoder_center(["gripper"]) == {"gripper": (CPR - 1) // 2}
    assert ("Goal_Position", (CPR - 1) // 2) in writes
    torque_writes = [value for name, value in writes if name == "Torque_Enable"]
    assert torque_writes[-1] == 0
