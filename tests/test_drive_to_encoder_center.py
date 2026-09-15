"""Centering leaves torque off and commands absolute mid-encoder.

Only Feetech hardware was on hand (covered in ``test_actuator_bus_offset.py``),
so the Dynamixel and Robstride paths are pinned here against fakes.
"""

import pytest

CPR = 4096
MID = (CPR - 1) // 2


class _NamedBus:
    """Records the named-register API a Dynamixel bus exposes."""

    def __init__(self, present=MID):
        self.writes: list[tuple[str, int]] = []
        self.torque: list[str] = []
        self.present = present

    def disable_torque(self, motors):
        del motors
        self.torque.append("off")

    def enable_torque(self, motors):
        del motors
        self.torque.append("on")

    def write(self, data_name, motor, value, normalize=True, **kwargs):
        del motor, normalize, kwargs
        self.writes.append((data_name, int(value)))

    def read(self, data_name, motor, normalize=True, **kwargs):
        del data_name, motor, normalize, kwargs
        return self.present


def test_dynamixel_opens_registers_then_commands_mid():
    pytest.importorskip("solarian_dynamixel")
    from solarian_dynamixel.bus import move_to_mid_encoder

    bus = _NamedBus()
    assert move_to_mid_encoder(bus, "servo", CPR) == MID

    assert ("Homing_Offset", 0) in bus.writes
    assert ("Min_Position_Limit", 0) in bus.writes
    assert ("Max_Position_Limit", CPR - 1) in bus.writes
    assert ("Goal_Position", MID) in bus.writes
    # Offsets and limits are EEPROM: they must be written before torque goes on.
    assert bus.writes.index(("Homing_Offset", 0)) < bus.writes.index(("Goal_Position", MID))
    assert bus.torque[-1] == "off"


def test_dynamixel_blocked_shaft_still_releases_torque():
    pytest.importorskip("solarian_dynamixel")
    from solarian_dynamixel.bus import move_to_mid_encoder

    bus = _NamedBus(present=200)
    assert move_to_mid_encoder(bus, "servo", CPR, settle_s=0.0) == 200
    assert bus.torque[-1] == "off"


def test_robstride_commands_zero_radians_and_disables():
    pytest.importorskip("solarian_robstride")
    from solarian_robstride.bus import RobstrideBus
    from solarian_robstride.can_codec import MotorFeedback
    from solarian_robstride.protocol import PROTOCOL_CPR

    sent: list[int] = []
    torque: list[str] = []

    bus = RobstrideBus.__new__(RobstrideBus)
    bus.host_can_id = 0xFD
    bus.components = {}
    bus._send = lambda can_id, data: sent.append(can_id)
    bus._motor_id = lambda name: 1
    bus._set_run_mode = lambda name, mode: None
    bus.enable_torque = lambda names: torque.append("on")
    bus.disable_torque = lambda names: torque.append("off")
    bus._get_feedback = lambda name, timeout=0.2: MotorFeedback(
        motor_id=1,
        position=0.0,
        velocity=0.0,
        torque=0.0,
        temperature=25.0,
        position_raw=PROTOCOL_CPR // 2,
        mode_state=2,
        fault_code=0,
    )

    assert bus.drive_to_encoder_center(["j"]) == {"j": PROTOCOL_CPR // 2}
    assert sent, "no LOC_REF frame was sent"
    assert torque == ["on", "off"]
