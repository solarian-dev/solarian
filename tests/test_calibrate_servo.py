"""Brand discovery and the calibrate-servo menu flow, with a fake bus class."""

import builtins

import pytest

from solarian.cli.commands.calibrate_servo import CalibrateServoCommand, select
from solarian.core import actuator_bus_registry as registry


class _FakeEntryPoint:
    def __init__(self, name, value):
        self.name = name
        self._value = value

    def load(self):
        return self._value


class _FakeBus:
    """Bus class stand-in: records what calibrate-servo asked it to do."""

    ports = ["/dev/fake0"]
    servos = [{"id": 1, "model": "sts3215", "baudrate": 1_000_000}]
    provisioned: list = []

    @classmethod
    def available_ports(cls):
        return list(cls.ports)

    @classmethod
    def scan_port(cls, port):
        del port
        return list(cls.servos)

    @classmethod
    def provision_servo(cls, port, found):
        cls.provisioned.append((port, found))


@pytest.fixture
def fake_buses(monkeypatch):
    _FakeBus.provisioned = []
    eps = [
        _FakeEntryPoint("feetech", _FakeBus),
        _FakeEntryPoint("dynamixel", _FakeBus),
    ]
    monkeypatch.setattr(registry, "_iter_entry_points", lambda group: list(eps))
    return _FakeBus


def _answers(monkeypatch, replies):
    """Feed *replies* to input(), failing if the flow asks more than expected."""
    queue = list(replies)

    def _input(prompt=""):
        del prompt
        assert queue, "flow asked for more input than expected"
        return queue.pop(0)

    monkeypatch.setattr(builtins, "input", _input)
    return queue


class TestRegistry:
    def test_available_names_are_sorted(self, fake_buses):
        assert registry.available_bus_names() == ["dynamixel", "feetech"]

    def test_get_bus_class_resolves(self, fake_buses):
        assert registry.get_bus_class("feetech") is fake_buses

    def test_unknown_name_lists_what_is_installed(self, fake_buses):
        with pytest.raises(KeyError, match="dynamixel, feetech"):
            registry.get_bus_class("nope")

    def test_discover_skips_broken_plugin(self, monkeypatch):
        class _Broken(_FakeEntryPoint):
            def load(self):
                raise ImportError("no such module")

        monkeypatch.setattr(
            registry,
            "_iter_entry_points",
            lambda group: [_Broken("broken", None), _FakeEntryPoint("ok", _FakeBus)],
        )
        assert registry.discover_bus_classes() == {"ok": _FakeBus}


class TestSelect:
    def test_single_candidate_is_auto_selected(self, monkeypatch):
        _answers(monkeypatch, [])
        assert select("port", ["/dev/only"]) == "/dev/only"

    def test_empty_returns_none(self):
        assert select("port", []) is None

    def test_number_picks_that_option(self, monkeypatch):
        _answers(monkeypatch, ["2"])
        assert select("port", ["/dev/a", "/dev/b"]) == "/dev/b"

    def test_blank_stops(self, monkeypatch):
        _answers(monkeypatch, [""])
        assert select("port", ["/dev/a", "/dev/b"]) is None


class TestCalibrateServoCommand:
    def test_provisions_the_only_servo_found(self, fake_buses, monkeypatch):
        # brand menu (2 brands), then port and servo auto-select, then stop.
        _answers(monkeypatch, ["1", "n"])
        CalibrateServoCommand().handle(None)

        assert fake_buses.provisioned == [
            ("/dev/fake0", {"id": 1, "model": "sts3215", "baudrate": 1_000_000})
        ]

    def test_unsupported_brand_does_not_crash(self, fake_buses, monkeypatch):
        def _unsupported(cls):
            raise NotImplementedError("FakeBus cannot enumerate ports")

        monkeypatch.setattr(
            fake_buses, "available_ports", classmethod(_unsupported)
        )
        _answers(monkeypatch, ["1"])
        CalibrateServoCommand().handle(None)

        assert fake_buses.provisioned == []

    def test_no_servo_found_offers_a_rescan(self, fake_buses, monkeypatch):
        monkeypatch.setattr(fake_buses, "servos", [])
        # brand, then "scan again?" twice: yes, then no.
        _answers(monkeypatch, ["1", "y", "n"])
        CalibrateServoCommand().handle(None)

        assert fake_buses.provisioned == []

    def test_second_round_rescans_before_asking_again(self, fake_buses, monkeypatch):
        scans: list[str] = []
        real_scan = fake_buses.scan_port

        def _counting_scan(cls, port):
            scans.append(port)
            return real_scan(port)

        monkeypatch.setattr(fake_buses, "scan_port", classmethod(_counting_scan))
        _answers(monkeypatch, ["1", "y", "n"])
        CalibrateServoCommand().handle(None)

        assert scans == ["/dev/fake0", "/dev/fake0"]
        assert len(fake_buses.provisioned) == 2
