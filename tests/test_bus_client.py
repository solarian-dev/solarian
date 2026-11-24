"""BusClient RPC: what reaches the server's bus, and what does not."""

import pytest

from solarian.core.bus_server import BusClient
from solarian.core.data import PosData


class _PluginBus:
    speed_cap_unit = "rad/s"

    def set_speed_cap(self, names, rad_per_s):
        pass

    def _private(self):
        pass


class _FakeConn:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)

    def recv(self):
        return None, None


def test_write_omits_snap_by_default():
    conn = _FakeConn()
    bus = BusClient(client_conn=conn, port="/dev/null", components={})
    bus.write({"j": PosData(value=1.0)})
    method, _args, kwargs = conn.sent[0]
    assert method == "write"
    assert "snap" not in kwargs


def test_write_forwards_snap_when_set():
    conn = _FakeConn()
    bus = BusClient(client_conn=conn, port="/dev/null", components={})
    bus.write({"j": PosData(value=1.0)}, snap=True)
    _method, _args, kwargs = conn.sent[0]
    assert kwargs["snap"] is True


def test_forwards_a_plugin_bus_method():
    conn = _FakeConn()
    bus = BusClient(client_conn=conn, port="/dev/null", components={}, bus_class=_PluginBus)
    bus.set_speed_cap(["j"], 1.5)
    assert conn.sent == [("set_speed_cap", (["j"], 1.5), {})]


@pytest.mark.parametrize("name", ["drive_to_nowhere", "speed_cap_unit", "_private"])
def test_does_not_claim_what_the_bus_lacks(name):
    bus = BusClient(client_conn=_FakeConn(), port="/dev/null", components={}, bus_class=_PluginBus)
    assert not hasattr(bus, name)


def test_forwards_nothing_without_a_bus_class():
    bus = BusClient(client_conn=_FakeConn(), port="/dev/null", components={})
    assert not hasattr(bus, "set_speed_cap")
