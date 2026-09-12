"""BusClient RPC kwargs: snap is sim-only and must not reach hardware buses."""

from solarian.core.bus_server import BusClient
from solarian.core.data import PosData


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
