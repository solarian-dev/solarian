"""Robot shutdown must not leave a module process holding the camera."""

from types import SimpleNamespace

from solarian.core.robot import (
    _MODULE_KILL_JOIN_S,
    _MODULE_STOP_JOIN_S,
    _MODULE_TERM_JOIN_S,
    Robot,
)


class StubbornModule:
    """Still alive after stop and SIGTERM, the way a stuck camera child is."""

    def __init__(self):
        self.calls = []
        self.alive = True

    def is_alive(self):
        return self.alive

    def stop(self, wait=True):
        self.calls.append("stop")

    def join(self, timeout=None):
        self.calls.append(("join", timeout))

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")
        self.alive = False

    def get_profiling_stats(self):
        return None

    def disconnect(self):
        self.calls.append("disconnect")


def test_disconnect_kills_a_module_that_survives_terminate():
    module = StubbornModule()
    host = SimpleNamespace(
        is_sim=False,
        recorder=None,
        action_socket=None,
        state_socket=None,
        modules={"cam": module},
        _bus_servers=[],
        context=None,
        connected=True,
        running=True,
    )
    host._await_seal = lambda: None

    Robot.disconnect(host)

    assert module.calls == [
        "stop",
        ("join", _MODULE_STOP_JOIN_S),
        "terminate",
        ("join", _MODULE_TERM_JOIN_S),
        "kill",
        ("join", _MODULE_KILL_JOIN_S),
        "disconnect",
    ]
    assert host.connected is False
