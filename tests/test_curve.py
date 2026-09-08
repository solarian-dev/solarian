"""CurveZMQ on the relay link: encryption, and an allowlist that actually bites."""

from __future__ import annotations

import io
import re
from unittest import mock

import pytest
import zmq

from solarian.core import curve
from solarian.core.configs import ConnectionConfig


def _connection(**keys) -> ConnectionConfig:
    return ConnectionConfig(
        ip="127.0.0.1",
        action_sender_port=1,
        action_receiver_port=2,
        state_sender_port=3,
        state_receiver_port=4,
        **keys,
    )


@pytest.fixture
def context():
    ctx = zmq.Context()
    yield ctx
    ctx.term()


def test_a_plaintext_link_is_left_alone(context):
    connection = _connection()
    socket = context.socket(zmq.PUSH)
    try:
        assert curve.apply_client(socket, connection, "test") is False
        assert curve.apply_server(socket, connection, "test") is False
        assert curve.is_enabled(connection) is False
    finally:
        socket.close(linger=0)


def test_half_configured_keys_fail_loudly(context):
    relay_public, _ = curve.generate_keypair()
    peer_public, peer_secret = curve.generate_keypair()

    socket = context.socket(zmq.PUSH)
    try:
        with pytest.raises(ValueError, match="own curve_public_key"):
            curve.apply_client(
                socket, _connection(curve_server_key=relay_public), "test"
            )
        with pytest.raises(ValueError, match="curve_server_key"):
            curve.apply_client(
                socket,
                _connection(
                    curve_public_key=peer_public, curve_secret_key=peer_secret
                ),
                "test",
            )
    finally:
        socket.close(linger=0)


def _relay(context, allowed_client_keys):
    """A CURVE REP socket standing in for one of the relay's five."""
    relay_public, relay_secret = curve.generate_keypair()
    authenticator = curve.start_authenticator(context, allowed_client_keys)
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    # Bounded, so a failed handshake fails the test instead of hanging it.
    socket.setsockopt(zmq.RCVTIMEO, 2000)
    curve.apply_server(
        socket,
        _connection(curve_public_key=relay_public, curve_secret_key=relay_secret),
        "test relay",
    )
    socket.bind("tcp://127.0.0.1:*")
    port = int(socket.getsockopt_string(zmq.LAST_ENDPOINT).rsplit(":", 1)[-1])
    return socket, port, relay_public, authenticator


def _client(context, port, relay_public, keypair):
    public, secret = keypair
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 1500)
    socket.setsockopt(zmq.SNDTIMEO, 1500)
    curve.apply_client(
        socket,
        _connection(
            curve_server_key=relay_public,
            curve_public_key=public,
            curve_secret_key=secret,
        ),
        "test client",
    )
    socket.connect(f"tcp://127.0.0.1:{port}")
    return socket


def test_an_allowlisted_client_gets_through(context):
    keypair = curve.generate_keypair()
    relay, port, relay_public, authenticator = _relay(context, [keypair[0]])
    client = _client(context, port, relay_public, keypair)
    try:
        client.send(b"ping")
        assert relay.recv() == b"ping"
        relay.send(b"pong")
        assert client.recv() == b"pong"
    finally:
        client.close(linger=0)
        relay.close(linger=0)
        authenticator.stop()


def test_generated_keys_never_spell_an_interpolation():
    """``${`` in a key surfaces at the far end, when the config is read."""
    for _ in range(2000):
        public, secret = curve.generate_keypair()
        assert "${" not in public
        assert "${" not in secret


def test_keygen_prints_yaml_a_parser_can_read_back():
    """The command's output is meant to be pasted into conf/, so it has to load.

    A Z85 key can open with []{}!&*>%@#, which YAML reads as syntax, so the
    unquoted form is a parse error for roughly one key in eight — often enough
    to look like bad luck rather than a bug.
    """
    from omegaconf import OmegaConf

    from solarian.cli.commands.keygen import KeygenCommand

    printed: list[str] = []
    with mock.patch("builtins.print", lambda *args: printed.append(" ".join(map(str, args)))):
        # Every key the command can emit, over enough runs that an indicator
        # character has to turn up in a leading position.
        for _ in range(60):
            KeygenCommand().handle(OmegaConf.create({}))

    # Each "# ..." heading starts a snippet meant to be pasted on its own, so
    # they are parsed separately: pasted together they would collide on
    # `connection:` and only the last would be checked.
    yaml_line = re.compile(r"^\s*[a-z_]+:( |$)")
    snippets: list[list[str]] = []
    for line in "\n".join(printed).splitlines():
        if line.startswith("# "):
            snippets.append([])
        elif snippets and yaml_line.match(line):
            snippets[-1].append(line)

    keys_seen = 0
    for snippet in snippets:
        loaded = OmegaConf.load(io.StringIO("\n".join(snippet)))
        for name, value in loaded.items():
            keys = value if name == "curve_allowed_client_keys" else value.values()
            for key in keys:
                assert isinstance(key, str), f"{name}: {key!r}"
                assert len(key) == 40, f"{name}: {key!r}"
                keys_seen += 1
    # Ten a run: the allowlist's two, the relay's own pair, and three each for
    # the robot and the controller.
    assert keys_seen == 60 * 10


def test_a_key_that_is_not_allowlisted_is_refused(context):
    allowed = curve.generate_keypair()
    stranger = curve.generate_keypair()
    relay, port, relay_public, authenticator = _relay(context, [allowed[0]])
    client = _client(context, port, relay_public, stranger)
    try:
        # Knowing the relay's public key is not enough: it ships in the
        # connection yaml, so a leaked file must not be a way in.
        client.send(b"ping")
        with pytest.raises(zmq.Again):
            relay.recv()
    finally:
        client.close(linger=0)
        relay.close(linger=0)
        authenticator.stop()
