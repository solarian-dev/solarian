"""CurveZMQ for the relay link: authentication and encryption, per socket.

Nothing on the ZMQ path was authenticated or encrypted, which is tolerable on a
LAN and is not tolerable for a relay with a public endpoint — anyone who can
reach the port can command the robot. CURVE closes that, and pyzmq supports it
natively, so it is socket options rather than new machinery.

Three parties, two roles:

* the relay is the CURVE **server**: it holds a secret key and publishes the
  matching public key;
* the robot and the controller are CURVE **clients**: each holds its own keypair
  and the relay's public key.

A public key is not a secret — it ships inside the connection yaml the robot
downloads — so encryption alone still lets anyone holding that file connect. The
relay closes the loop with ``allowed_client_keys``: a ZAP handler that admits only
the robot and controller keys registered for that deployment. Leave every field
unset and the link stays plaintext, exactly as before.
"""

from __future__ import annotations

import logging

import zmq
import zmq.auth
from zmq.auth.thread import ThreadAuthenticator

logger = logging.getLogger(__name__)


def generate_keypair() -> tuple[str, str]:
    """A fresh (public, secret) Z85 keypair, as ``str``.

    Z85 text rather than bytes because these live in yaml files and database
    columns far more often than in memory.

    Z85 contains both ``$`` and ``{``, and a key that happens to spell ``${``
    is an interpolation to OmegaConf: it fails to resolve the moment the config
    is read, wherever that key was written. About one key in two hundred, which
    is rare enough to reach a user as a mystery. Cheaper to not issue those.
    """
    while True:
        public, secret = (key.decode() for key in zmq.curve_keypair())
        if "${" not in public and "${" not in secret:
            return public, secret


def is_enabled(connection) -> bool:
    """Whether this connection config asks for CURVE at all."""
    return bool(connection.curve_secret_key and connection.curve_public_key)


def apply_client(socket: zmq.Socket, connection, role: str) -> bool:
    """Give *socket* the keys to reach a CURVE relay. Returns whether it did.

    Both halves are required: the peer's own keypair to prove who it is, and the
    relay's public key to know who it is talking to. A half-configured link fails
    here, loudly, rather than silently falling back to plaintext against a relay
    that will refuse it anyway.
    """
    if not is_enabled(connection):
        if connection.curve_server_key:
            raise ValueError(
                f"{role}: connection.curve_server_key is set without this peer's "
                f"own curve_public_key / curve_secret_key. Run `solarian keygen` "
                f"and add both."
            )
        return False
    if not connection.curve_server_key:
        raise ValueError(
            f"{role}: CURVE keys are set but connection.curve_server_key (the "
            f"relay's public key) is missing, so there is no server to trust."
        )

    socket.curve_publickey = connection.curve_public_key.encode()
    socket.curve_secretkey = connection.curve_secret_key.encode()
    socket.curve_serverkey = connection.curve_server_key.encode()
    return True


def apply_server(socket: zmq.Socket, connection, role: str) -> bool:
    """Make *socket* a CURVE server. Returns whether it did.

    The relay's own keypair, so ``curve_server_key`` is not read here — that is
    the copy handed to clients.
    """
    if not is_enabled(connection):
        return False
    socket.curve_publickey = connection.curve_public_key.encode()
    socket.curve_secretkey = connection.curve_secret_key.encode()
    socket.curve_server = True
    logger.debug("CURVE enabled on %s", role)
    return True


class _KeyAllowlist:
    """ZAP credentials provider admitting exactly the keys a relay was given.

    Keys are compared as Z85 bytes because that is what the ZAP handler passes
    in, while configuration carries them as text.
    """

    def __init__(self, allowed: list[str]):
        self._allowed = frozenset(
            key.encode() if isinstance(key, str) else key for key in allowed
        )

    def callback(self, domain: str, key) -> bool:
        if (key.encode() if isinstance(key, str) else key) in self._allowed:
            return True
        logger.warning("CURVE: refused a client presenting key %s", key)
        return False


def start_authenticator(
    context: zmq.Context, allowed_client_keys: list[str]
) -> ThreadAuthenticator | None:
    """Admit only *allowed_client_keys* on this context's CURVE servers.

    Without a ZAP handler libzmq skips authorization entirely, so the relay would
    happily encrypt traffic for anyone who knows its public key — and every copy
    of the connection yaml contains it. With one, a leaked yaml is not a way in.

    An empty list means no allowlist was configured: encryption without
    authorization, which is a reasonable default on a LAN and the wrong one on
    the internet.
    """
    if not allowed_client_keys:
        return None
    authenticator = ThreadAuthenticator(context)
    authenticator.start()
    authenticator.configure_curve_callback(
        domain="*", credentials_provider=_KeyAllowlist(allowed_client_keys)
    )
    logger.info(
        "CURVE authenticator admitting %d client key(s)", len(allowed_client_keys)
    )
    return authenticator
