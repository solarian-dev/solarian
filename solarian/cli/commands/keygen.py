"""Generate the CurveZMQ keypairs a relay link needs."""

from __future__ import annotations

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.curve import generate_keypair


@CommandRegistry.register("keygen", None)
class KeygenCommand(BaseCommand):
    help = "Generate CurveZMQ keypairs for the relay, the robot and the controller."
    description = """
    Prints three fresh keypairs and the yaml blocks that use them: the relay's
    allowlist for conf/server.yaml, and one connection block per peer for that
    peer's conf/deployment/<name>.yaml.

    Generates only; nothing is written. Each peer keeps its own secret key, so the
    three blocks usually belong in three different deployment profiles.
    """
    usage = ""
    examples = (
        "solarian keygen",
        "solarian keygen > keys.txt",
    )

    def handle(self, cfg: DictConfig) -> None:
        relay_public, relay_secret = generate_keypair()
        robot_public, robot_secret = generate_keypair()
        controller_public, controller_secret = generate_keypair()

        print(
            "CurveZMQ keys generated. Each peer keeps its own secret key and\n"
            "carries the relay's public key; the relay carries the peers'\n"
            "public keys as its allowlist. A public key is not a secret, a\n"
            "secret key is.\n"
            "\n"
            "The three peers usually read different deployment profiles, since\n"
            "each holds a secret the other two must not. Where they share one,\n"
            "keep only that peer's keys in it.\n"
        )
        # Quoted: Z85 draws on []{}!&*>%@#, and a scalar opening with one of
        # those is YAML syntax rather than a character, for about one key in
        # eight. Z85 has no quote of its own to collide with.
        print("# relay — conf/server.yaml")
        print(
            f"curve_allowed_client_keys: ['{robot_public}', '{controller_public}']"
        )
        print()
        print("# relay — conf/deployment/<name>.yaml")
        print("connection:")
        print(f"    curve_public_key: '{relay_public}'")
        print(f"    curve_secret_key: '{relay_secret}'")
        print()
        print("# robot — conf/deployment/<name>.yaml")
        print("connection:")
        print(f"    curve_server_key: '{relay_public}'")
        print(f"    curve_public_key: '{robot_public}'")
        print(f"    curve_secret_key: '{robot_secret}'")
        print()
        print("# controller — conf/deployment/<name>.yaml")
        print("connection:")
        print(f"    curve_server_key: '{relay_public}'")
        print(f"    curve_public_key: '{controller_public}'")
        print(f"    curve_secret_key: '{controller_secret}'")
