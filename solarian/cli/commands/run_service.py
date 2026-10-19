"""
``solarian run-service <name>`` — run a service: a model, tracker or policy server.

``<name>`` is a project conf with a ``server:`` block (``conf/services/<name>.yaml``
or ``conf/skills/<name>.yaml``) or an installed ``solarian.services`` entry
point; see :mod:`solarian.services.launch`. The server module runs as
``python -m <server_module> --host … --port … [<checkpoint_arg> <checkpoint>]``,
with arguments after the known flags passed through unchanged.
"""

import argparse
import subprocess
import sys

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry


@CommandRegistry.register('run-service')
class RunServiceCommand(BaseCommand):
    category = "Controller"
    uses_config = False
    help = 'Run a service: a project conf\'s server (tracker, policy) or an installed service'
    description = """
    A service computes something on a machine with a GPU: a tracked bounding
    box an observation is enriched with, or a policy's next action. This runs
    its server; whatever calls it (a calculated field, a skill) reaches it over
    HTTP at the URL its conf names, usually a ${deployment.*} key.

    Name a project conf to run the server its server: block describes, with
    its checkpoint, port and arguments, or name an installed service to run it
    with your own. A policy is a service that takes a checkpoint.
    """
    usage = "<conf_or_service> [--checkpoint PATH] [--host HOST] [--port PORT] [server args...]"
    options = {
        "<conf_or_service>": "A conf in conf/services/ or conf/skills/ with a server: block "
                             "(name, filename or path), or an installed service; unknown "
                             "names print the installed services.",
        "--checkpoint PATH": "Checkpoint to serve, overriding the conf's server.checkpoint.",
        "--host HOST": "Address to bind (default: the conf's, else 0.0.0.0).",
        "--port PORT": "Port to bind (default: the conf's, else the service's).",
        "server args": "Anything else is passed to the server, after the conf's server.args.",
    }
    examples = (
        "solarian run-service bbox_tracker",
        "solarian run-service umi_pnp",
        "solarian run-service umi_pnp --checkpoint models/pick/checkpoints/latest.ckpt",
        "solarian run-service sam2-tracker --port 8765 --model small",
    )

    def parse_args(self, argv: list[str]) -> list[str]:
        parser = argparse.ArgumentParser(prog="solarian run-service")
        parser.add_argument("name", help="Conf with a server: block, or an installed service.")
        parser.add_argument("--checkpoint", "-c", default=None, help="Checkpoint to serve.")
        parser.add_argument("--host", default=None, help="Host to bind the server to.")
        parser.add_argument("--port", type=int, default=None, help="Port to bind the server to.")
        ns, self.server_args = parser.parse_known_args(argv)
        self.name = ns.name
        self.checkpoint = ns.checkpoint
        self.host = ns.host
        self.port = ns.port
        return []

    def handle(self, cfg: DictConfig) -> None:
        from solarian.services.launch import resolve_server

        try:
            spec = resolve_server(self.name, checkpoint=self.checkpoint, host=self.host, port=self.port)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        spec.args += self.server_args
        cmd = spec.command()
        print(f"Running service {spec.service!r} for {self.name!r}:")
        print("  " + " ".join(cmd))
        try:
            completed = subprocess.run(cmd)
        except KeyboardInterrupt:
            return
        if completed.returncode != 0:
            sys.exit(completed.returncode)
