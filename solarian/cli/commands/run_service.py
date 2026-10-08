"""
``solarian run-service <name>`` — run the server of an installed service plugin.

Looks the name up under the ``solarian.services`` entry points and runs the
plugin's server module (``python -m <server_module> --host … --port …``).
Arguments after the known flags are passed to the server unchanged, so each
service keeps its own options (model size, checkpoint directory, …).
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
    help = 'Run the server of an installed service plugin (e.g. a bbox tracker)'
    description = """
    A service computes something an observation is enriched with, such as a
    tracked bounding box, on a machine with a GPU. This runs that service's
    server; the calculated field that uses it reaches it over HTTP at the URL
    its conf names (usually a ${deployment.*} key).
    """
    usage = "<service_name> [--host HOST] [--port PORT] [server args...]"
    options = {
        "<service_name>": "Service to run; unknown names print the installed services.",
        "--host HOST": "Address to bind (default 0.0.0.0).",
        "--port PORT": "Port to bind (default: the service's default port).",
        "server args": "Anything else is passed to the service's server.",
    }
    examples = (
        "solarian run-service sam2-tracker",
        "solarian run-service sam2-tracker --port 8765 --model small",
    )

    def parse_args(self, argv: list[str]) -> list[str]:
        parser = argparse.ArgumentParser(prog="solarian run-service")
        parser.add_argument("service", help="Name of the service to run.")
        parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to.")
        parser.add_argument("--port", type=int, default=None, help="Port to bind the server to.")
        ns, self.server_args = parser.parse_known_args(argv)
        self.service = ns.service
        self.host = ns.host
        self.port = ns.port
        return []

    def handle(self, cfg: DictConfig) -> None:
        from solarian.services import discover_services

        services = discover_services()
        service_cls = services.get(self.service)
        if service_cls is None:
            available = ", ".join(sorted(services)) or "(none installed)"
            print(f"Error: unknown service {self.service!r}. Available services: {available}.")
            sys.exit(1)

        port = self.port if self.port is not None else service_cls.default_port
        cmd = [
            sys.executable,
            "-m",
            service_cls.server_module,
            "--host",
            self.host,
            "--port",
            str(port),
            *self.server_args,
        ]
        print(f"Running service {self.service!r}:")
        print("  " + " ".join(cmd))
        try:
            completed = subprocess.run(cmd)
        except KeyboardInterrupt:
            return
        if completed.returncode != 0:
            sys.exit(completed.returncode)
