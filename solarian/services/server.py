"""ServiceServer — base class for hosting a service behind a small HTTP API.

Mirrors :class:`~solarian.core.ai.system1.policy_server.PolicyServer`: the
base hides FastAPI + uvicorn, and a subclass implements :meth:`load`,
:attr:`info` and :meth:`add_routes` for its own endpoints. Every server
answers::

    GET  /        → liveness banner
    GET  /health  → {"status", "loaded"}
    GET  /info    → arbitrary JSON metadata

Request and response bodies use :mod:`solarian.utils.wire`, so numpy arrays
cross as raw bytes.
"""

import abc
import argparse
import logging
from typing import Any, Dict, Optional

from solarian.utils.wire import to_serializable

logger = logging.getLogger(__name__)


class ServiceServer(abc.ABC):
    title: str = "Solarian Service"

    def __init__(self, **config: Any):
        self.config: Dict[str, Any] = dict(config)
        self.loaded = False

    @abc.abstractmethod
    def load(self) -> None:
        """Load the model; called once before the server accepts requests."""

    @property
    @abc.abstractmethod
    def info(self) -> Dict[str, Any]:
        """JSON-serialisable metadata about the loaded service."""

    @abc.abstractmethod
    def add_routes(self, app) -> None:
        """Register this service's endpoints on the FastAPI ``app``."""

    def build_app(self):
        # Lazy: a robot or controller importing a plugin's processor must not
        # need the web stack.
        from fastapi import FastAPI, HTTPException

        server = self
        app = FastAPI(title=self.title, version="1.0.0")

        @app.get("/")
        async def root():
            return {"message": f"{server.title} is running", "status": "ok"}

        @app.get("/health")
        async def health():
            if not server.loaded:
                raise HTTPException(status_code=503, detail="Service not loaded")
            return {"status": "healthy", "loaded": True}

        @app.get("/info")
        async def info():
            return to_serializable(server.info)

        self.add_routes(app)
        return app

    def serve(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        import uvicorn

        self.load()
        self.loaded = True
        app = self.build_app()
        logger.info("Serving %s on http://%s:%d", self.title, host, port)
        uvicorn.run(app, host=host, port=port)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add subclass-specific CLI arguments."""

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ServiceServer":
        return cls()

    @classmethod
    def main(cls, argv: Optional[list] = None, default_port: int = 8765) -> None:
        """Console entry point: parse args, build, load and :meth:`serve`."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        parser = argparse.ArgumentParser(description=f"{cls.__name__} service")
        parser.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0)")
        parser.add_argument("--port", "-p", type=int, default=default_port, help="Bind port")
        cls.add_arguments(parser)
        args = parser.parse_args(argv)
        cls.from_args(args).serve(host=args.host, port=args.port)
