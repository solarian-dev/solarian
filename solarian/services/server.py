"""ServiceServer — base class for hosting a model behind a small HTTP API.

The base hides FastAPI + uvicorn, and a subclass implements :meth:`load`,
:attr:`info` and :meth:`add_routes` for its own endpoints. A policy is a
service whose routes are fixed (:class:`~solarian.services.policy.PolicyServer`).
Every server answers::

    GET  /        → liveness banner
    GET  /health  → {"status", "loaded"}
    GET  /info    → arbitrary JSON metadata

Request and response bodies use :mod:`solarian.utils.wire`, so numpy arrays
cross as raw bytes. Every response carries ``X-Server-Ms``, the time the
server spent on the request, so a client can tell a slow model from a slow
link.

When ``SOLARIAN_ENDPOINT_TOKEN`` is set, every request must carry
``Authorization: Bearer <token>``: a server reachable from the internet (a
hosted endpoint on Solarian Base) is otherwise anyone's GPU.
"""

import abc
import argparse
import hmac
import logging
import os
import time
from typing import Any, Dict, Optional

from solarian.utils.wire import to_serializable

logger = logging.getLogger(__name__)

#: Environment variable holding the bearer token a server requires.
TOKEN_ENV = "SOLARIAN_ENDPOINT_TOKEN"
#: Response header with the server-side time of a request, in ms.
SERVER_MS_HEADER = "X-Server-Ms"


class ServiceServer(abc.ABC):
    title: str = "Solarian Service"
    #: Port :meth:`main` binds when not given one.
    default_port: int = 8765

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

    def build_app(self, token: Optional[str] = None):
        """The FastAPI app; ``token`` defaults to ``$SOLARIAN_ENDPOINT_TOKEN``, and
        an empty one leaves the server open."""
        # Lazy: a robot or controller importing a plugin's processor must not
        # need the web stack.
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse

        server = self
        token = (os.environ.get(TOKEN_ENV) if token is None else token) or None
        app = FastAPI(title=self.title, version="1.0.0")

        @app.middleware("http")
        async def authenticate_and_time(request: Request, call_next):
            if token is not None:
                sent = request.headers.get("authorization", "")
                if not hmac.compare_digest(sent.encode(), f"Bearer {token}".encode()):
                    return JSONResponse({"detail": "Missing or wrong bearer token"}, status_code=401)
            t0 = time.perf_counter()
            response = await call_next(request)
            response.headers[SERVER_MS_HEADER] = f"{(time.perf_counter() - t0) * 1000.0:.3f}"
            return response

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

    def serve(self, host: str = "0.0.0.0", port: Optional[int] = None) -> None:
        import uvicorn

        port = self.default_port if port is None else port
        self.load()
        self.loaded = True
        app = self.build_app()
        logger.info(
            "Serving %s on http://%s:%d%s", self.title, host, port,
            " (bearer token required)" if os.environ.get(TOKEN_ENV) else "",
        )
        uvicorn.run(app, host=host, port=port)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add subclass-specific CLI arguments."""

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ServiceServer":
        return cls()

    @classmethod
    def main(cls, argv: Optional[list] = None) -> None:
        """Console entry point: parse args, build, load and :meth:`serve`."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        parser = argparse.ArgumentParser(description=f"{cls.__name__} service")
        parser.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0)")
        parser.add_argument(
            "--port", "-p", type=int, default=cls.default_port,
            help=f"Bind port (default {cls.default_port})",
        )
        cls.add_arguments(parser)
        args = parser.parse_args(argv)
        cls.from_args(args).serve(host=args.host, port=args.port)
