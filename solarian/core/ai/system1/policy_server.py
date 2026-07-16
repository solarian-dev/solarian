"""
PolicyServer — base class for hosting a policy behind a small HTTP API.

A *policy server* loads a policy (a deep-learning model such as diffusion,
ACT or a VLA, **or** a classical method such as fixed waypoints or a SLAM
planner) and exposes it over HTTP so that a :class:`Skill` can drive it
remotely via :class:`~solarian.core.ai.system1.policy_client.PolicyClient`.

The base class hides FastAPI + uvicorn so plugin authors only implement a
handful of methods:

* :meth:`load`        — load the model / initialise the method (required)
* :meth:`predict`     — map an observation dict to an action (required)
* :attr:`model_info`  — metadata describing the loaded policy (required)
* :attr:`shape_meta`  — observation / action shape spec (required)
* :meth:`reset`       — clear per-episode state (optional)
* :meth:`reload`      — reload weights from disk (optional)

The HTTP contract (see :class:`PolicyClient`) is intentionally tiny and
policy-agnostic::

    GET  /            → liveness banner
    GET  /health      → {"status", "model_loaded"}
    GET  /model_info  → arbitrary JSON metadata dict
    GET  /shape_meta  → arbitrary JSON shape spec
    POST /reset       → clear episode state
    POST /reload      → reload from disk
    POST /predict     → {"obs_dict": {...}, "kwargs": {...}}
                         → {"action": [...], "raw_action": [...]|null,
                            "info": {...}|null}

Observations and actions cross the wire as plain (nested) JSON lists; the
base class converts to/from ``numpy`` so subclasses always see and return
``np.ndarray``.

Typical plugin usage::

    class MyPolicyServer(PolicyServer):
        def load(self): ...
        def predict(self, obs_dict, **kwargs): ...
        @property
        def model_info(self): ...
        @property
        def shape_meta(self): ...

        @classmethod
        def add_arguments(cls, parser):
            parser.add_argument("--checkpoint", required=True)

        @classmethod
        def from_args(cls, args):
            return cls(checkpoint=args.checkpoint)

    if __name__ == "__main__":
        MyPolicyServer.main()
"""

import abc
import argparse
import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _to_numpy_obs(obs_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert a JSON-decoded observation dict back into numpy arrays."""
    out: Dict[str, Any] = {}
    for key, value in obs_dict.items():
        if isinstance(value, np.ndarray):
            out[key] = value
        elif isinstance(value, list):
            out[key] = np.array(value)
        else:
            out[key] = value
    return out


def _to_serializable(value: Any) -> Any:
    """Recursively turn numpy types into JSON-serialisable Python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


class PolicyServer(abc.ABC):
    """Abstract base class for policy servers.

    Subclasses implement :meth:`load`, :meth:`predict`, :attr:`model_info`
    and :attr:`shape_meta`.  Run a server with :meth:`serve` (or the
    :meth:`main` CLI helper); a client talks to it via
    :class:`PolicyClient`.
    """

    #: Title advertised by the FastAPI app (override in subclasses if desired).
    title: str = "Solarian Policy Server"

    def __init__(self, **config: Any):
        self.config: Dict[str, Any] = dict(config)
        self._loaded = False

    # ------------------------------------------------------------------
    # Methods to implement
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load(self) -> None:
        """Load the policy (checkpoint, planner, …).

        Called once by :meth:`setup` before the server starts accepting
        requests.  Must populate whatever internal state :meth:`predict`
        needs.
        """

    @abc.abstractmethod
    def predict(self, obs_dict: Dict[str, np.ndarray], **kwargs: Any) -> Dict[str, Any]:
        """Run inference for a single observation.

        Args:
            obs_dict: Mapping of observation keys to ``np.ndarray`` (no
                batch dimension; shapes follow :attr:`shape_meta`).
            **kwargs: Optional policy-specific parameters forwarded from
                the client (ignored by policies that don't use them).

        Returns:
            A dict with at least an ``"action"`` ``np.ndarray``.  May also
            include ``"raw_action"`` (model-space output before any
            post-processing) and ``"info"`` (a JSON-serialisable dict of
            diagnostics).
        """

    def reset(self) -> None:
        """Clear per-episode state. Default is a no-op."""

    def reload(self) -> None:
        """Reload the policy from disk. Default re-runs :meth:`load`."""
        self.load()

    @property
    @abc.abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """Arbitrary JSON-serialisable metadata about the loaded policy."""

    @property
    @abc.abstractmethod
    def shape_meta(self) -> Dict[str, Any]:
        """Observation / action shape specification (JSON-serialisable)."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        """Whether :meth:`load` has completed successfully."""
        return self._loaded

    def setup(self) -> None:
        """Idempotently :meth:`load` the policy."""
        if not self._loaded:
            self.load()
            self._loaded = True

    # ------------------------------------------------------------------
    # HTTP app
    # ------------------------------------------------------------------

    def build_app(self):
        """Build and return a FastAPI app wired to this server's methods.

        FastAPI / pydantic are imported lazily so that merely importing a
        plugin's skill module (e.g. for discovery on the controller side)
        does not require the web stack to be installed.
        """
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel

        server = self

        class PredictRequest(BaseModel):
            obs_dict: Dict[str, Any]
            kwargs: Dict[str, Any] = {}

        app = FastAPI(title=self.title, version="1.0.0")

        @app.get("/")
        async def root():
            return {"message": f"{server.title} is running", "status": "ok"}

        @app.get("/health")
        async def health():
            if not server.loaded:
                raise HTTPException(status_code=503, detail="Policy not loaded")
            return {"status": "healthy", "model_loaded": True}

        @app.get("/model_info")
        async def model_info():
            if not server.loaded:
                raise HTTPException(status_code=503, detail="Policy not loaded")
            return _to_serializable(server.model_info)

        @app.get("/shape_meta")
        async def shape_meta():
            if not server.loaded:
                raise HTTPException(status_code=503, detail="Policy not loaded")
            return _to_serializable(server.shape_meta)

        @app.post("/reset")
        async def reset():
            if not server.loaded:
                raise HTTPException(status_code=503, detail="Policy not loaded")
            try:
                server.reset()
                return {"status": "success", "message": "Policy reset"}
            except Exception as exc:  # noqa: BLE001 - surface to client
                logger.exception("Error during reset")
                raise HTTPException(status_code=500, detail=str(exc))

        @app.post("/reload")
        async def reload():
            try:
                server.reload()
                server._loaded = True
                return {"status": "success", "message": "Policy reloaded"}
            except Exception as exc:  # noqa: BLE001 - surface to client
                logger.exception("Error during reload")
                raise HTTPException(status_code=500, detail=str(exc))

        @app.post("/predict")
        async def predict(request: PredictRequest):
            if not server.loaded:
                raise HTTPException(status_code=503, detail="Policy not loaded")
            try:
                obs_dict = _to_numpy_obs(request.obs_dict)
                result = server.predict(obs_dict, **(request.kwargs or {}))
                if not isinstance(result, dict) or "action" not in result:
                    raise ValueError(
                        "PolicyServer.predict must return a dict containing 'action'"
                    )
                action = result["action"]
                raw_action = result.get("raw_action", None)
                if raw_action is None:
                    raw_action = action
                return {
                    "action": _to_serializable(action),
                    "raw_action": _to_serializable(raw_action),
                    "info": _to_serializable(result.get("info")),
                }
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001 - surface to client
                logger.exception("Error during inference")
                raise HTTPException(status_code=500, detail=str(exc))

        return app

    def serve(self, host: str = "0.0.0.0", port: int = 8001) -> None:
        """Load the policy and run the HTTP server (blocking).

        Binds ``0.0.0.0`` by default so the server is reachable from other
        hosts (robot / server / controller / dedicated GPU box).  Use
        ``127.0.0.1`` for a co-located, loopback-only deployment.
        """
        import uvicorn

        self.setup()
        app = self.build_app()
        logger.info("Serving %s on http://%s:%d", self.title, host, port)
        uvicorn.run(app, host=host, port=port)

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add subclass-specific CLI arguments. Override as needed."""

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PolicyServer":
        """Build an instance from parsed CLI args. Override as needed."""
        return cls()

    @classmethod
    def main(cls, argv: Optional[list] = None) -> None:
        """Console entry point: parse args, build, load and :meth:`serve`."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        parser = argparse.ArgumentParser(description=f"{cls.__name__} policy server")
        parser.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0)")
        parser.add_argument("--port", "-p", type=int, default=8001, help="Bind port")
        cls.add_arguments(parser)
        args = parser.parse_args(argv)
        server = cls.from_args(args)
        server.serve(host=args.host, port=args.port)
