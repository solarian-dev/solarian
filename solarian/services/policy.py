"""Policies: services with one standard API, so a skill can drive any of them.

A *policy server* loads a policy (a deep-learning model such as diffusion,
ACT or a VLA, **or** a classical method such as fixed waypoints or a SLAM
planner) and a skill drives it through :class:`PolicyClient`. Because every
policy answers the same routes, a skill switches policies by config::

    GET  /model_info  → arbitrary JSON metadata dict
    GET  /shape_meta  → arbitrary JSON shape spec
    POST /reset       → clear episode state
    POST /reload      → reload from disk
    POST /predict     → {"obs_dict": {...}, "kwargs": {...}}
                         → {"action": [...], "raw_action": [...]|null,
                            "info": {...}|null}

on top of what every :class:`~solarian.services.server.ServiceServer` serves
(``/``, ``/health``, ``/info``, bearer auth, ``X-Server-Ms``).

A policy plugin subclasses :class:`PolicyServer`, implementing :meth:`load`,
:meth:`predict`, :attr:`model_info` and :attr:`shape_meta`, and advertises a
:class:`~solarian.services.service.Service` with ``checkpoint_arg`` set, so
``solarian run-service`` can hand it a checkpoint::

    class MyPolicyServer(PolicyServer):
        def load(self): ...
        def predict(self, obs_dict, **kwargs): ...
        @property
        def model_info(self): ...
        @property
        def shape_meta(self): ...

        @classmethod
        def add_arguments(cls, parser):
            parser.add_argument("--input", "-i", required=True)

        @classmethod
        def from_args(cls, args):
            return cls(checkpoint=args.input)

    if __name__ == "__main__":
        MyPolicyServer.main()
"""

import abc
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from solarian.services.client import ServiceClient
from solarian.services.server import ServiceServer
from solarian.utils.wire import decode_arrays, to_serializable

logger = logging.getLogger(__name__)


class PolicyServer(ServiceServer):
    """A service serving the policy API; subclasses implement the policy."""

    title: str = "Solarian Policy Server"
    default_port: int = 8001

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

    @property
    def info(self) -> Dict[str, Any]:
        return self.model_info

    def add_routes(self, app) -> None:
        from fastapi import HTTPException
        from pydantic import BaseModel

        server = self

        class PredictRequest(BaseModel):
            obs_dict: Dict[str, Any]
            kwargs: Dict[str, Any] = {}

        def require_loaded() -> None:
            # Only after a /reload that failed: serve() loads before listening.
            if not server.loaded:
                raise HTTPException(status_code=503, detail="Policy not loaded")

        @app.get("/model_info")
        async def model_info():
            require_loaded()
            return to_serializable(server.model_info)

        @app.get("/shape_meta")
        async def shape_meta():
            require_loaded()
            return to_serializable(server.shape_meta)

        @app.post("/reset")
        async def reset():
            require_loaded()
            try:
                server.reset()
                return {"status": "success", "message": "Policy reset"}
            except Exception as exc:  # noqa: BLE001 - surface to client
                logger.exception("Error during reset")
                raise HTTPException(status_code=500, detail=str(exc))

        @app.post("/reload")
        async def reload():
            server.loaded = False
            try:
                server.reload()
            except Exception as exc:  # noqa: BLE001 - surface to client
                logger.exception("Error during reload")
                raise HTTPException(status_code=500, detail=str(exc))
            server.loaded = True
            return {"status": "success", "message": "Policy reloaded"}

        @app.post("/predict")
        async def predict(request: PredictRequest):
            require_loaded()
            try:
                result = server.predict(decode_arrays(request.obs_dict), **(request.kwargs or {}))
                if not isinstance(result, dict) or "action" not in result:
                    raise ValueError("PolicyServer.predict must return a dict containing 'action'")
                action = result["action"]
                raw_action = result.get("raw_action")
                return {
                    "action": to_serializable(action),
                    "raw_action": to_serializable(action if raw_action is None else raw_action),
                    "info": to_serializable(result.get("info")),
                }
            except Exception as exc:  # noqa: BLE001 - surface to client
                logger.exception("Error during inference")
                raise HTTPException(status_code=500, detail=str(exc))


class PolicyClient(ServiceClient):
    """Drives any :class:`PolicyServer`.

    The server may run anywhere — co-located with the robot, the relay, the
    controller, a GPU box or Solarian Base — so ``server_url`` (and its
    token) is all a skill needs to know about where it is.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8001",
        timeout: float = 30.0,
        token: Optional[str] = None,
        connect_retries: int = 0,
        retry_interval: float = 1.0,
    ):
        """``connect_retries`` health checks, ``retry_interval`` seconds apart,
        let the skill start before the server is up; ``0`` is one best-effort
        check."""
        super().__init__(server_url, timeout=timeout, token=token)
        self.wait_until_ready(connect_retries, retry_interval)

    def get_model_info(self) -> Dict[str, Any]:
        return self.get("/model_info")

    def get_shape_meta(self) -> Dict[str, Any]:
        return self.get("/shape_meta")

    def reset(self) -> None:
        self.post("/reset", {})

    def reload(self, timeout: Optional[float] = None) -> None:
        """Ask the server to reload its policy from disk (can be slow)."""
        self.post("/reload", {}, timeout=timeout or max(self.timeout, 120.0))

    def predict(self, obs_dict: Dict[str, np.ndarray], **kwargs: Any) -> np.ndarray:
        """The action for ``obs_dict``; ``kwargs`` reach the server's ``predict``."""
        return np.asarray(self.post("/predict", {"obs_dict": obs_dict, "kwargs": kwargs})["action"])

    def predict_with_raw(
        self, obs_dict: Dict[str, np.ndarray], **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict, returning both the processed and raw (model-space) actions."""
        result = self.post("/predict", {"obs_dict": obs_dict, "kwargs": kwargs})
        return np.asarray(result["action"]), np.asarray(result["raw_action"])

    def predict_full(self, obs_dict: Dict[str, np.ndarray], **kwargs: Any) -> Dict[str, Any]:
        """Predict, returning the full server response (action, raw_action, info)."""
        result = self.post("/predict", {"obs_dict": obs_dict, "kwargs": kwargs})
        return {
            "action": np.asarray(result["action"]),
            "raw_action": np.asarray(result["raw_action"]),
            "info": result.get("info"),
        }
