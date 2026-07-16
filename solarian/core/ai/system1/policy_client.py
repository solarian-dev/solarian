"""
PolicyClient — generic HTTP client for a :class:`PolicyServer`.

The client is deliberately policy-agnostic: it serialises an observation
dict to JSON, posts it to a policy server, and deserialises the returned
action.  It works with any policy hosted behind the
:class:`~solarian.core.ai.system1.policy_server.PolicyServer` HTTP contract,
whether that policy is a deep-learning model (diffusion / ACT / VLA) or a
classical method (fixed waypoints / SLAM / motion planner).

Usage::

    from solarian.core.ai.system1.policy_client import PolicyClient

    client = PolicyClient(server_url="http://localhost:8001")
    action = client.predict(obs_dict)

The server may run anywhere — co-located with the robot, the relay server,
the controller, or on a dedicated GPU box — so ``server_url`` is the only
thing the skill needs to know about its location.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)


def _serialize_obs(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy arrays in an obs dict to JSON-serialisable lists."""
    out: Dict[str, Any] = {}
    for key, value in obs_dict.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


class PolicyClient:
    """Client for communicating with a policy server."""

    def __init__(
        self,
        server_url: str = "http://localhost:8001",
        timeout: float = 30.0,
        connect_retries: int = 0,
        retry_interval: float = 1.0,
    ):
        """
        Args:
            server_url: Base URL of the policy server.
            timeout: Per-request timeout in seconds.
            connect_retries: How many times to retry the initial health
                check before giving up (lets the skill start before the
                server is up). ``0`` means a single best-effort check.
            retry_interval: Seconds to wait between connection attempts.
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._check_connection(connect_retries, retry_interval)

    def _check_connection(self, connect_retries: int, retry_interval: float) -> None:
        """Best-effort health check, optionally retrying until the server is up."""
        attempts = max(1, connect_retries + 1)
        for attempt in range(attempts):
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5.0)
                response.raise_for_status()
                logger.info("Connected to policy server at %s", self.server_url)
                return
            except requests.exceptions.RequestException as exc:
                if attempt < attempts - 1:
                    logger.info(
                        "Policy server at %s not ready (attempt %d/%d): %s; retrying in %.1fs",
                        self.server_url,
                        attempt + 1,
                        attempts,
                        exc,
                        retry_interval,
                    )
                    time.sleep(retry_interval)
                else:
                    logger.warning(
                        "Could not connect to policy server at %s: %s",
                        self.server_url,
                        exc,
                    )

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded policy."""
        response = requests.get(f"{self.server_url}/model_info", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_shape_meta(self) -> Dict[str, Any]:
        """Return the policy's observation / action shape specification."""
        response = requests.get(f"{self.server_url}/shape_meta", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def reset(self) -> None:
        """Reset the policy's per-episode state."""
        response = requests.post(f"{self.server_url}/reset", timeout=self.timeout)
        response.raise_for_status()
        logger.debug("Policy reset successfully")

    def reload(self, timeout: Optional[float] = None) -> None:
        """Ask the server to reload its policy from disk (can be slow)."""
        response = requests.post(
            f"{self.server_url}/reload", timeout=timeout or max(self.timeout, 120.0)
        )
        response.raise_for_status()
        logger.info("Policy reloaded on server")

    def health_check(self) -> bool:
        """Return ``True`` if the server is healthy, ``False`` otherwise."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5.0)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False

    def _post_predict(self, obs_dict: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        payload = {"obs_dict": _serialize_obs(obs_dict), "kwargs": kwargs}
        response = requests.post(
            f"{self.server_url}/predict", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def predict(self, obs_dict: Dict[str, np.ndarray], **kwargs: Any) -> np.ndarray:
        """Predict an action from an observation.

        Args:
            obs_dict: Observation keyed by name, values are numpy arrays.
            **kwargs: Optional policy-specific parameters forwarded to the
                server's ``predict`` (ignored by policies that don't use them).

        Returns:
            The predicted action as a numpy array.
        """
        result = self._post_predict(obs_dict, **kwargs)
        return np.array(result["action"])

    def predict_with_raw(
        self, obs_dict: Dict[str, np.ndarray], **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict, returning both the processed and raw (model-space) actions."""
        result = self._post_predict(obs_dict, **kwargs)
        action = np.array(result["action"])
        raw_action = np.array(result.get("raw_action", result["action"]))
        return action, raw_action

    def predict_full(self, obs_dict: Dict[str, np.ndarray], **kwargs: Any) -> Dict[str, Any]:
        """Predict, returning the full server response (action, raw_action, info)."""
        result = self._post_predict(obs_dict, **kwargs)
        out: Dict[str, Any] = {"action": np.array(result["action"])}
        if result.get("raw_action") is not None:
            out["raw_action"] = np.array(result["raw_action"])
        out["info"] = result.get("info")
        return out
