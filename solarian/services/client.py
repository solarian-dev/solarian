"""ServiceClient — HTTP client for a :class:`~solarian.services.server.ServiceServer`."""

import json
import logging
import time
from typing import Any, Dict, Optional

import requests

from solarian.services.server import SERVER_MS_HEADER
from solarian.utils.wire import decode_arrays, encode_arrays

logger = logging.getLogger(__name__)


class ServiceClient:
    def __init__(self, server_url: str, timeout: float = 5.0, token: Optional[str] = None):
        """``token`` is the server's bearer token: a hosted endpoint's
        ``solarian_base_endpoint_token``, or None for an open server."""
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        #: Breakdown of the last :meth:`post`, in ms, plus the request size in
        #: MB. encode is this host's CPU, server the far end's, and transport
        #: whatever is left of the round trip: the link and both ends' HTTP.
        self.last_timing: Dict[str, float] = {
            "encode_ms": 0.0,
            "server_ms": 0.0,
            "transport_ms": 0.0,
            "payload_mb": 0.0,
        }
        #: Pooled so a service called at camera rate is not paying TCP setup,
        #: and mDNS resolution of the server's host, on every frame.
        self._http = requests.Session()
        if token:
            self._http.headers["Authorization"] = f"Bearer {token}"

    def health_check(self) -> bool:
        try:
            self._http.get(f"{self.server_url}/health", timeout=5.0).raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False

    def wait_until_ready(self, retries: int = 0, interval: float = 1.0) -> bool:
        """Health-check up to ``retries + 1`` times, ``interval`` seconds apart.

        Lets a caller start before its server is up, or while a hosted one
        scales up from zero. Returns whether the server answered; a server
        that never does is logged, not raised, so the first real call reports
        the actual error.
        """
        attempts = max(1, retries + 1)
        for attempt in range(attempts):
            try:
                self._http.get(f"{self.server_url}/health", timeout=5.0).raise_for_status()
                logger.info("Connected to %s", self.server_url)
                return True
            except requests.exceptions.RequestException as exc:
                if attempt < attempts - 1:
                    logger.info(
                        "%s not ready (attempt %d/%d): %s; retrying in %.1fs",
                        self.server_url, attempt + 1, attempts, exc, interval,
                    )
                    time.sleep(interval)
                else:
                    logger.warning("Could not connect to %s: %s", self.server_url, exc)
        return False

    def get(self, path: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        response = self._http.get(
            f"{self.server_url}/{path.lstrip('/')}", timeout=timeout or self.timeout
        )
        self._raise_for_status(response, path)
        return response.json()

    def info(self) -> Dict[str, Any]:
        return self.get("/info")

    def post(
        self, path: str, payload: Dict[str, Any], timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """POST ``payload`` (numpy arrays allowed) and return the decoded reply."""
        # Encoded here rather than by ``json=`` so the cost and the size are
        # measurable. On a small board an image serialised to decimal text is
        # not a rounding error against the model it is being sent to.
        t0 = time.perf_counter()
        body = json.dumps(encode_arrays(payload))
        t1 = time.perf_counter()
        response = self._http.post(
            f"{self.server_url}/{path.lstrip('/')}",
            data=body,
            headers={"Content-Type": "application/json"},
            timeout=timeout or self.timeout,
        )
        self._raise_for_status(response, path)
        result = decode_arrays(response.json())
        server_ms = float(response.headers.get(SERVER_MS_HEADER) or 0.0)
        self.last_timing = {
            "encode_ms": (t1 - t0) * 1000.0,
            "server_ms": server_ms,
            "transport_ms": (time.perf_counter() - t1) * 1000.0 - server_ms,
            "payload_mb": len(body) / 1e6,
        }
        return result

    @staticmethod
    def _raise_for_status(response: requests.Response, path: str) -> None:
        if not response.ok:
            raise requests.HTTPError(
                f"{response.status_code} on /{path.lstrip('/')}: {response.text[:300]}",
                response=response,
            )
