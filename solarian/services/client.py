"""ServiceClient — HTTP client for a :class:`~solarian.services.server.ServiceServer`."""

import json
import logging
import time
from typing import Any, Dict, Optional

import requests

from solarian.utils.wire import decode_arrays, encode_arrays

logger = logging.getLogger(__name__)


class ServiceClient:
    def __init__(self, server_url: str, timeout: float = 5.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        #: Round trip of the last :meth:`post`, in ms.
        self.last_ms = 0.0
        #: Pooled so a service called at camera rate is not paying TCP setup,
        #: and mDNS resolution of the server's host, on every frame.
        self._http = requests.Session()

    def health_check(self) -> bool:
        try:
            self._http.get(f"{self.server_url}/health", timeout=2.0).raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False

    def info(self) -> Dict[str, Any]:
        response = self._http.get(f"{self.server_url}/info", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def post(
        self, path: str, payload: Dict[str, Any], timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """POST ``payload`` (numpy arrays allowed) and return the decoded reply."""
        t0 = time.perf_counter()
        response = self._http.post(
            f"{self.server_url}/{path.lstrip('/')}",
            data=json.dumps(encode_arrays(payload)),
            headers={"Content-Type": "application/json"},
            timeout=timeout or self.timeout,
        )
        if not response.ok:
            raise requests.HTTPError(
                f"{response.status_code} on /{path.lstrip('/')}: {response.text[:300]}",
                response=response,
            )
        result = decode_arrays(response.json())
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return result
