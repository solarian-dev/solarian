"""Service plugins: a model served over HTTP plus whatever calls it.

A *service* computes something on a machine that can afford it: a tracked
bounding box or a depth map an observation is enriched with, or a policy's
next action. A plugin ships:

* a :class:`~solarian.services.server.ServiceServer` subclass, run with
  ``solarian run-service`` on the GPU host (a policy plugin's is a
  :class:`~solarian.services.policy.PolicyServer`);
* the caller: a :class:`~solarian.services.processor.ServiceProcessor`
  subclass declared as a ``CalculatedField``, or a skill driving a
  :class:`~solarian.services.policy.PolicyClient`;
* a :class:`Service` describing the server, advertised under the
  ``solarian.services`` entry-point group::

      [project.entry-points."solarian.services"]
      sam2-tracker = "solarian_sam2.service:Sam2TrackerService"
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solarian.core.ai.system1.policy_trainer import ModalSpec

logger = logging.getLogger(__name__)

SERVICE_ENTRY_POINT_GROUP = "solarian.services"


class Service:
    """Describes one service plugin; subclass and set the class attributes."""

    #: Module run with ``python -m`` to serve it; must call ``ServiceServer.main``.
    server_module: str = ""
    #: Port the server binds when ``run-service`` is not given one.
    default_port: int = 8765
    #: The ``ServiceProcessor`` subclass a ``CalculatedField`` declares, if any.
    processor_cls: type | None = None
    #: The server's flag taking the checkpoint to load (``--input`` for a
    #: policy); None for a service that is not given one.
    checkpoint_arg: str | None = None

    @classmethod
    def modal_spec(cls) -> ModalSpec | None:
        """The image to serve it in on Modal; None derives one from the
        installed requirements of the server's distribution."""
        return None


def discover_services() -> dict[str, type[Service]]:
    """Return ``{entry_point_name: Service subclass}`` for installed plugins."""
    out: dict[str, type[Service]] = {}
    for ep in importlib.metadata.entry_points(group=SERVICE_ENTRY_POINT_GROUP):
        try:
            out[ep.name] = ep.load()
        except Exception:  # noqa: BLE001 - a broken plugin must not kill discovery
            logger.exception("Failed to load service entry point %r", ep.name)
    return out
