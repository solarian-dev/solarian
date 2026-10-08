"""Service plugins: a model served over HTTP plus the processor that calls it.

A *service* computes something an observation is enriched with — a tracked
bounding box, a depth map — on a machine that can afford it. A plugin ships:

* a :class:`~solarian.services.server.ServiceServer` subclass, run with
  ``solarian run-service <name>`` on the GPU host;
* a :class:`~solarian.services.processor.ServiceProcessor` subclass, declared
  as a ``CalculatedField`` so the observation carries its result;
* a :class:`Service` describing both, advertised under the
  ``solarian.services`` entry-point group::

      [project.entry-points."solarian.services"]
      sam2-tracker = "solarian_sam2_tracker.service:Sam2TrackerService"
"""

import importlib.metadata
import logging

logger = logging.getLogger(__name__)

SERVICE_ENTRY_POINT_GROUP = "solarian.services"


class Service:
    """Describes one service plugin; subclass and set the class attributes."""

    #: Module run with ``python -m`` to serve it; must call ``ServiceServer.main``.
    server_module: str = ""
    #: Port the server binds when ``run-service`` is not given one.
    default_port: int = 8765
    #: The ``ServiceProcessor`` subclass a ``CalculatedField`` declares.
    processor_cls: type | None = None


def discover_services() -> dict[str, type[Service]]:
    """Return ``{entry_point_name: Service subclass}`` for installed plugins."""
    out: dict[str, type[Service]] = {}
    for ep in importlib.metadata.entry_points(group=SERVICE_ENTRY_POINT_GROUP):
        try:
            out[ep.name] = ep.load()
        except Exception:  # noqa: BLE001 - a broken plugin must not kill discovery
            logger.exception("Failed to load service entry point %r", ep.name)
    return out
