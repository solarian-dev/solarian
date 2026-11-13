"""
Camera backend registry — discovery of plugin camera backends via entry points.

Camera plugins whose cameras OpenCV cannot enumerate (the Pi's CSI cameras,
for instance) advertise a backend module under the ``solarian.camera_backends``
entry-point group, e.g. in a plugin's ``pyproject.toml``::

    [project.entry-points."solarian.camera_backends"]
    picamera2 = "solarian_picamera.backend"

The entry-point name is the ``backend`` value the plugin's cameras report.
The loaded object provides:

- ``find() -> list[dict]``: one ``{"port", "backend", "model"}`` entry per camera
- ``iter_frames(port, width=None, height=None)``: yields BGR frames

``solarian list-cameras`` and ``calibrate-umi-camera`` use these so they can
list and preview a plugin's cameras without core importing the vendor SDK.
"""

import importlib.metadata
import logging
from types import ModuleType
from typing import Dict

logger = logging.getLogger(__name__)

#: Entry-point group camera backends register under.
CAMERA_BACKEND_ENTRY_POINT_GROUP = "solarian.camera_backends"


def discover_camera_backends() -> Dict[str, ModuleType]:
    """Return ``{backend_name: backend}`` for installed plugins."""
    out: Dict[str, ModuleType] = {}
    for ep in importlib.metadata.entry_points(group=CAMERA_BACKEND_ENTRY_POINT_GROUP):
        try:
            out[ep.name] = ep.load()
        except Exception:  # noqa: BLE001 - a broken plugin must not kill discovery
            logger.exception("Failed to load camera backend entry point %r", ep.name)
    return out
