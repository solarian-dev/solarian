"""Simulated camera sensor.

Renders a named MuJoCo camera instead of capturing from a physical device. It
subclasses :class:`~solarian.core.opencv_camera.OpenCVCamera` so that it shares
the exact same ``rgb_image`` schema descriptor -- this keeps the observation
schema (which references the real camera) correctly sized when a simulated
robot is run.
"""

from __future__ import annotations

import numpy as np

from solarian.core.opencv_camera import OpenCVCamera
from solarian.core.configs import CameraConfig

from solarian.sim.mujoco_bus import MujocoBus


class SimCamera(OpenCVCamera):
    """MuJoCo-rendered camera.

    The MuJoCo camera name to render and the output size are taken from the
    sensor config: the ``sim.camera`` field names the MJCF camera, and the
    render resolution follows ``resize_width``/``resize_height`` (falling back
    to ``width``/``height``), exactly like the OpenCV camera.
    """

    bus_class = MujocoBus

    def __init__(self, cfg: CameraConfig | None = None, name: str | None = None):
        # Initialize the component without OpenCVCamera's own shape logic (it
        # writes to ``type(self).__dict__`` which does not hold ``rgb_image``
        # for this subclass). We set the shape on whichever class in the MRO
        # actually owns the shared descriptor.
        super(OpenCVCamera, self).__init__(cfg)
        if cfg is not None:
            if cfg.resize_width is not None and cfg.resize_height is not None:
                h, w = cfg.resize_height, cfg.resize_width
            elif cfg.rotation in (90, -90):
                h, w = cfg.width, cfg.height
            else:
                h, w = cfg.height, cfg.width
            for klass in type(self).__mro__:
                if "rgb_image" in klass.__dict__:
                    img = klass.__dict__["rgb_image"]
                    img.shape = (h, w, cfg.channels)
                    img.payload = np.zeros((h, w, cfg.channels), dtype=np.uint8)
                    break
