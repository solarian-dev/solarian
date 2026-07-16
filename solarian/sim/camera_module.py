"""Simulated camera module.

Mirror of :class:`~solarian.core.opencv_module.OpenCVCameraModule` whose sensor
renders a MuJoCo camera instead of an OpenCV device. Reuses
``OpenCVCameraConfig`` so the sim camera config yaml is parsed identically.
"""

from __future__ import annotations

from solarian.core.opencv_module import OpenCVCameraModule, OpenCVCameraConfig

from solarian.sim.camera import SimCamera
from solarian.sim.module import SimModule


class SimCameraModule(OpenCVCameraModule, SimModule):
    """MuJoCo-rendered camera module (drop-in for ``OpenCVCameraModule``)."""

    config_class = OpenCVCameraConfig

    camera = SimCamera()
