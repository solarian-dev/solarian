"""Fisheye distortion for MuJoCo-rendered camera frames.

MuJoCo renders a rectilinear (pinhole) image. To make a simulated camera match
a real fisheye lens (e.g. a GoPro with a fisheye mod) for sim-to-real training,
we warp the pinhole render into a distorted image using the real camera's
calibrated OpenCV fisheye intrinsics ``K`` and coefficients ``D``.

The warp is a static pixel remap: for each *output* (distorted) pixel we find
the ray it corresponds to via ``cv2.fisheye.undistortPoints`` and reproject that
ray through the pinhole *source* camera (whose focal length comes from the
MuJoCo camera's vertical FOV). The resulting ``(map1, map2)`` are built once and
applied per frame with ``cv2.remap``.

Because a pinhole source can only represent rays with angle ``< fovy/2`` from
the optical axis, the MuJoCo camera ``fovy`` must be wider than the fisheye's
max ray angle (and cannot reach 180). Output pixels whose ray falls outside the
source FOV map to black, like the dark corners of a real circular fisheye.
"""

from __future__ import annotations

import math

import numpy as np


def _scaled_K(K: np.ndarray, calib_w: int, calib_h: int, out_w: int, out_h: int) -> np.ndarray:
    """Rescale calibrated intrinsics from the calibration resolution to output."""
    sx = out_w / float(calib_w)
    sy = out_h / float(calib_h)
    scaled = K.astype(np.float64).copy()
    scaled[0, 0] *= sx  # fx
    scaled[0, 2] *= sx  # cx
    scaled[1, 1] *= sy  # fy
    scaled[1, 2] *= sy  # cy
    return scaled


def build_fisheye_maps(
    dist_cfg,
    fovy_deg: float,
    out_w: int,
    out_h: int,
    src_w: int,
    src_h: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build the (map1, map2) that warp a pinhole render into a fisheye image.

    Args:
        dist_cfg: A ``FisheyeDistortionConfig`` (has ``K``, ``D``,
            ``calib_width``, ``calib_height``).
        fovy_deg: Vertical FOV (degrees) of the MuJoCo pinhole source camera.
        out_w, out_h: Size of the distorted output image.
        src_w, src_h: Size of the pinhole source render (>= output when
            supersampling).

    Returns:
        ``(map1, map2, outside_frac)`` where ``map1``/``map2`` are ``float32``
        arrays of shape ``(out_h, out_w)`` for ``cv2.remap`` (source x/y per
        output pixel), and ``outside_frac`` is the fraction of output pixels
        whose ray falls outside the source FOV (so they render black).
    """
    import cv2

    K_out = _scaled_K(
        np.asarray(dist_cfg.K, dtype=np.float64),
        int(dist_cfg.calib_width),
        int(dist_cfg.calib_height),
        out_w,
        out_h,
    )
    D = np.asarray(dist_cfg.D, dtype=np.float64).reshape(-1, 1)

    # Output pixel grid -> normalized ray directions (x, y) on the z=1 plane.
    us, vs = np.meshgrid(np.arange(out_w), np.arange(out_h))
    pts = np.stack([us.ravel(), vs.ravel()], axis=1).astype(np.float64)[:, None, :]
    norm = cv2.fisheye.undistortPoints(pts, K_out, D)  # (N, 1, 2)
    x = norm[:, 0, 0]
    y = norm[:, 0, 1]

    # Pinhole source intrinsics from the MuJoCo vertical FOV (square pixels).
    f = (src_h / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    cx = src_w / 2.0
    cy = src_h / 2.0
    u_s = f * x + cx
    v_s = f * y + cy

    map1 = u_s.reshape(out_h, out_w).astype(np.float32)
    map2 = v_s.reshape(out_h, out_w).astype(np.float32)

    # A ray beyond the source half-FOV lands outside the render (black corner).
    inside = (map1 >= 0) & (map1 <= src_w - 1) & (map2 >= 0) & (map2 <= src_h - 1)
    outside_frac = float(1.0 - inside.mean())

    return map1, map2, outside_frac


def apply(img: np.ndarray, map1: np.ndarray, map2: np.ndarray) -> np.ndarray:
    """Remap a pinhole render into the fisheye-distorted output image."""
    import cv2

    return cv2.remap(
        img,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
