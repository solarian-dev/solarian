"""QR encode/decode used by ``calibrate-camera-latency``."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from solarian.cli.commands.calibrate_camera_latency import (
    _decode_qr,
    _opencv_bgr,
    _qr_detectors,
    _render_qr,
)

_STAMP = 1234567890.123456


def _paste_qr(canvas_hw: tuple[int, int], qr_px: int) -> np.ndarray:
    canvas = np.full(canvas_hw, 255, np.uint8)
    qr = _render_qr(f"{_STAMP:.6f}", qr_px)
    y = (canvas_hw[0] - qr_px) // 2
    x = (canvas_hw[1] - qr_px) // 2
    canvas[y : y + qr_px, x : x + qr_px] = qr
    return canvas


def test_roundtrip_fullscreen_qr():
    detectors = _qr_detectors()
    frame = _render_qr(f"{_STAMP:.6f}", 720)
    assert _decode_qr(detectors, frame) == pytest.approx(_STAMP)


def test_roundtrip_rgb_observation():
    """Wrist cameras publish RGB; gray must not depend on BGR2GRAY."""
    detectors = _qr_detectors()
    gray = _render_qr(f"{_STAMP:.6f}", 720)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    assert _decode_qr(detectors, rgb) == pytest.approx(_STAMP)


def test_opencv_bgr_swaps_rgb_observations():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    bgr = _opencv_bgr(rgb, "rgb")
    assert bgr[0, 0, 0] == 0
    assert bgr[0, 0, 2] == 255
    same = _opencv_bgr(rgb, "bgr")
    assert same[0, 0, 0] == 255


@pytest.mark.skipif(
    not hasattr(cv2, "QRCodeDetectorAruco"),
    reason="QRCodeDetectorAruco reads the small codes QRCodeDetector misses",
)
def test_decode_small_qr_on_640x360_observation():
    """After remap+resize the code can occupy ~90 px of a 640x360 frame."""
    detectors = _qr_detectors()
    frame = _paste_qr((360, 640), 90)
    assert _decode_qr(detectors, frame) == pytest.approx(_STAMP)
