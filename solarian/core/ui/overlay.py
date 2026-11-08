"""A tracked target's mask, drawn the way Solarian Base reviews a dataset's tracking.

The mask is tinted translucent red and edged with its outer and hole rings, the
same rings Solarian Base serves as ``TargetOutlines`` and fills even-odd. Every
front end draws it this way, so a target looks the same live as in review.
"""

from __future__ import annotations

import cv2
import numpy as np

from solarian.core.ui.base import UIImage

#: RGB, Solarian Base's ``#ef4444``.
MASK_COLOR = (239, 68, 68)
MASK_FILL_ALPHA = 0.35
#: RGB.
BBOX_COLOR = (255, 0, 0)


def mask_rings(mask: np.ndarray) -> list[np.ndarray]:
    """The outer and hole contours of a 0/1 ``(H, W)`` mask, each ``(N, 2)`` int32 ``(x, y)``."""
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    return [cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2) for c in contours if len(c) >= 3]


def tint_masks(frame: np.ndarray, masks) -> np.ndarray:
    """``frame`` with every ``frame``-sized mask's pixels blended toward :data:`MASK_COLOR`.

    Returns ``frame`` itself when nothing is tinted.
    """
    out = frame
    for mask in masks:
        inside = np.asarray(mask).astype(bool)
        if not inside.any():
            continue
        if out is frame:
            out = frame.copy()
        out[inside] = (
            out[inside] * (1.0 - MASK_FILL_ALPHA)
            + np.asarray(MASK_COLOR, dtype=np.float32) * MASK_FILL_ALPHA
        ).astype(np.uint8)
    return out


def draw_overlays(image: UIImage) -> np.ndarray:
    """``image``'s RGB frame with its masks tinted and edged and its boxes drawn and labelled."""
    if not image.bboxes and not image.masks:
        return image.data
    frame = tint_masks(image.data, image.masks).copy()
    for mask in image.masks:
        cv2.polylines(frame, mask_rings(mask), True, MASK_COLOR, 2)
    texts = list(image.bbox_texts[:len(image.bboxes)])
    texts += [None] * (len(image.bboxes) - len(texts))
    for bbox, text in zip(image.bboxes, texts):
        if all(value == 0 for value in bbox[:4]):
            continue
        x1, y1, x2, y2 = bbox[:4]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), BBOX_COLOR, 2)
        if text:
            cv2.putText(
                frame, text, (int(x1), max(14, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BBOX_COLOR, 1, cv2.LINE_AA,
            )
    return frame
