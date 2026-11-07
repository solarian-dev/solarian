"""What the agent sees, and the way back from its pixels to the camera's.

A planner points by naming a pixel in an image it was shown. That image was
downscaled (and drawn on) for the model, so every point has to be mapped back
to capture pixels before it seeds a tracker — and it is only meaningful for
the frame it was chosen on. Each :class:`Frame` therefore carries an id the
planner quotes back, and each :class:`View` the scale that undoes its resize.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from solarian.core.ai.agent.model import ImagePart
from solarian.core.ui.base import UIImage
from solarian.core.ui.overlay import MASK_COLOR, mask_rings, tint_masks

_GRID_COLOR = (255, 255, 255)
_POINT_COLOR = (0, 200, 255)
#: Cyan: unlike red or orange, not the colour of a tracker tint or of the so106 fingers.
_TARGET_COLOR = (0, 255, 255)
#: Pixels between the target's bounds and its box; JPEG ringing spreads about 2.
_TARGET_MARGIN = 3


@dataclass
class View:
    """One camera image at model resolution, overlays drawn."""

    #: ``module/field`` of the camera image.
    label: str
    #: ``(width, height)`` of the capture.
    capture_size: tuple[int, int]
    #: Model pixels per capture pixel.
    scale: float
    rgb: np.ndarray
    jpeg: bytes

    @property
    def module(self) -> str:
        return self.label.partition("/")[0]

    @property
    def size(self) -> tuple[int, int]:
        return int(self.rgb.shape[1]), int(self.rgb.shape[0])

    def contains(self, point) -> bool:
        w, h = self.size
        return 0 <= float(point[0]) < w and 0 <= float(point[1]) < h

    def to_capture(self, point) -> list[float]:
        w, h = self.capture_size
        x = min(max(float(point[0]) / self.scale, 0.0), w - 1.0)
        y = min(max(float(point[1]) / self.scale, 0.0), h - 1.0)
        return [round(x, 1), round(y, 1)]


@dataclass
class Frame:
    """Everything observed at one instant, under one id the planner can cite."""

    obs_id: str
    #: ``time.monotonic()`` when it was captured, for staleness.
    t_mono: float
    #: Synced wall clock, for the run log.
    t: float
    views: list[View]
    state: dict = field(default_factory=dict)

    def view(self, module: str | None = None) -> View | None:
        """The view of ``module``'s camera, or the first one when ``module`` is None."""
        for view in self.views:
            if module is None or view.module == module:
                return view
        return None

    def image_parts(self) -> list[ImagePart]:
        return [ImagePart(jpeg=v.jpeg, label=f"{self.obs_id} {v.label} ({v.size[0]}x{v.size[1]} px)") for v in self.views]

    def age_s(self) -> float:
        return time.monotonic() - self.t_mono


def _draw_overlays(image: UIImage) -> np.ndarray:
    rgb = tint_masks(np.ascontiguousarray(image.data), image.masks)
    if rgb is image.data:
        rgb = rgb.copy()
    for mask in image.masks:
        for ring in mask_rings(mask):
            cv2.polylines(rgb, [ring.reshape(-1, 1, 2)], True, MASK_COLOR, 2)
    for box in image.bboxes:
        if box is None or not np.any(box):
            continue
        x0, y0, x1, y1 = (int(round(float(v))) for v in list(box)[:4])
        cv2.rectangle(rgb, (x0, y0), (x1, y1), MASK_COLOR, 2)
    return rgb


def _draw_grid(rgb: np.ndarray, step: int = 100) -> None:
    """Faint lines every ``step`` model pixels, labelled, so a point can be read off."""
    h, w = rgb.shape[:2]
    overlay = rgb.copy()
    for x in range(step, w, step):
        cv2.line(overlay, (x, 0), (x, h - 1), _GRID_COLOR, 1)
    for y in range(step, h, step):
        cv2.line(overlay, (0, y), (w - 1, y), _GRID_COLOR, 1)
    cv2.addWeighted(overlay, 0.35, rgb, 0.65, 0, dst=rgb)
    for x in range(step, w, step):
        cv2.putText(rgb, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, _GRID_COLOR, 1, cv2.LINE_AA)
    for y in range(step, h, step):
        cv2.putText(rgb, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, _GRID_COLOR, 1, cv2.LINE_AA)


def encode_jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def make_view(image: UIImage, max_side: int, grid: bool, overlays: bool = True,
              target: tuple[float, float, float, float] | None = None) -> View:
    """``target``: a box in capture pixels to mark with :func:`mark_target`."""
    capture_h, capture_w = image.data.shape[:2]
    scale = min(1.0, float(max_side) / float(max(capture_w, capture_h)))
    rgb = _draw_overlays(image) if overlays else np.ascontiguousarray(image.data)
    if scale < 1.0:
        size = (max(1, int(round(capture_w * scale))), max(1, int(round(capture_h * scale))))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        scale = rgb.shape[1] / float(capture_w)
    if target is not None:
        rgb = mark_target(rgb, [v * scale for v in target])
    if grid:
        _draw_grid(rgb)
    return View(
        label=image.label,
        capture_size=(int(capture_w), int(capture_h)),
        scale=scale,
        rgb=rgb,
        jpeg=encode_jpeg(rgb),
    )


def resized(view: View, max_side: int) -> ImagePart:
    """A smaller copy of ``view`` for a model that needs less detail (the monitor)."""
    w, h = view.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    rgb = view.rgb
    if scale < 1.0:
        rgb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return ImagePart(jpeg=encode_jpeg(rgb, quality=80), label=view.label)


def close_up(image: UIImage, region, target: tuple[float, float, float, float] | None = None) -> ImagePart:
    """``region`` (``x0, y0, x1, y1`` fractions) of the capture at capture
    resolution, without overlays, so a tracker mask that slipped off the object
    cannot hide it. ``target``: a box in capture pixels to mark with :func:`mark_target`."""
    h, w = image.data.shape[:2]
    left, top = int(w * region[0]), int(h * region[1])
    rgb = np.ascontiguousarray(image.data[top:int(h * region[3]), left:int(w * region[2])])
    if target is not None:
        x0, y0, x1, y1 = target
        rgb = mark_target(rgb, (x0 - left, y0 - top, x1 - left, y1 - top))
    return ImagePart(jpeg=encode_jpeg(rgb, quality=85), label=f"{image.label} close-up")


def target_box(image: UIImage) -> tuple[float, float, float, float] | None:
    """Where the tracker has its target, as ``x0, y0, x1, y1`` capture pixels:
    the bounds of its first mask, else its first box; None when it has neither."""
    for mask in image.masks:
        ys, xs = np.nonzero(mask)
        if len(xs):
            return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    for box in image.bboxes:
        if box is not None and np.any(box):
            x0, y0, x1, y1 = (float(v) for v in list(box)[:4])
            return x0, y0, x1, y1
    return None


def mark_target(rgb: np.ndarray, box) -> np.ndarray:
    """``rgb`` with a 1 px box drawn ``_TARGET_MARGIN`` px outside ``box``
    (inclusive ``x0, y0, x1, y1``), so the line never covers the object's own
    pixels. A box wholly off the image draws nothing."""
    out = rgb.copy()
    h, w = out.shape[:2]
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    x0, y0, x1, y1 = x0 - _TARGET_MARGIN, y0 - _TARGET_MARGIN, x1 + _TARGET_MARGIN, y1 + _TARGET_MARGIN
    if x1 < 0 or y1 < 0 or x0 >= w or y0 >= h:
        return out
    cv2.rectangle(out, (x0, y0), (x1, y1), _TARGET_COLOR, 1, cv2.LINE_8)
    return out


def mark_point(rgb: np.ndarray, point, text: str) -> np.ndarray:
    """``rgb`` with a crosshair at ``point`` (model pixels), for the agent's view panel."""
    out = rgb.copy()
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    cv2.drawMarker(out, (x, y), _POINT_COLOR, cv2.MARKER_CROSS, 22, 2)
    cv2.circle(out, (x, y), 9, _POINT_COLOR, 2)
    cv2.putText(out, text, (x + 12, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _POINT_COLOR, 1, cv2.LINE_AA)
    return out
