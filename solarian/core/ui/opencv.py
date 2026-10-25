"""OpenCV front end: one window per camera, with click-to-track."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from solarian.core.ui.base import UICommand, UICommandType, UIImage, UISnapshot
from solarian.core.ui.overlay import MASK_COLOR, mask_rings, tint_masks

logger = logging.getLogger(__name__)

_BBOX_COLOR = (255, 0, 0)

# Monitor geometry used to center windows, resolved lazily and at most once.
# ``screeninfo`` on macOS imports pyobjc (AppKit / Foundation / Cocoa) at module
# load, which then breaks any subsequent fork() in the process — Apple's
# fork-safety net, which OBJC_DISABLE_INITIALIZE_FORK_SAFETY does not bypass. A
# process that later forks module workers must not pay that at import time.
_screen_geometry: tuple[int, int, int, int] | None = None


def _get_screen_geometry() -> tuple[int, int, int, int]:
    global _screen_geometry
    if _screen_geometry is not None:
        return _screen_geometry
    _screen_geometry = (0, 0, 0, 0)
    try:
        from screeninfo import get_monitors

        monitor = get_monitors()[0]
        _screen_geometry = (monitor.width, monitor.height, monitor.x, monitor.y)
    except Exception:
        logger.debug(
            "screeninfo unavailable; centering windows against the origin",
            exc_info=True,
        )
    return _screen_geometry


class OpenCVUI:
    """A HighGUI window per camera field, plus click-to-track.

    The oldest of the front ends and the only one with no key capture of its
    own: HighGUI reports a key only while one of its windows is focused and only
    as a press, so the controller starts the global OS listener for it.
    """

    captures_keys = False

    def __init__(self):
        self._windows: list[str] = []
        self._pending: list[UICommand] = []
        self._last_tick = time.perf_counter()

    def poll(self) -> list[UICommand]:
        pending, self._pending = self._pending, []
        return pending

    def render(self, snapshot: UISnapshot) -> None:
        for image in snapshot.images:
            if image.label not in self._windows:
                self._open_window(image)
            frame = self._with_overlays(image)
            cv2.imshow(image.label, np.ascontiguousarray(frame[:, :, ::-1]))

    def tick(self, fps: int) -> None:
        # waitKey is what pumps the HighGUI event loop; without it the windows
        # stop redrawing. Key events come from the global listener, so its
        # return value is read only for the 'q' shortcut.
        dt_s = time.perf_counter() - self._last_tick
        wait_ms = max(1, int(((1.0 / fps) - dt_s) * 1000)) if fps else 1
        if (cv2.waitKey(wait_ms) & 0xFF) == ord("q"):
            self._pending.append(UICommand(UICommandType.QUIT))
        self._last_tick = time.perf_counter()

    def close(self) -> None:
        for name in self._windows:
            try:
                cv2.destroyWindow(name)
            except cv2.error:
                logger.debug("destroyWindow(%s) raised", name, exc_info=True)
        if self._windows:
            cv2.destroyAllWindows()
        self._windows = []

    def _open_window(self, image: UIImage) -> None:
        screen_w, screen_h, screen_x, screen_y = _get_screen_geometry()
        height, width = image.data.shape[:2]
        cv2.namedWindow(image.label, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(image.label, width, height)
        cv2.setMouseCallback(image.label, self._on_mouse, image.label)
        cv2.moveWindow(
            image.label,
            screen_x + (screen_w - width) // 2,
            screen_y + (screen_h - height) // 2,
        )
        self._windows.append(image.label)

    def _on_mouse(self, event, x, y, flags, label) -> None:
        """``(x, y)`` is already in image pixels: OpenCV undoes the window scaling."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        logger.info("Clicked %s at (x=%s, y=%s)", label, x, y)
        self._pending.append(
            UICommand(
                UICommandType.CLICK,
                {"label": label, "point": (x, y), "button": 1},
            )
        )

    @staticmethod
    def _with_overlays(image: UIImage) -> np.ndarray:
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
            cv2.rectangle(
                frame, (int(x1), int(y1)), (int(x2), int(y2)), _BBOX_COLOR, 2
            )
            if text:
                cv2.putText(
                    frame, text, (int(x1), max(14, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BBOX_COLOR, 1, cv2.LINE_AA,
                )
        return frame
