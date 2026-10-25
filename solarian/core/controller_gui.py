"""Single-window controller GUI (pygame).

A :class:`ControllerGUI` is the non-headless front-end for a
:class:`~solarian.core.controller.Controller`. It replaces the split
Rich-console dashboard + one-OpenCV-window-per-camera setup with a single
focused window that shows, side by side:

* a **dashboard** of the latest observation / action fields (text panels),
* a **camera grid** of every image field in the observation, and
* a **keys panel** listing the key bindings and highlighting the keys that are
  currently held.

Crucially, key capture happens through this window's own event queue (pygame),
so it needs **no OS-level accessibility / input-monitoring permission** the way
the global ``pynput`` listener does -- the window just has to be focused. pygame
also reports real key-down / key-up events (and multiple simultaneously held
keys), which the timeout-based OpenCV ``waitKey`` path cannot.

``pygame`` is imported lazily inside :meth:`__init__` so importing this module
(or :mod:`solarian`) never pulls in SDL.
"""

from __future__ import annotations

import os
from itertools import zip_longest
from typing import Callable, Optional, Sequence

import numpy as np

import logging

from solarian.core.ui.overlay import MASK_COLOR, mask_rings, tint_masks

logger = logging.getLogger(__name__)

# Colors (RGB).
_BG = (18, 18, 22)
_PANEL_BG = (30, 30, 38)
_PANEL_BORDER = (70, 70, 90)
_TITLE = (120, 200, 255)
_TEXT = (220, 220, 230)
_DIM = (140, 140, 155)
_KEY_IDLE = (90, 90, 110)
_KEY_HELD = (90, 220, 130)
_KEY_HELD_BG = (35, 70, 45)
_WARN = (240, 200, 80)
_HIGH = (240, 80, 80)

#: Keys pygame reports by name rather than as a character, which controllers
#: nonetheless bind. Without these, ``space`` and ``delete`` never reach a
#: handler. Every other multi-character name (arrows, modifiers, function
#: keys) is still dropped.
_NAMED_KEYS = frozenset({"space", "delete", "backspace"})
LEVEL_COLORS = {"mid": _WARN, "high": _HIGH}
_BBOX = (90, 220, 130)


def display_to_image(
    pos: tuple[int, int],
    rect: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int] | None:
    """Map a window position to pixels of the image drawn at *rect*.

    *rect* is ``(x, y, width, height)`` of the scaled image on screen and
    *image_size* its ``(width, height)`` as captured. ``None`` when *pos* is
    outside the image.
    """
    mx, my = pos
    bx, by, dw, dh = rect
    if not (bx <= mx < bx + dw and by <= my < by + dh):
        return None
    iw, ih = image_size
    return (
        min(iw - 1, int((mx - bx) * iw / dw)),
        min(ih - 1, int((my - by) * ih / dh)),
    )


class ControllerGUI:
    """pygame window rendering the controller dashboard, cameras and keys.

    Args:
        title: Window title.
        width / height: Initial window size in pixels (resizable).
        font_size: Base font size for panel text.
    """

    def __init__(
        self,
        title: str = "Solarian",
        width: int = 1280,
        height: int = 760,
        font_size: int = 16,
    ):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        self._pygame = pygame
        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.font = pygame.font.SysFont("menlo,consolas,monospace", font_size)
        self.title_font = pygame.font.SysFont("menlo,consolas,monospace", font_size + 2, bold=True)
        self.clock = pygame.time.Clock()
        self._line_h = self.font.get_linesize()
        self._held: set[str] = set()
        self._surface_cache: dict[tuple, object] = {}
        #: ``(label, screen rect, (width, height) as captured)`` of every image
        #: drawn by the last :meth:`render`, so a click maps back to pixels.
        self._image_rects: list[tuple[str, tuple[int, int, int, int], tuple[int, int]]] = []

    # -- event handling ----------------------------------------------------
    def poll(
        self,
        on_press: Optional[Callable[[str], None]] = None,
        on_release: Optional[Callable[[str], None]] = None,
        on_click: Optional[Callable[[str, tuple[int, int], int], None]] = None,
    ) -> bool:
        """Drain the event queue.

        Routes key-down / key-up events to *on_press* / *on_release* and tracks
        the held-key set. A mouse button pressed over a camera image calls
        *on_click* with the image's label, the point in image pixels and the
        button. Returns ``True`` if the user asked to quit (window closed or
        ``Esc``).
        """
        pygame = self._pygame
        quit_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if on_click is None:
                    continue
                for label, rect, image_size in self._image_rects:
                    point = display_to_image(event.pos, rect, image_size)
                    if point is not None:
                        on_click(label, point, event.button)
                        break
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE,):
                    quit_requested = True
                char = self._event_char(event.key)
                if char is None:
                    continue
                if char not in self._held:
                    self._held.add(char)
                    if on_press is not None:
                        on_press(char)
            elif event.type == pygame.KEYUP:
                char = self._event_char(event.key)
                if char is None:
                    continue
                if char in self._held:
                    self._held.discard(char)
                    if on_release is not None:
                        on_release(char)
        return quit_requested

    def _event_char(self, key: int) -> Optional[str]:
        """Return the lowercase name for a pygame key code, or None for the
        keys nothing binds (arrows, modifiers, function keys, ...)."""
        name = self._pygame.key.name(key).lower()
        if len(name) == 1 or name in _NAMED_KEYS:
            return name
        return None

    @property
    def held(self) -> set[str]:
        return set(self._held)

    # -- rendering ---------------------------------------------------------
    def render(
        self,
        *,
        sections: Sequence[tuple[str, Sequence[tuple[str, str]]]],
        images: Sequence[tuple[str, np.ndarray, Sequence, Sequence]],
        key_help: Sequence[tuple[str, str]],
        status: str = "",
        fps: Optional[float] = None,
    ) -> None:
        """Draw one frame.

        Args:
            sections: ``(title, [(field, value), ...])`` dashboard panels.
            images: ``(label, rgb_array, bboxes, masks, bbox_texts)`` camera
                frames (HxWx3, uint8, RGB), each with its pixel
                ``(x0, y0, x1, y1)`` boxes, its HxW 0/1 masks, and per box the
                text written above it or None.
            key_help: ``(keys, description)`` rows for the bindings panel;
                individual keys (split on ``/`` and spaces) that are currently
                held are highlighted.
            status: One-line status string shown at the bottom.
            fps: Optional measured control loop rate to display.
        """
        pygame = self._pygame
        w, h = self.screen.get_size()
        self.screen.fill(_BG)

        pad = 10
        left_w = max(320, int(w * 0.34))

        # Left column: dashboard sections stacked vertically.
        y = pad
        for title, rows in sections:
            y = self._draw_panel(pad, y, left_w - 2 * pad, title, rows) + pad

        # Keys panel below the dashboard.
        key_rows = [(keys, desc) for keys, desc in key_help]
        self._draw_keys_panel(pad, y, left_w - 2 * pad, key_rows)

        # Right area: camera grid.
        cam_x = left_w
        self._draw_camera_grid(cam_x, pad, w - cam_x - pad, h - 2 * pad - self._line_h, images)

        # Bottom status bar.
        parts = []
        if fps is not None:
            parts.append(f"{fps:4.1f} Hz")
        if status:
            parts.append(status)
        parts.append("focus this window to control · Esc to quit")
        self._blit_text(" · ".join(parts), pad, h - self._line_h, _DIM)

        pygame.display.flip()

    def tick(self, fps: int) -> None:
        self.clock.tick(fps)

    # -- panel primitives --------------------------------------------------
    def _draw_panel(self, x, y, width, title, rows) -> int:
        pygame = self._pygame
        n = max(1, len(rows))
        # Two-column layout inside the panel when there are many rows.
        two_col = len(rows) > 8
        col_rows = (n + 1) // 2 if two_col else n
        height = self._line_h * (col_rows + 1) + 14

        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, _PANEL_BG, rect, border_radius=6)
        pygame.draw.rect(self.screen, _PANEL_BORDER, rect, width=1, border_radius=6)

        self._blit_text(title, x + 10, y + 6, _TITLE, font=self.title_font)

        ty0 = y + 8 + self._line_h
        if two_col:
            half = col_rows
            col_w = (width - 20) // 2
            for i, row in enumerate(rows):
                field, value, *rest = row
                color = rest[0] if rest else None
                col = 0 if i < half else 1
                row_i = i if i < half else i - half
                tx = x + 10 + col * col_w
                ty = ty0 + row_i * self._line_h
                self._blit_kv(field, value, tx, ty, col_w, color)
        else:
            for i, row in enumerate(rows):
                field, value, *rest = row
                color = rest[0] if rest else None
                self._blit_kv(field, value, x + 10, ty0 + i * self._line_h, width - 20, color)
        return y + height

    def _blit_kv(self, field, value, x, y, width, color=None):
        self._blit_text(str(field), x, y, _DIM)
        val = str(value)
        val_surf = self.font.render(val, True, color or _TEXT)
        self.screen.blit(val_surf, (x + width - val_surf.get_width() - 6, y))

    def _draw_keys_panel(self, x, y, width, rows) -> int:
        pygame = self._pygame
        height = self._line_h * (len(rows) + 1) + 14
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, _PANEL_BG, rect, border_radius=6)
        pygame.draw.rect(self.screen, _PANEL_BORDER, rect, width=1, border_radius=6)
        self._blit_text("Keys", x + 10, y + 6, _TITLE, font=self.title_font)

        ty = y + 8 + self._line_h
        for keys, desc in rows:
            tx = x + 10
            for token in keys.split():
                held = token.lower() in self._held
                color = _KEY_HELD if held else _KEY_IDLE
                surf = self.font.render(token, True, color)
                kw = surf.get_width()
                if held:
                    pygame.draw.rect(
                        self.screen, _KEY_HELD_BG,
                        pygame.Rect(tx - 3, ty - 1, kw + 6, self._line_h - 1),
                        border_radius=4,
                    )
                self.screen.blit(surf, (tx, ty))
                tx += kw + 10
            self._blit_text(desc, x + 10 + int(width * 0.42), ty, _TEXT)
            ty += self._line_h
        return y + height

    def _draw_camera_grid(self, x, y, width, height, images):
        pygame = self._pygame
        if width <= 0 or height <= 0:
            return
        if not images:
            self._blit_text("(no cameras)", x + 8, y + 8, _DIM)
            return
        n = len(images)
        cols = 1 if n == 1 else 2
        rows = (n + cols - 1) // cols
        cell_w = width // cols
        cell_h = height // rows
        self._image_rects = []
        for i, (label, img, bboxes, masks, bbox_texts) in enumerate(images):
            c = i % cols
            r = i // cols
            cx = x + c * cell_w
            cy = y + r * cell_h
            drawn = self._blit_image(
                img, bboxes, masks, cx, cy + self._line_h, cell_w - 6, cell_h - self._line_h - 6,
                bbox_texts,
            )
            if drawn is not None:
                self._image_rects.append((label, *drawn))
            self._blit_text(label, cx + 2, cy, _TITLE)

    def _blit_image(self, img: np.ndarray, bboxes, masks, x, y, max_w, max_h, bbox_texts=()):
        """Draw *img* scaled into the box; returns ``(rect, (iw, ih))`` or None."""
        pygame = self._pygame
        if img is None or max_w <= 0 or max_h <= 0:
            return None
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        # Tinted at full resolution so the fill keeps the mask's own edge.
        arr = tint_masks(arr, masks)
        ih, iw = arr.shape[:2]
        scale = min(max_w / iw, max_h / ih)
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        # pygame surfaces are (width, height); frombuffer expects row-major RGB.
        surf = pygame.image.frombuffer(np.ascontiguousarray(arr).tobytes(), (iw, ih), "RGB")
        if (dw, dh) != (iw, ih):
            surf = pygame.transform.smoothscale(surf, (dw, dh))
        self.screen.blit(surf, (x, y))
        # Rings are drawn after scaling so the edge stays 2 px on screen.
        for mask in masks:
            for ring in mask_rings(mask):
                points = [(x + int(px * dw / iw), y + int(py * dh / ih)) for px, py in ring]
                pygame.draw.lines(self.screen, MASK_COLOR, True, points, 2)
        for bbox, text in zip_longest(bboxes, list(bbox_texts)[:len(bboxes)]):
            x0, y0, x1, y1 = (int(v) for v in bbox[:4])
            if x0 == y0 == x1 == y1 == 0:
                continue
            left, top = x + int(x0 * dw / iw), y + int(y0 * dh / ih)
            pygame.draw.rect(
                self.screen,
                _BBOX,
                pygame.Rect(
                    left,
                    top,
                    max(1, int((x1 - x0) * dw / iw)),
                    max(1, int((y1 - y0) * dh / ih)),
                ),
                width=2,
            )
            if text:
                surf = self.font.render(text, True, _BBOX, _BG)
                self.screen.blit(surf, (left, max(y, top - surf.get_height() - 2)))
        return (x, y, dw, dh), (iw, ih)

    def _blit_text(self, text, x, y, color, font=None):
        font = font or self.font
        self.screen.blit(font.render(text, True, color), (x, y))

    def close(self):
        try:
            self._pygame.display.quit()
            self._pygame.quit()
        except Exception:
            logger.debug("pygame shutdown raised", exc_info=True)
