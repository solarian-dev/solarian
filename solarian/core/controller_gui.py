"""Single-window controller GUI (pygame).

A :class:`ControllerGUI` is the non-headless front-end for a
:class:`~solarian.core.controller.Controller`. It replaces the split
Rich-console dashboard + one-OpenCV-window-per-camera setup with a single
focused window that shows, side by side:

* a **dashboard** of the latest observation / action fields (text panels),
* a **camera grid** of every image field in the observation, and
* a **keys panel** listing the key bindings and highlighting the keys that are
  currently held.

The readout and the keys share one column. When that column is taller than the
window it scrolls, and a status bar keeps the last row from sliding under the
frame rate. Words are a system sans; numbers and keycaps are monospace.

The window is created with SDL's high-DPI flag. ``pygame.display.set_mode``
never sets it, so on a retina display the framebuffer would be 1x and macOS
would scale every glyph. Mouse positions arrive in points; the framebuffer is
in pixels. :meth:`_to_pixels` is the conversion.

Crucially, key capture happens through this window's own event queue (pygame),
so it needs **no OS-level accessibility / input-monitoring permission** the way
the global ``pynput`` listener does -- the window just has to be focused. pygame
also reports real key-down / key-up events (and multiple simultaneously held
keys), which the timeout-based OpenCV ``waitKey`` path cannot.

``pygame`` is imported lazily inside :meth:`__init__` so importing this module
(or :mod:`solarian`) never pulls in SDL.
"""

from __future__ import annotations

import ctypes
import ctypes.util
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
_TITLE = (148, 176, 196)
_TEXT = (230, 230, 236)
_DIM = (150, 150, 164)
_KEY_BG = (46, 46, 58)
_KEY_HELD = (90, 220, 130)
_KEY_HELD_BG = (35, 70, 45)
_WARN = (240, 200, 80)
_HIGH = (240, 80, 80)
_STATUS_BG = (24, 24, 30)

# BGRA masks of the SDL window surface on little-endian. The pygame surface we
# draw into uses the same masks so a frame can be copied straight across.
_BGRA_MASKS = (0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)

_SDL_WINDOWPOS_UNDEFINED = 0x1FFF0000
_SDL_WINDOW_SHOWN = 0x00000004
_SDL_WINDOW_RESIZABLE = 0x00000020
_SDL_WINDOW_ALLOW_HIGHDPI = 0x00002000

_UI_FONTS = "helveticaneue,segoeui,arial,freesans,dejavusans"
_MONO_FONTS = "menlo,consolas,dejavusansmono,monospace"

#: Keys pygame reports by name rather than as a character, which controllers
#: nonetheless bind. Without these, ``space`` and ``delete`` never reach a
#: handler. Every other multi-character name (arrows, modifiers, function
#: keys) is still dropped.
_NAMED_KEYS = frozenset({"space", "delete", "backspace"})
LEVEL_COLORS = {"mid": _WARN, "high": _HIGH}
_BBOX = (90, 220, 130)


class _SDLSurface(ctypes.Structure):
    """The prefix of SDL2's ``SDL_Surface``.

    ctypes inserts the padding a 64-bit build needs. Only these fields are
    read; the real struct continues past ``pixels``.
    """

    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("format", ctypes.c_void_p),
        ("w", ctypes.c_int),
        ("h", ctypes.c_int),
        ("pitch", ctypes.c_int),
        ("pixels", ctypes.c_void_p),
    ]


def clip_text(measure: Callable[[str], tuple[int, int]], text: str, max_width: int) -> str:
    """Return *text* fitting in *max_width*, ending in an ellipsis when cut.

    *measure* is ``font.size``.
    """
    if max_width <= 0 or not text:
        return ""
    if measure(text)[0] <= max_width:
        return text
    ell = "…"
    if measure(ell)[0] > max_width:
        return ""
    lo, hi = 0, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + ell
        if measure(candidate)[0] <= max_width:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def split_temperature_rows(rows: Sequence[tuple]) -> tuple[list, list]:
    """Split a section into positions and ``*_temp`` fields, preserving order.

    Temperatures are a different reading from the joint value they follow, so
    the panel can draw them dimmer instead of in the same list.
    """
    main: list = []
    temps: list = []
    for row in rows:
        if str(row[0]).endswith("_temp"):
            temps.append(row)
        else:
            main.append(row)
    return main, temps


def display_to_image(
    pos: tuple[int, int],
    rect: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int] | None:
    """Map a window position to pixels of the image drawn at *rect*.

    *rect* is ``(x, y, width, height)`` of the scaled image on screen and
    *image_size* its ``(width, height)`` as captured. ``None`` when *pos* is
    outside the image. *pos* and *rect* are both in framebuffer pixels.
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


def _load_sdl() -> ctypes.CDLL:
    """The SDL2 pygame itself is linked against, so the event queue is shared."""
    root = os.path.dirname(__import__("pygame").__file__)
    for folder, prefix in ((".dylibs", "libSDL2-2.0"), (".libs", "libSDL2-2.0")):
        path = os.path.join(root, folder)
        if not os.path.isdir(path):
            continue
        names = sorted(name for name in os.listdir(path) if name.startswith(prefix))
        if names:
            return ctypes.CDLL(os.path.join(path, names[-1]))
    found = ctypes.util.find_library("SDL2")
    if not found:
        raise OSError("SDL2 library not found")
    return ctypes.CDLL(found)


def _bind_sdl(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.SDL_CreateWindow.restype = ctypes.c_void_p
    lib.SDL_CreateWindow.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint32,
    ]
    lib.SDL_DestroyWindow.argtypes = [ctypes.c_void_p]
    lib.SDL_GetWindowSurface.restype = ctypes.c_void_p
    lib.SDL_GetWindowSurface.argtypes = [ctypes.c_void_p]
    lib.SDL_UpdateWindowSurface.restype = ctypes.c_int
    lib.SDL_UpdateWindowSurface.argtypes = [ctypes.c_void_p]
    lib.SDL_GetWindowSize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.SDL_GetWindowSizeInPixels.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.SDL_GetError.restype = ctypes.c_char_p
    return lib


def _sdl_size(fn, window) -> tuple[int, int]:
    width, height = ctypes.c_int(), ctypes.c_int()
    fn(window, ctypes.byref(width), ctypes.byref(height))
    return width.value, height.value


class ControllerGUI:
    """pygame window rendering the controller dashboard, cameras and keys.

    Args:
        title: Window title.
        width / height: Initial window size in points (resizable). On a
            high-DPI display the framebuffer is larger, and text is drawn
            at that scale.
        font_size: Body size in points. Keycaps and numbers use the same
            size in a monospace face.
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
        self._font_size = font_size
        self._font_px = 0
        self._scale = 1.0
        self._pad = 10
        self._inner = 8
        self._radius = 6
        self._line_h = font_size + 4
        self._scroll = 0.0
        self._max_scroll = 0
        self._left_w = 0
        self._held: set[str] = set()
        self._win = None
        self._sdl = None
        self.screen = None
        self.font = None
        self.mono = None
        self.title_font = None
        #: ``(label, screen rect, (width, height) as captured)`` of every image
        #: drawn by the last :meth:`render`, so a click maps back to pixels.
        self._image_rects: list[tuple[str, tuple[int, int, int, int], tuple[int, int]]] = []
        self._open(title, width, height)
        self.clock = pygame.time.Clock()

    def _open(self, title: str, width: int, height: int) -> None:
        pygame = self._pygame
        try:
            self._sdl = _bind_sdl(_load_sdl())
            win = self._sdl.SDL_CreateWindow(
                title.encode("utf-8"),
                _SDL_WINDOWPOS_UNDEFINED,
                _SDL_WINDOWPOS_UNDEFINED,
                int(width),
                int(height),
                _SDL_WINDOW_SHOWN | _SDL_WINDOW_RESIZABLE | _SDL_WINDOW_ALLOW_HIGHDPI,
            )
            if not win:
                err = self._sdl.SDL_GetError()
                raise OSError(err.decode() if err else "SDL_CreateWindow failed")
            self._win = win
            self._sync_backing()
        except Exception as exc:
            logger.info("high-DPI window unavailable (%s); using a 1x surface", exc)
            if self._win and self._sdl:
                self._sdl.SDL_DestroyWindow(self._win)
            self._win = None
            self._sdl = None
            pygame.display.set_caption(title)
            self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            self._scale = 1.0
            self._build_fonts()

    def _sync_backing(self) -> None:
        """Match ``self.screen`` to the window.

        The high-DPI window's pixel size is not its point size. The fallback
        path lets pygame own the surface, which already tracks resizes.
        """
        pygame = self._pygame
        if not self._win:
            self.screen = pygame.display.get_surface()
            return
        ww, wh = _sdl_size(self._sdl.SDL_GetWindowSize, self._win)
        pw, ph = _sdl_size(self._sdl.SDL_GetWindowSizeInPixels, self._win)
        if min(ww, wh, pw, ph) <= 0:
            return
        self._scale = pw / ww
        if self.screen is None or self.screen.get_size() != (pw, ph):
            self.screen = pygame.Surface((pw, ph), depth=32, masks=_BGRA_MASKS)
        px = max(1, round(self._font_size * self._scale))
        if self.font is None or px != self._font_px:
            self._font_px = px
            self._build_fonts()

    def _build_fonts(self) -> None:
        pygame = self._pygame
        px = self._font_px or max(1, round(self._font_size * self._scale))
        self._font_px = px
        title_px = max(1, round((self._font_size - 2) * self._scale))
        self.font = pygame.font.SysFont(_UI_FONTS, px)
        self.title_font = pygame.font.SysFont(_UI_FONTS, title_px)
        self.mono = pygame.font.SysFont(_MONO_FONTS, px)
        self._line_h = max(self.font.get_linesize(), self.mono.get_linesize())
        self._pad = max(8, round(10 * self._scale))
        self._inner = max(6, round(8 * self._scale))
        self._radius = max(4, round(6 * self._scale))

    def _to_pixels(self, pos: tuple[int, int]) -> tuple[int, int]:
        """Window points to framebuffer pixels.

        SDL delivers mouse positions in the window's point size. A high-DPI
        framebuffer is an integer multiple of that, so a click has to be
        scaled before it can hit an image rect.
        """
        scale = self._scale
        return (int(pos[0] * scale), int(pos[1] * scale))

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
        button. The wheel scrolls the readout when the pointer is over it.
        Returns ``True`` if the user asked to quit (window closed or ``Esc``).
        """
        pygame = self._pygame
        quit_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            elif event.type == pygame.MOUSEWHEEL:
                if self._to_pixels(pygame.mouse.get_pos())[0] < self._left_w:
                    self._scroll = min(
                        self._max_scroll,
                        max(0.0, self._scroll - event.precise_y * self._line_h),
                    )
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if on_click is None:
                    continue
                pos = self._to_pixels(event.pos)
                for label, rect, image_size in self._image_rects:
                    point = display_to_image(pos, rect, image_size)
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
            status: One-line status string shown in the bottom bar.
            fps: Optional measured control loop rate to display.
        """
        pygame = self._pygame
        self._sync_backing()
        if self.screen is None or self.font is None:
            return
        w, h = self.screen.get_size()
        if w < 8 or h < 8:
            return
        # The framebuffer has an alpha channel. A 3-tuple fill leaves it at 0,
        # and the window composites as transparent.
        self.screen.fill((*_BG, 255))

        status_h = self._line_h + self._inner
        left_w = min(
            max(round(300 * self._scale), int(w * 0.34)),
            max(0, w - round(120 * self._scale)),
        )
        self._left_w = left_w
        viewport_h = max(0, h - status_h)
        content_h = self._column_height(sections, key_help)
        self._max_scroll = max(0, content_h - viewport_h)
        self._scroll = min(self._max_scroll, max(0.0, self._scroll))

        self.screen.set_clip(pygame.Rect(0, 0, left_w, viewport_h))
        try:
            self._draw_column(sections, key_help, left_w)
        finally:
            self.screen.set_clip(None)

        if left_w > 0:
            pygame.draw.line(self.screen, _PANEL_BORDER, (left_w - 1, 0), (left_w - 1, viewport_h))

        self._draw_camera_grid(
            left_w,
            self._pad,
            w - left_w - self._pad,
            viewport_h - 2 * self._pad,
            images,
        )
        self._draw_status(w, h, status_h, fps, status)
        self._present()

    def _column_height(self, sections, key_rows) -> int:
        y = self._pad
        for _title, rows in sections:
            y += self._panel_height(len(self._panel_lines(rows))) + self._pad
        if key_rows:
            y += self._panel_height(len(key_rows)) + self._pad
        return y

    def _panel_height(self, body_lines: int) -> int:
        return self._inner * 2 + self._line_h * (1 + body_lines)

    def _panel_lines(self, rows) -> list[tuple]:
        main, temps = split_temperature_rows(rows)
        lines: list[tuple] = [("kv", row, False) for row in main]
        if temps and main:
            lines.append(("label", "temperature", True))
        lines.extend(("kv", row, True) for row in temps)
        return lines

    def _draw_column(self, sections, key_rows, left_w: int) -> None:
        width = left_w - 2 * self._pad
        if width <= 0:
            return
        y = self._pad - int(self._scroll)
        for title, rows in sections:
            y = self._draw_panel(self._pad, y, width, title, rows) + self._pad
        if key_rows:
            self._draw_keys_panel(self._pad, y, width, key_rows)

    def tick(self, fps: int) -> None:
        self.clock.tick(fps)

    def _present(self) -> None:
        if not self._win:
            self._pygame.display.flip()
            return
        ptr = self._sdl.SDL_GetWindowSurface(self._win)
        if not ptr:
            return
        target = ctypes.cast(ptr, ctypes.POINTER(_SDLSurface)).contents
        src = self.screen
        if target.pixels is None or (target.w, target.h) != src.get_size():
            return
        raw = src.get_buffer().raw
        src_pitch = src.get_pitch()
        if target.pitch == src_pitch:
            ctypes.memmove(target.pixels, raw, target.pitch * target.h)
        else:
            row_bytes = min(src_pitch, target.pitch)
            for y in range(target.h):
                ctypes.memmove(
                    target.pixels + y * target.pitch,
                    raw[y * src_pitch : y * src_pitch + row_bytes],
                    row_bytes,
                )
        self._sdl.SDL_UpdateWindowSurface(self._win)

    # -- panel primitives --------------------------------------------------
    def _draw_panel(self, x, y, width, title, rows) -> int:
        pygame = self._pygame
        lines = self._panel_lines(rows)
        height = self._panel_height(len(lines))
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, _PANEL_BG, rect, border_radius=self._radius)
        pygame.draw.rect(
            self.screen, _PANEL_BORDER, rect, width=1, border_radius=self._radius
        )
        inset = self._inner
        inner_w = width - 2 * inset
        self._blit(str(title), x + inset, y + inset, _TITLE, self.title_font, _PANEL_BG, inner_w)
        ty = y + inset + self._line_h
        for kind, payload, dim in lines:
            if kind == "label":
                self._blit(payload, x + inset, ty, _DIM, self.title_font, _PANEL_BG, inner_w)
            else:
                field, value, *rest = payload
                color = rest[0] if rest else None
                self._draw_kv(field, value, x + inset, ty, inner_w, color, dim)
            ty += self._line_h
        return y + height

    def _draw_kv(self, field, value, x, y, width, color, dim: bool) -> None:
        if width <= 0:
            return
        value_s = str(value)
        value_w = self.mono.size(value_s)[0]
        gap = max(8, self._pad // 2)
        val_color = color or (_DIM if dim else _TEXT)
        if value_w >= width:
            self._blit(value_s, x, y, val_color, self.mono, _PANEL_BG, width)
            return
        self._blit(str(field), x, y, _DIM, self.font, _PANEL_BG, width - value_w - gap)
        self._blit(value_s, x + width - value_w, y, val_color, self.mono, _PANEL_BG)

    def _draw_keys_panel(self, x, y, width, rows) -> int:
        pygame = self._pygame
        height = self._panel_height(len(rows))
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, _PANEL_BG, rect, border_radius=self._radius)
        pygame.draw.rect(
            self.screen, _PANEL_BORDER, rect, width=1, border_radius=self._radius
        )
        inset = self._inner
        inner_w = width - 2 * inset
        self._blit("Keys", x + inset, y + inset, _TITLE, self.title_font, _PANEL_BG, inner_w)

        # Descriptions share a column that starts just after the widest key
        # group, so "0" and "delete" don't each invent their own indent.
        gap = max(4, self._inner // 2)
        widest = 0
        for keys, _desc in rows:
            tokens = str(keys).split()
            if not tokens:
                continue
            group = sum(self.mono.size(token)[0] + self._inner for token in tokens)
            group += gap * (len(tokens) - 1)
            widest = max(widest, group)
        gutter = min(widest + gap, int(inner_w * 0.5))
        ty = y + inset + self._line_h
        for keys, desc in rows:
            tx = x + inset
            limit = x + inset + gutter
            for token in str(keys).split():
                held = token.lower() in self._held
                label_w = self.mono.size(token)[0]
                cap_w = label_w + self._inner
                if tx + cap_w > limit and tx > x + inset:
                    break
                bg = _KEY_HELD_BG if held else _KEY_BG
                fg = _KEY_HELD if held else _TEXT
                cap = pygame.Rect(tx, ty + 1, cap_w, self._line_h - 2)
                pygame.draw.rect(self.screen, bg, cap, border_radius=max(3, self._radius // 2))
                text = self.mono.render(token, True, fg, bg)
                self.screen.blit(
                    text,
                    (tx + (cap_w - text.get_width()) // 2, ty + (self._line_h - text.get_height()) // 2),
                )
                tx += cap_w + max(4, self._inner // 2)
            desc_x = max(tx + max(4, self._inner // 2), x + inset + gutter)
            self._blit(str(desc), desc_x, ty, _TEXT, self.font, _PANEL_BG, x + width - inset - desc_x)
            ty += self._line_h
        return y + height

    def _draw_status(self, w, h, status_h, fps, status: str) -> None:
        pygame = self._pygame
        y = h - status_h
        pygame.draw.rect(self.screen, _STATUS_BG, pygame.Rect(0, y, w, status_h))
        pygame.draw.line(self.screen, _PANEL_BORDER, (0, y), (w, y))
        ty = y + (status_h - self._line_h) // 2

        hint = "focus this window to control · Esc to quit"
        hint_s = clip_text(self.font.size, hint, max(0, w // 3))
        hint_w = self.font.size(hint_s)[0] if hint_s else 0
        budget = w - self._pad * 2 - (hint_w + self._pad if hint_w else 0)

        parts: list[tuple[str, object, tuple]] = []
        if fps is not None:
            parts.append((f"{fps:.1f} Hz", self.mono, _TEXT))
        for piece in status.split(" · "):
            if piece:
                color = _HIGH if piece.startswith("REC") else _DIM
                parts.append((piece, self.font, color))

        x = self._pad
        for i, (piece, font, color) in enumerate(parts):
            if i:
                if x + self._pad > budget:
                    break
                x += self._pad
            remaining = budget - x
            shown = clip_text(font.size, piece, remaining)
            if not shown:
                break
            drawn = self._blit(shown, x, ty, color, font, _STATUS_BG)
            x += drawn
            if shown != piece:
                break

        if hint_s and x + self._pad + hint_w <= w - self._pad:
            self._blit(hint_s, w - self._pad - hint_w, ty, _DIM, self.font, _STATUS_BG)

    def _draw_camera_grid(self, x, y, width, height, images):
        self._image_rects = []
        if width <= 0 or height <= 0:
            return
        if not images:
            self._blit("(no cameras)", x + 8, y + 8, _DIM, self.font, _BG)
            return
        n = len(images)
        cols = 1 if n == 1 else 2
        rows = (n + cols - 1) // cols
        cell_w = width // cols
        cell_h = height // rows
        gap = max(4, self._pad // 2)
        for i, (label, img, bboxes, masks, bbox_texts) in enumerate(images):
            c = i % cols
            r = i // cols
            cx = x + c * cell_w
            cy = y + r * cell_h
            drawn = self._blit_image(
                img, bboxes, masks, cx, cy, cell_w - gap, cell_h - gap, bbox_texts,
            )
            if drawn is not None:
                rect, image_size = drawn
                self._image_rects.append((label, rect, image_size))
                self._draw_chip(str(label), rect)

    def _draw_chip(self, label: str, rect: tuple[int, int, int, int]) -> None:
        pygame = self._pygame
        x, y, rw, rh = rect
        inset = max(4, self._inner // 2)
        if rw < inset * 4 or rh < self._line_h:
            return
        max_w = rw - 2 * inset
        text = clip_text(self.title_font.size, label, max_w - self._inner)
        if not text:
            return
        surf = self.title_font.render(text, True, _TEXT, _PANEL_BG)
        pad_x = max(4, self._inner // 2)
        pad_y = max(2, self._inner // 4)
        box = pygame.Rect(x + inset, y + inset, surf.get_width() + 2 * pad_x, surf.get_height() + 2 * pad_y)
        pygame.draw.rect(self.screen, _PANEL_BG, box, border_radius=max(3, self._radius // 2))
        pygame.draw.rect(
            self.screen, _PANEL_BORDER, box, width=1, border_radius=max(3, self._radius // 2)
        )
        self.screen.blit(surf, (box.x + pad_x, box.y + pad_y))

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
                surf = self.font.render(str(text), True, _BBOX, _BG)
                self.screen.blit(surf, (left, max(y, top - surf.get_height() - 2)))
        return (x, y, dw, dh), (iw, ih)

    def _blit(self, text, x, y, color, font, bg, max_width=None) -> int:
        if max_width is not None:
            text = clip_text(font.size, str(text), max_width)
        else:
            text = str(text)
        if not text:
            return 0
        surf = font.render(text, True, color, bg)
        self.screen.blit(surf, (x, y))
        return surf.get_width()

    def close(self):
        try:
            if self._win and self._sdl:
                self._sdl.SDL_DestroyWindow(self._win)
                self._win = None
            self._pygame.display.quit()
            self._pygame.quit()
        except Exception:
            logger.debug("pygame shutdown raised", exc_info=True)
