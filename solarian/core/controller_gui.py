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
_SELECT_BG = (52, 74, 120)

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
#: Key repeat while typing a chat line, close to the macOS default.
_REPEAT_DELAY_MS = 400
_REPEAT_INTERVAL_MS = 35
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


def wrap_text(measure: Callable[[str], tuple[int, int]], text: str, max_width: int) -> list[str]:
    """Break *text* into lines no wider than *max_width*, at spaces where possible.

    *measure* is ``font.size``. Explicit newlines are kept; a word wider than
    the line is cut with :func:`clip_text` rather than overflowing.
    """
    if max_width <= 0:
        return []
    lines: list[str] = []
    for paragraph in str(text).split("\n"):
        line = ""
        for word in paragraph.split(" "):
            candidate = f"{line} {word}" if line else word
            if measure(candidate)[0] <= max_width:
                line = candidate
                continue
            if line:
                lines.append(line)
            line = word if measure(word)[0] <= max_width else clip_text(measure, word, max_width)
        lines.append(line)
    return lines


def selected_text(
    lines: Sequence[tuple[str, str]], start: tuple[int, int], end: tuple[int, int]
) -> str:
    """The text from *start* to *end*, ``(line, column)`` positions in wrapped *lines*.

    Each line is ``(text, joiner)``: the joiner is what wrapping replaced
    before it, a space inside a paragraph or a newline between them, so the
    copy reads as the original rather than as the column's line breaks.
    """
    out: list[str] = []
    for i in range(start[0], min(end[0], len(lines) - 1) + 1):
        text, joiner = lines[i]
        piece = text[start[1] if i == start[0] else 0 : end[1] if i == end[0] else len(text)]
        out.append(piece if i == start[0] else joiner + piece)
    return "".join(out)


def format_tokens(n: int) -> str:
    """``850``, ``275k``, ``1.2M``."""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.0f}k"
    return f"{n / 1_000_000:.1f}M"


class LineEdit:
    """One line of editable text: a cursor, and a selection from ``anchor`` to it."""

    def __init__(self, text: str = ""):
        self.text = text
        self.cursor = len(text)
        self.anchor = self.cursor

    def selection(self) -> tuple[int, int]:
        return min(self.anchor, self.cursor), max(self.anchor, self.cursor)

    def selected(self) -> str:
        start, end = self.selection()
        return self.text[start:end]

    def insert(self, text: str) -> None:
        """Type *text* over the selection."""
        start, end = self.selection()
        self.text = self.text[:start] + text + self.text[end:]
        self.cursor = self.anchor = start + len(text)

    def move_to(self, pos: int, select: bool = False) -> None:
        self.cursor = max(0, min(len(self.text), pos))
        if not select:
            self.anchor = self.cursor

    def select_all(self) -> None:
        self.anchor, self.cursor = 0, len(self.text)

    def word_left(self) -> int:
        pos = self.cursor
        while pos > 0 and self.text[pos - 1] == " ":
            pos -= 1
        while pos > 0 and self.text[pos - 1] != " ":
            pos -= 1
        return pos

    def word_right(self) -> int:
        pos = self.cursor
        while pos < len(self.text) and self.text[pos] == " ":
            pos += 1
        while pos < len(self.text) and self.text[pos] != " ":
            pos += 1
        return pos

    def delete_to(self, pos: int) -> None:
        """Delete the selection, or else from the cursor to *pos*."""
        if self.anchor == self.cursor:
            self.anchor = max(0, min(len(self.text), pos))
        self.insert("")


#: Plan step status -> the mark drawn before it.
_PLAN_MARKS = {"done": "[x]", "in_progress": "[>]", "failed": "[!]", "skipped": "[-]"}
#: Transcript role -> (prefix, color).
_CHAT_ROLES = {
    "user": ("you: ", _KEY_HELD),
    "agent": ("", _TEXT),
    "tool": ("· ", _DIM),
    "system": ("", _WARN),
}


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


def _recording_lines(recording: dict) -> list[tuple[str, str, bool, bool, bool]]:
    """``(kind, side or hint, on, recording now, available)`` per panel line."""
    targets, robot, controller = recording["targets"], recording["robot"], recording["controller"]
    lines = [
        ("switch", "robot", bool(targets["robot"]), bool(robot["active"]), bool(robot["enabled"])),
        ("switch", "controller", bool(targets["controller"]), bool(controller["active"]), True),
    ]
    if recording["active"]:
        lines.append(("hint", "switches apply to the next episode", False, False, True))
    return lines


def _inside(pos: tuple[int, int], rect: tuple[int, int, int, int] | None) -> bool:
    if rect is None:
        return False
    x, y, w, h = rect
    return x <= pos[0] < x + w and y <= pos[1] < y + h


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


class _Clipboard:
    """The system clipboard through SDL.

    ``pygame.scrap`` needs a window ``pygame.display`` opened, and the
    high-DPI window is SDL's own. Without SDL, copy and paste do nothing.
    """

    def __init__(self):
        try:
            lib = _load_sdl()
            lib.SDL_SetClipboardText.argtypes = [ctypes.c_char_p]
            lib.SDL_SetClipboardText.restype = ctypes.c_int
            lib.SDL_GetClipboardText.restype = ctypes.c_void_p
            lib.SDL_free.argtypes = [ctypes.c_void_p]
            self._sdl: ctypes.CDLL | None = lib
        except (OSError, AttributeError) as exc:
            logger.info("clipboard unavailable (%s)", exc)
            self._sdl = None

    def get(self) -> str:
        if self._sdl is None:
            return ""
        ptr = self._sdl.SDL_GetClipboardText()
        if not ptr:
            return ""
        try:
            return ctypes.string_at(ptr).decode("utf-8", "replace")
        finally:
            self._sdl.SDL_free(ptr)

    def set(self, text: str) -> None:
        if self._sdl is not None and text:
            self._sdl.SDL_SetClipboardText(text.encode("utf-8"))


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
        self.small_mono = None
        self.title_font = None
        #: ``(label, screen rect, (width, height) as captured)`` of every image
        #: drawn by the last :meth:`render`, so a click maps back to pixels.
        self._image_rects: list[tuple[str, tuple[int, int, int, int], tuple[int, int]]] = []
        #: ``(side, screen rect)`` of each recording switch drawn last frame.
        self._toggle_rects: list[tuple[str, tuple[int, int, int, int]]] = []
        #: The chat message being typed, or ``None`` when keys drive the robot.
        self._input: LineEdit | None = None
        #: First character of the input drawn, so the cursor stays in view;
        #: and ``(x, that index)`` of the last frame, to map clicks to columns.
        self._input_offset = 0
        self._input_view = (0, 0)
        #: Every wrapped transcript line as ``(text, joiner)`` (see
        #: :func:`selected_text`), and ``(line index, x, y)`` of those drawn.
        self._chat_lines: list[tuple[str, str]] = []
        self._chat_drawn: list[tuple[int, int, int]] = []
        #: Screen rects ``(x, y, w, h)`` of the transcript and the input box.
        self._chat_rect: tuple[int, int, int, int] | None = None
        self._input_rect: tuple[int, int, int, int] | None = None
        #: Transcript lines scrolled back from the newest.
        self._chat_scroll = 0.0
        #: ``(anchor, focus)`` as ``(line, column)`` in the transcript.
        self._selection: tuple[tuple[int, int], tuple[int, int]] | None = None
        #: What a mouse drag is selecting: ``"chat"``, ``"input"`` or ``None``.
        self._drag: str | None = None
        self._clipboard = _Clipboard()
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
        self.small_mono = pygame.font.SysFont(_MONO_FONTS, max(1, round((self._font_size - 3) * self._scale)))
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
        on_submit: Optional[Callable[[str], None]] = None,
        on_toggle: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Drain the event queue.

        Routes key-down / key-up events to *on_press* / *on_release* and tracks
        the held-key set. A mouse button pressed over a camera image calls
        *on_click* with the image's label, the point in image pixels and the
        button; a left click on a recording switch calls *on_toggle* with its
        side. The wheel scrolls the readout when the pointer is over it.
        Returns ``True`` if the user asked to quit (window closed or ``Esc``).

        With *on_submit*, Enter (or a click on the input box) opens a chat
        line: from then until Enter sends it (to *on_submit*) or Esc drops it,
        keys type text and reach neither callback, so typing "w" never drives
        the arm. Dragging over the transcript selects text; Cmd/Ctrl+C copies
        it, and Cmd/Ctrl+V pastes into the chat line.
        """
        pygame = self._pygame
        shortcut = pygame.KMOD_CTRL | pygame.KMOD_META
        quit_requested = False
        for event in pygame.event.get():
            if self._input is not None and self._type(event, on_submit):
                continue
            if event.type == pygame.QUIT:
                quit_requested = True
            elif (
                event.type == pygame.KEYDOWN
                and event.mod & shortcut
                and event.key in (pygame.K_c, pygame.K_v)
            ):
                if event.key == pygame.K_c:
                    self._clipboard.set(self._selected_chat())
                elif on_submit is not None:
                    self._open_input(on_release, " ".join(self._clipboard.get().splitlines()))
            elif (
                event.type == pygame.KEYDOWN
                and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER)
                and on_submit is not None
            ):
                self._open_input(on_release)
            elif event.type == pygame.MOUSEWHEEL:
                x = self._to_pixels(pygame.mouse.get_pos())[0]
                if x < self._left_w:
                    self._scroll = min(
                        self._max_scroll,
                        max(0.0, self._scroll - event.precise_y * self._line_h),
                    )
                elif self._chat_rect is not None and x >= self._chat_rect[0]:
                    self._chat_scroll = max(0.0, self._chat_scroll + event.precise_y)
            elif event.type == pygame.MOUSEMOTION:
                pos = self._to_pixels(event.pos)
                if self._drag == "chat" and self._selection is not None:
                    focus = self._chat_hit(pos)
                    if focus is not None:
                        self._selection = (self._selection[0], focus)
                elif self._drag == "input" and self._input is not None:
                    self._input.move_to(self._input_hit(pos), select=True)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._drag = None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = self._to_pixels(event.pos)
                if event.button == 1 and _inside(pos, self._input_rect):
                    if on_submit is None:
                        continue
                    if self._input is None:
                        self._open_input(on_release)
                    self._selection = None
                    extend = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                    self._input.move_to(self._input_hit(pos), select=extend)
                    self._drag = "input"
                    continue
                if event.button == 1 and _inside(pos, self._chat_rect):
                    hit = self._chat_hit(pos)
                    self._selection = (hit, hit) if hit is not None else None
                    self._drag = "chat" if hit is not None else None
                    if self._input is not None:
                        self._input.move_to(self._input.cursor)
                    continue
                if event.button == 1 and on_toggle is not None:
                    side = next(
                        (s for s, rect in self._toggle_rects if _inside(pos, rect)), None
                    )
                    if side is not None:
                        on_toggle(side)
                        continue
                if on_click is None:
                    continue
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

    def _open_input(self, on_release, text: str = "") -> None:
        # Keys held now would never see their release while typing.
        for char in sorted(self._held):
            if on_release is not None:
                on_release(char)
        self._held.clear()
        self._input = LineEdit(text)
        self._input_offset = 0
        self._pygame.key.start_text_input()
        # Held keys repeat only while typing: a held teleop key is one press
        # that lasts, never a stream of them.
        self._pygame.key.set_repeat(_REPEAT_DELAY_MS, _REPEAT_INTERVAL_MS)

    def _close_input(self) -> str:
        text, self._input = self._input.text.strip(), None
        self._pygame.key.stop_text_input()
        self._pygame.key.set_repeat(0, 0)
        return text

    def _type(self, event, on_submit) -> bool:
        """Apply one event to the chat line being typed; ``True`` if it was consumed.

        The editing keys of a text field: arrows, Home/End, Shift to select,
        Option/Alt for words, Cmd/Ctrl for the whole line, and Cmd/Ctrl with
        A, C, X, V.
        """
        pygame = self._pygame
        edit = self._input
        if event.type == pygame.TEXTINPUT:
            edit.insert(event.text)
            return True
        if event.type == pygame.KEYUP:
            return True
        if event.type != pygame.KEYDOWN:
            return False
        key = event.key
        line = bool(event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META))
        word = bool(event.mod & pygame.KMOD_ALT)
        shift = bool(event.mod & pygame.KMOD_SHIFT)
        start, end = edit.selection()
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            text = self._close_input()
            if text and on_submit is not None:
                on_submit(text)
        elif key == pygame.K_ESCAPE:
            self._close_input()
        elif key == pygame.K_BACKSPACE:
            edit.delete_to(0 if line else edit.word_left() if word else edit.cursor - 1)
        elif key == pygame.K_DELETE:
            edit.delete_to(len(edit.text) if line else edit.word_right() if word else edit.cursor + 1)
        elif key in (pygame.K_LEFT, pygame.K_HOME):
            if line or key == pygame.K_HOME:
                target = 0
            elif word:
                target = edit.word_left()
            else:
                target = start if start != end and not shift else edit.cursor - 1
            edit.move_to(target, select=shift)
        elif key in (pygame.K_RIGHT, pygame.K_END):
            if line or key == pygame.K_END:
                target = len(edit.text)
            elif word:
                target = edit.word_right()
            else:
                target = end if start != end and not shift else edit.cursor + 1
            edit.move_to(target, select=shift)
        elif line and key == pygame.K_a:
            edit.select_all()
        elif line and key == pygame.K_c:
            self._clipboard.set(edit.selected() or self._selected_chat() or edit.text)
        elif line and key == pygame.K_x:
            if start == end:
                edit.select_all()
            self._clipboard.set(edit.selected())
            edit.insert("")
        elif line and key == pygame.K_v:
            edit.insert(" ".join(self._clipboard.get().splitlines()))
        return True

    def _selection_span(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        if self._selection is None or self._selection[0] == self._selection[1]:
            return None
        start, end = sorted(self._selection)
        return start, end

    def _selected_chat(self) -> str:
        span = self._selection_span()
        return selected_text(self._chat_lines, *span) if span else ""

    def _chat_hit(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        """The ``(line, column)`` of the transcript under *pos*, clamped to the drawn lines."""
        if not self._chat_drawn:
            return None
        px, py = pos
        index, x, _ = next(
            (row for row in self._chat_drawn if py < row[2] + self._line_h), self._chat_drawn[-1]
        )
        return index, self._column_at(self._chat_lines[index][0], px - x)

    def _input_hit(self, pos: tuple[int, int]) -> int:
        x, offset = self._input_view
        return offset + self._column_at(self._input.text[offset:], pos[0] - x)

    def _column_at(self, text: str, dx: float) -> int:
        """The character boundary of *text* nearest *dx* pixels from its start."""
        col, left = 0, 0
        while col < len(text):
            right = self.font.size(text[: col + 1])[0]
            if (left + right) / 2 > dx:
                break
            col, left = col + 1, right
        return col

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
        agent: Optional[dict] = None,
        recording: Optional[dict] = None,
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
            agent: The agent's UI state; draws the chat column when given.
            recording: The controller's ``recording_state``; draws the
                recording switches at the top of the readout when given.
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
            max(round(200 * self._scale), int(w * 0.23)),
            max(0, w - round(120 * self._scale)),
        )
        self._left_w = left_w
        viewport_h = max(0, h - status_h)
        content_h = self._column_height(sections, key_help, recording)
        self._max_scroll = max(0, content_h - viewport_h)
        self._scroll = min(self._max_scroll, max(0.0, self._scroll))

        self.screen.set_clip(pygame.Rect(0, 0, left_w, viewport_h))
        try:
            self._draw_column(sections, key_help, left_w, recording, viewport_h)
        finally:
            self.screen.set_clip(None)

        if left_w > 0:
            pygame.draw.line(self.screen, _PANEL_BORDER, (left_w - 1, 0), (left_w - 1, viewport_h))

        chat_w = 0
        if agent is not None:
            chat_w = min(
                max(round(320 * self._scale), int(w * 0.28)),
                max(0, w - left_w - round(160 * self._scale)),
            )
        self._draw_camera_grid(
            left_w,
            self._pad,
            w - left_w - chat_w - self._pad,
            viewport_h - 2 * self._pad,
            images,
        )
        if chat_w > 0:
            self._draw_chat(w - chat_w, chat_w, viewport_h, agent)
        self._draw_status(w, h, status_h, fps, status)
        self._present()

    def _column_height(self, sections, key_rows, recording) -> int:
        y = self._pad
        if recording is not None:
            y += self._panel_height(len(_recording_lines(recording))) + self._pad
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

    def _draw_column(self, sections, key_rows, left_w: int, recording, viewport_h: int) -> None:
        self._toggle_rects = []
        width = left_w - 2 * self._pad
        if width <= 0:
            return
        y = self._pad - int(self._scroll)
        if recording is not None:
            y = self._draw_recording_panel(self._pad, y, width, recording, viewport_h) + self._pad
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
        value_w = self.small_mono.size(value_s)[0]
        # The smaller value font sits on the row's centre, not its top.
        value_y = y + (self._line_h - self.small_mono.get_linesize()) // 2
        gap = max(8, self._pad // 2)
        val_color = color or (_DIM if dim else _TEXT)
        if value_w >= width:
            self._blit(value_s, x, value_y, val_color, self.small_mono, _PANEL_BG, width)
            return
        self._blit(str(field), x, y, _DIM, self.title_font, _PANEL_BG, width - value_w - gap)
        self._blit(value_s, x + width - value_w, value_y, val_color, self.small_mono, _PANEL_BG)

    def _draw_recording_panel(self, x, y, width, recording: dict, viewport_h: int) -> int:
        """Which sides record the next episode, as two switches.

        A switch is clickable only where it is on screen: the column scrolls,
        and a rect kept for a switch scrolled under the status bar would take
        clicks meant for nothing.
        """
        pygame = self._pygame
        lines = _recording_lines(recording)
        height = self._panel_height(len(lines))
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, _PANEL_BG, rect, border_radius=self._radius)
        pygame.draw.rect(
            self.screen, _PANEL_BORDER, rect, width=1, border_radius=self._radius
        )
        inset = self._inner
        inner_w = width - 2 * inset
        self._blit("Recording", x + inset, y + inset, _TITLE, self.title_font, _PANEL_BG, inner_w)

        knob = self._line_h - 8
        track_w = 2 * knob + 4
        ty = y + inset + self._line_h
        for kind, side, on, live, available in lines:
            if kind == "hint":
                self._blit(side, x + inset, ty, _DIM, self.title_font, _PANEL_BG, inner_w)
                ty += self._line_h
                continue
            label = side.capitalize() if available else f"{side.capitalize()} (unavailable)"
            track = pygame.Rect(x + width - inset - track_w, ty + 3, track_w, knob + 2)
            if live:
                rec_w = self.small_mono.size("REC")[0]
                self._blit("REC", track.x - rec_w - inset, ty, _HIGH, self.small_mono, _PANEL_BG)
            self._blit(label, x + inset, ty, _TEXT if available else _DIM, self.title_font, _PANEL_BG,
                       max(0, track.x - x - 2 * inset))
            pygame.draw.rect(self.screen, _KEY_HELD_BG if on else _KEY_BG, track,
                             border_radius=track.height // 2)
            cx = track.right - knob // 2 - 2 if on else track.x + knob // 2 + 2
            pygame.draw.circle(self.screen, _KEY_HELD if on else _DIM, (cx, track.centery), knob // 2)
            if 0 <= track.y and track.bottom <= viewport_h:
                self._toggle_rects.append((side, (track.x, track.y, track.w, track.h)))
            ty += self._line_h
        return y + height

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
            self._blit(str(desc), desc_x, ty, _TEXT, self.title_font, _PANEL_BG, x + width - inset - desc_x)
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

    def _draw_chat(self, x: int, width: int, height: int, agent: dict) -> None:
        """The agent column: status, plan, running actions, chat, and the input line."""
        pygame = self._pygame
        pygame.draw.line(self.screen, _PANEL_BORDER, (x, 0), (x, height))
        pad, line_h = self._pad, self._line_h
        left, inner_w = x + pad, width - 2 * pad
        if inner_w <= 0:
            return
        y = pad

        status = str(agent.get("status") or "idle").replace("_", " ")
        dry = " · dry run" if agent.get("dry_run") else ""
        self._blit(f"Agent · {status}{dry}", left, y, _TITLE, self.title_font, _BG, inner_w)
        y += line_h
        for role, usage in (agent.get("usage") or {}).items():
            # Non-breaking spaces keep each count with its unit when the line wraps.
            text = (
                f"{role}: {usage.get('model', '')} · {usage.get('calls', 0)}\u00a0turns · tokens:\u00a0"
                f"{format_tokens(int(usage.get('input_tokens', 0)))}\u00a0in\u00a0·\u00a0"
                f"{format_tokens(int(usage.get('output_tokens', 0)))}\u00a0out"
            )
            for line in wrap_text(self.title_font.size, text, inner_w):
                self._blit(line, left, y, _DIM, self.title_font, _BG, inner_w)
                y += line_h
        for step in (agent.get("plan") or [])[:10]:
            mark = _PLAN_MARKS.get(str(step.get("status")), "[ ]")
            color = _KEY_HELD if step.get("status") == "in_progress" else _TEXT
            self._blit(f"{mark} {step.get('step', '')}", left, y, color, self.font, _BG, inner_w)
            y += line_h
        for action in agent.get("actions") or []:
            if action.get("status") not in ("requested", "running"):
                continue
            checks = ", ".join(action.get("monitor") or [])
            text = f"{action['id']} {action['tool']} {action.get('elapsed_s', 0):.0f}s {checks}"
            self._blit(text, left, y, _WARN, self.mono, _BG, inner_w)
            y += line_h
        y += pad // 2
        pygame.draw.line(self.screen, _PANEL_BORDER, (left, y), (left + inner_w, y))
        y += pad // 2

        box_h = line_h + 2 * self._inner
        box_y = height - pad - box_h
        lines: list[tuple[str, tuple, str]] = []
        for entry in agent.get("transcript") or []:
            prefix, color = _CHAT_ROLES.get(str(entry.get("role")), ("", _TEXT))
            for paragraph in (prefix + str(entry.get("text", ""))).split("\n"):
                wrapped = wrap_text(self.font.size, paragraph, inner_w)
                lines += [(line, color, " " if k else "\n") for k, line in enumerate(wrapped)]
        self._chat_lines = [(text, joiner) for text, _color, joiner in lines]
        self._chat_rect = (x, y, width, box_y - y)
        self._input_rect = (left, box_y, inner_w, box_h)
        visible = max(0, (box_y - pad - y) // line_h)
        self._chat_scroll = min(self._chat_scroll, float(max(0, len(lines) - visible)))
        end = len(lines) - int(self._chat_scroll)
        span = self._selection_span()
        self._chat_drawn = []
        for i in range(max(0, end - visible), end):
            text, color, _joiner = lines[i]
            self._chat_drawn.append((i, left, y))
            self._blit(text, left, y, color, self.font, _BG, inner_w)
            if span is not None and span[0][0] <= i <= span[1][0]:
                c0 = span[0][1] if i == span[0][0] else 0
                c1 = span[1][1] if i == span[1][0] else len(text)
                if c1 > c0:
                    surf = self.font.render(text[c0:c1], True, color, _SELECT_BG)
                    self.screen.blit(surf, (left + self.font.size(text[:c0])[0], y))
            y += line_h

        box = pygame.Rect(left, box_y, inner_w, box_h)
        edit = self._input
        pygame.draw.rect(self.screen, _PANEL_BG, box, border_radius=self._radius)
        pygame.draw.rect(
            self.screen, _PANEL_BORDER if edit is None else _KEY_HELD, box, width=1, border_radius=self._radius
        )
        tx, ty, avail = left + self._inner, box_y + self._inner, inner_w - 2 * self._inner
        if edit is None:
            self._blit("Enter or click: message the agent (/cancel, /reset)", tx, ty, _DIM, self.font, _PANEL_BG, avail)
            return
        caret_w = max(1, round(self._scale))
        # Scroll just enough to keep the cursor inside the box.
        self._input_offset = min(self._input_offset, edit.cursor)
        while self.font.size(edit.text[self._input_offset : edit.cursor])[0] > avail - caret_w:
            self._input_offset += 1
        offset = self._input_offset
        self._input_view = (tx, offset)

        def at(col: int) -> int:
            return tx + self.font.size(edit.text[offset:max(offset, col)])[0]

        self.screen.set_clip(pygame.Rect(tx, box_y, avail, box_h))
        try:
            # Each run is rendered on its own opaque background: text without
            # one leaves the framebuffer's alpha at 0, which shows as a hole.
            start, end = (max(offset, col) for col in edit.selection())
            for a, b, bg in ((offset, start, _PANEL_BG), (start, end, _SELECT_BG), (end, len(edit.text), _PANEL_BG)):
                if b > a:
                    self.screen.blit(self.font.render(edit.text[a:b], True, _TEXT, bg), (at(a), ty))
            pygame.draw.rect(self.screen, (*_TEXT, 255), pygame.Rect(at(edit.cursor), ty + 2, caret_w, line_h - 4))
        finally:
            self.screen.set_clip(None)

    def _draw_camera_grid(self, x, y, width, height, images):
        self._image_rects = []
        if width <= 0 or height <= 0:
            return
        if not images:
            self._blit("(no cameras)", x + 8, y + 8, _DIM, self.font, _BG)
            return
        # One column: a view added later (the agent's) goes below, so the
        # first camera keeps the full width.
        cell_h = height // len(images)
        gap = max(4, self._pad // 2)
        for i, (label, img, bboxes, masks, bbox_texts) in enumerate(images):
            drawn = self._blit_image(
                img, bboxes, masks, x, y + i * cell_h, width - gap, cell_h - gap, bbox_texts,
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
