"""Live overview viewer for a robot running with ``is_sim = True``.

The MuJoCo world runs in its own (spawned) :class:`MujocoWorldServer` process so
that module subprocesses can talk to it over pipes. MuJoCo's native
``launch_passive`` viewer can't run there on macOS (it needs ``mjpython`` to own
the process main thread, which a spawned child never gets), so instead this
:class:`SimViewer` runs as a *separate* process whose own main thread hosts an
OpenCV window. Each frame it asks the world server (over a pipe, reusing the
same RPC protocol as the module ``BusClient``) to render the whole scene from an
orbit camera and displays the returned image.

Because it's a plain OpenCV window (not MuJoCo's own viewer) there are no native
UI widgets, so the camera controls are drawn onto the frame as an on-screen
toolbar of clickable buttons, plus a help overlay listing the shortcuts.

Interaction:
    * left-drag                 -> orbit (azimuth / elevation)
    * right/middle-drag         -> pan (move look-at point)
    * scroll / ``+`` / ``-``    -> zoom
    * arrow keys                -> orbit
    * ``W`` ``A`` ``S`` ``D``   -> pan
    * ``1``..``5``              -> front / back / left / right / top presets
    * ``R`` (or ``0``)          -> reset camera to the default framing
    * ``H`` / ``?``             -> toggle the on-screen help overlay
    * on-screen buttons         -> same actions via mouse
    * ``ESC`` or closing        -> close the viewer (the sim keeps running)
"""

from __future__ import annotations

import math
import time
from multiprocessing import Process
from multiprocessing.connection import Connection

import numpy as np

import logging

logger = logging.getLogger("sim_viewer")

# -- Camera step sizes for keyboard / button nudges --------------------------
ORBIT_STEP = 5.0  # degrees per key press
PAN_STEP = 28.0  # pixel-equivalent per key press (scaled by distance in _pan)
ZOOM_IN = 0.9
ZOOM_OUT = 1.1

# Preset views as (azimuth, elevation) in degrees. Distance / look-at are kept.
PRESET_VIEWS = {
    "front": (90.0, -10.0),
    "back": (270.0, -10.0),
    "left": (180.0, -10.0),
    "right": (0.0, -10.0),
    "top": (90.0, -89.0),
}

# Extended key codes for the arrow keys vary by platform / OpenCV backend, so
# match against the union of the common ones (Win32 / GTK / Cocoa).
KEY_LEFT = {2424832, 65361, 63234}
KEY_UP = {2490368, 65362, 63232}
KEY_RIGHT = {2555904, 65363, 63235}
KEY_DOWN = {2621440, 65364, 63233}

HELP_LINES = [
    "Camera controls",
    "",
    "left-drag        orbit",
    "right-drag       pan",
    "scroll / + -     zoom",
    "arrow keys       orbit",
    "W A S D          pan",
    "1 2 3 4 5        front/back/left/right/top",
    "R  (or 0)        reset view",
    "H  /  ?          toggle this help",
    "ESC              close viewer",
]


class SimViewer(Process):
    """Separate process that renders and displays the sim's overview window."""

    def __init__(
        self,
        conn: Connection,
        width: int = 960,
        height: int = 720,
        title: str = "Solarian Sim",
        log_queue=None,
    ):
        super().__init__(daemon=True)
        self._conn = conn
        self._w = int(width)
        self._h = int(height)
        self._title = title
        self._log_queue = log_queue

    # -- RPC to the world server ---------------------------------------
    def _rpc(self, method: str, *args):
        self._conn.send((method, args, {}))
        result, error = self._conn.recv()
        if error is not None:
            raise error
        return result

    # ------------------------------------------------------------------
    def run(self):
        try:
            from solarian.utils import setup_process_logging

            setup_process_logging(logger_name=None, log_queue=self._log_queue, install_fallback=False)
        except Exception:  # noqa: BLE001 - logging is best-effort
            pass

        try:
            import cv2
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sim viewer unavailable (OpenCV import failed: %s).", exc)
            return

        try:
            cam = self._rpc("overview_defaults")
        except Exception as exc:  # noqa: BLE001 - world server not ready/alive
            logger.warning("Sim viewer could not query the world (%s); not opening.", exc)
            return

        # Default framing, kept so "reset" can restore it exactly.
        defaults = {
            "azimuth": float(cam.get("azimuth", 90.0)),
            "elevation": float(cam.get("elevation", -20.0)),
            "distance": float(cam.get("distance", 1.5)),
            "lookat": np.asarray(cam.get("lookat", [0.0, 0.0, 0.2]), dtype=float),
        }

        # Mutable camera + drag state, shared with the OpenCV mouse callback.
        state = {
            "azimuth": defaults["azimuth"],
            "elevation": defaults["elevation"],
            "distance": defaults["distance"],
            "lookat": defaults["lookat"].copy(),
            "drag": None,
            "x": 0,
            "y": 0,
            "show_help": False,
        }

        buttons = _build_buttons(cv2)

        def apply_action(action: str) -> None:
            if action in PRESET_VIEWS:
                state["azimuth"], state["elevation"] = PRESET_VIEWS[action]
            elif action == "zoom_in":
                state["distance"] *= ZOOM_IN
            elif action == "zoom_out":
                state["distance"] *= ZOOM_OUT
            elif action == "reset":
                state["azimuth"] = defaults["azimuth"]
                state["elevation"] = defaults["elevation"]
                state["distance"] = defaults["distance"]
                state["lookat"] = defaults["lookat"].copy()
            elif action == "help":
                state["show_help"] = not state["show_help"]

        def on_mouse(event, x, y, flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for btn in buttons:
                    if btn["x1"] <= x <= btn["x2"] and btn["y1"] <= y <= btn["y2"]:
                        apply_action(btn["action"])
                        return
                state["drag"], state["x"], state["y"] = "orbit", x, y
            elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
                state["drag"], state["x"], state["y"] = "pan", x, y
            elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
                state["drag"] = None
            elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
                dx, dy = x - state["x"], y - state["y"]
                state["x"], state["y"] = x, y
                if state["drag"] == "orbit":
                    state["azimuth"] = (state["azimuth"] - dx * 0.4) % 360.0
                    state["elevation"] = max(-89.0, min(89.0, state["elevation"] - dy * 0.4))
                else:
                    _pan(state, dx, dy)
            elif event == cv2.EVENT_MOUSEWHEEL:
                delta = cv2.getMouseWheelDelta(flags) if hasattr(cv2, "getMouseWheelDelta") else flags
                state["distance"] *= ZOOM_IN if delta > 0 else ZOOM_OUT

        try:
            cv2.namedWindow(self._title, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self._title, on_mouse)
        except Exception as exc:  # noqa: BLE001 - no display available
            logger.warning("Sim viewer could not open a window (%s); running headless.", exc)
            return

        logger.info(
            "Sim viewer window opened (drag/arrows orbit, WASD pan, scroll zoom, "
            "1-5 presets, R reset, H help, ESC close)."
        )
        period = 1.0 / 30.0
        try:
            while True:
                start = time.perf_counter()
                cam_req = {
                    "width": self._w,
                    "height": self._h,
                    "azimuth": state["azimuth"],
                    "elevation": state["elevation"],
                    "distance": state["distance"],
                    "lookat": [float(v) for v in state["lookat"]],
                }
                try:
                    frame = self._rpc("render_overview", cam_req)
                except (EOFError, OSError, ConnectionError):
                    break  # world server gone
                if frame is None:
                    break

                # RGB -> BGR, then draw the (BGR) overlay on top.
                display = np.ascontiguousarray(frame[:, :, ::-1])
                _draw_overlay(cv2, display, buttons, state)
                cv2.imshow(self._title, display)

                code = cv2.waitKeyEx(1)
                if code != -1 and _handle_key(code, state, apply_action):
                    break
                try:
                    if cv2.getWindowProperty(self._title, cv2.WND_PROP_VISIBLE) < 1:
                        break  # window closed
                except Exception:  # noqa: BLE001
                    break

                sleep = period - (time.perf_counter() - start)
                if sleep > 0:
                    time.sleep(sleep)
        finally:
            try:
                cv2.destroyWindow(self._title)
                cv2.waitKey(1)
            except Exception:  # noqa: BLE001
                pass
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass


def _handle_key(code: int, state: dict, apply_action) -> bool:
    """Apply a keyboard action. Returns ``True`` when the viewer should close."""
    if code == 27:  # ESC
        return True

    # Arrow keys use extended codes; handle and return before the ASCII checks
    # below, since some arrow codes alias ASCII letters under ``& 0xFF``.
    if code in KEY_LEFT:
        state["azimuth"] = (state["azimuth"] + ORBIT_STEP) % 360.0
        return False
    if code in KEY_RIGHT:
        state["azimuth"] = (state["azimuth"] - ORBIT_STEP) % 360.0
        return False
    if code in KEY_UP:
        state["elevation"] = max(-89.0, min(89.0, state["elevation"] - ORBIT_STEP))
        return False
    if code in KEY_DOWN:
        state["elevation"] = max(-89.0, min(89.0, state["elevation"] + ORBIT_STEP))
        return False

    key = code & 0xFF
    if key in (ord("+"), ord("=")):
        state["distance"] *= ZOOM_IN
    elif key in (ord("-"), ord("_")):
        state["distance"] *= ZOOM_OUT
    elif key in (ord("w"), ord("W")):
        _pan(state, 0, PAN_STEP)
    elif key in (ord("s"), ord("S")):
        _pan(state, 0, -PAN_STEP)
    elif key in (ord("a"), ord("A")):
        _pan(state, PAN_STEP, 0)
    elif key in (ord("d"), ord("D")):
        _pan(state, -PAN_STEP, 0)
    elif key in (ord("r"), ord("R"), ord("0")):
        apply_action("reset")
    elif key in (ord("h"), ord("H"), ord("?")):
        apply_action("help")
    elif key == ord("1"):
        apply_action("front")
    elif key == ord("2"):
        apply_action("back")
    elif key == ord("3"):
        apply_action("left")
    elif key == ord("4"):
        apply_action("right")
    elif key == ord("5"):
        apply_action("top")
    return False


def _build_buttons(cv2) -> list:
    """Lay out the clickable toolbar buttons across the top of the window."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    specs = [
        ("Front", "front"),
        ("Back", "back"),
        ("Left", "left"),
        ("Right", "right"),
        ("Top", "top"),
        ("Reset", "reset"),
        ("-", "zoom_out"),
        ("+", "zoom_in"),
        ("?", "help"),
    ]
    buttons = []
    x = 8
    y1, height, pad, gap = 8, 26, 10, 6
    for label, action in specs:
        (tw, _), _ = cv2.getTextSize(label, font, scale, 1)
        w = max(tw + 2 * pad, 26)
        buttons.append(
            {"label": label, "action": action, "x1": x, "y1": y1, "x2": x + w, "y2": y1 + height}
        )
        x += w + gap
    return buttons


def _draw_overlay(cv2, img: np.ndarray, buttons: list, state: dict) -> None:
    """Draw the toolbar (and help panel, if toggled) onto the BGR frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    overlay = img.copy()
    for btn in buttons:
        cv2.rectangle(overlay, (btn["x1"], btn["y1"]), (btn["x2"], btn["y2"]), (45, 45, 45), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for btn in buttons:
        cv2.rectangle(img, (btn["x1"], btn["y1"]), (btn["x2"], btn["y2"]), (210, 210, 210), 1)
        tx = btn["x1"] + 8
        ty = btn["y2"] - 8
        cv2.putText(img, btn["label"], (tx, ty), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    hint = "H = help"
    cv2.putText(img, hint, (10, img.shape[0] - 12), font, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

    if state["show_help"]:
        _draw_help(cv2, img)


def _draw_help(cv2, img: np.ndarray) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = 22
    box_w = 330
    box_h = line_h * len(HELP_LINES) + 24
    x0, y0 = 8, 44
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (210, 210, 210), 1)
    y = y0 + 26
    for i, line in enumerate(HELP_LINES):
        color = (120, 220, 255) if i == 0 else (235, 235, 235)
        cv2.putText(img, line, (x0 + 14, y), font, 0.45, color, 1, cv2.LINE_AA)
        y += line_h


def _pan(state: dict, dx: float, dy: float) -> None:
    """Move the look-at point in the camera's image plane."""
    az = math.radians(state["azimuth"])
    el = math.radians(state["elevation"])
    forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    right = np.array([-math.sin(az), math.cos(az), 0.0])
    up = np.cross(right, forward)
    scale = state["distance"] * 0.0015
    state["lookat"] = state["lookat"] - right * (dx * scale) + up * (dy * scale)
