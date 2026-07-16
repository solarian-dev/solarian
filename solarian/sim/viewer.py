"""Live overview viewer for a running :class:`~solarian.sim.robot.SimRobot`.

The MuJoCo world runs in its own (spawned) :class:`MujocoWorldServer` process so
that module subprocesses can talk to it over pipes. MuJoCo's native
``launch_passive`` viewer can't run there on macOS (it needs ``mjpython`` to own
the process main thread, which a spawned child never gets), so instead this
:class:`SimViewer` runs as a *separate* process whose own main thread hosts an
OpenCV window. Each frame it asks the world server (over a pipe, reusing the
same RPC protocol as the module ``BusClient``) to render the whole scene from an
orbit camera and displays the returned image.

Interaction:
    * left-drag  -> orbit (azimuth / elevation)
    * right-drag -> pan (move look-at point)
    * scroll / ``+`` / ``-`` -> zoom
    * ``ESC`` or closing the window -> close the viewer (the sim keeps running)
"""

from __future__ import annotations

import math
import time
from multiprocessing import Process
from multiprocessing.connection import Connection

import numpy as np

import logging

logger = logging.getLogger("sim_viewer")


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

        # Mutable camera + drag state, shared with the OpenCV mouse callback.
        state = {
            "azimuth": float(cam.get("azimuth", 90.0)),
            "elevation": float(cam.get("elevation", -20.0)),
            "distance": float(cam.get("distance", 1.5)),
            "lookat": np.asarray(cam.get("lookat", [0.0, 0.0, 0.2]), dtype=float),
            "drag": None,
            "x": 0,
            "y": 0,
        }

        def on_mouse(event, x, y, flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
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
                state["distance"] *= 0.9 if delta > 0 else 1.1

        try:
            cv2.namedWindow(self._title, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self._title, on_mouse)
        except Exception as exc:  # noqa: BLE001 - no display available
            logger.warning("Sim viewer could not open a window (%s); running headless.", exc)
            return

        logger.info("Sim viewer window opened (left-drag orbit, right-drag pan, scroll zoom, ESC to close).")
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

                cv2.imshow(self._title, np.ascontiguousarray(frame[:, :, ::-1]))  # RGB -> BGR
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                if key in (ord("+"), ord("=")):
                    state["distance"] *= 0.9
                elif key in (ord("-"), ord("_")):
                    state["distance"] *= 1.1
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


def _pan(state: dict, dx: int, dy: int) -> None:
    """Move the look-at point in the camera's image plane."""
    az = math.radians(state["azimuth"])
    el = math.radians(state["elevation"])
    forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    right = np.array([-math.sin(az), math.cos(az), 0.0])
    up = np.cross(right, forward)
    scale = state["distance"] * 0.0015
    state["lookat"] = state["lookat"] - right * (dx * scale) + up * (dy * scale)
