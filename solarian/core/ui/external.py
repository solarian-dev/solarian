"""WebSocket front end: hand the UI to another process, in any language.

``ui=external`` turns a controller into a headless engine with a machine-readable
UI. It serves msgpack over a loopback WebSocket:

* ``hello``    — once per connection: the skills that can be started, the key
  bindings, the observation schema. Enough to build a UI without importing the
  project's Python.
* ``snapshot`` — one per frame. A camera JPEG is the one the robot sent;
  only a frame that arrived as pixels is encoded here. Pictures are rate-limited.
* commands flow the other way as ``{"type": ..., "data": {...}}``, exactly the
  :class:`~solarian.core.ui.base.UICommand` wire form.

Frames are latest-only: a snapshot that a slow client has not picked up is
replaced rather than queued, the same choice ``ZMQ_CONFLATE`` makes on the
observation socket. A UI that stalls falls behind in freshness, never in time.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque

import msgpack
import websockets
from websockets.asyncio.server import serve

from solarian.core.data import encode_jpeg
from solarian.core.ui.base import UICommand, UICommandType, UIImage, UISnapshot
from solarian.core.ui.overlay import mask_rings

logger = logging.getLogger(__name__)


class _Slot:
    """Single-consumer latest-only mailbox.

    An :class:`asyncio.Queue` of size one would block the producer once full;
    here a new frame overwrites the pending one, which is what "latest-only"
    means. Producer side runs on the event loop thread (via
    ``call_soon_threadsafe``), consumer is the one task that owns the socket.
    """

    def __init__(self):
        self._event = asyncio.Event()
        self._value: bytes | None = None

    def put(self, value: bytes) -> None:
        self._value = value
        self._event.set()

    async def get(self) -> bytes:
        await self._event.wait()
        self._event.clear()
        value, self._value = self._value, None
        return value


class ExternalUI:
    """Serve the UI protocol over a loopback WebSocket.

    Args:
        bind: ``host:port`` to listen on. Loopback by default: the intended peer
            is a bridge in the same container, not the public internet.
        token: Shared secret the client must present in an ``auth`` frame. A
            controller with no token accepts any local connection.
        image_fps: Cap on snapshot frames that carry images. The dashboard is
            cheap; JPEG is not. The JPEG is the one the robot sent, at that
            frame's own size and quality.
        hello: Payload for the one-shot ``hello`` message.
    """

    captures_keys = True

    def __init__(
        self,
        *,
        bind: str = "127.0.0.1:8700",
        token: str | None = None,
        image_fps: int = 15,
        hello: dict | None = None,
    ):
        self._token = token
        self._image_period_ns = int(1e9 / image_fps) if image_fps else 0
        self._hello = hello or {}

        self._commands: deque[UICommand] = deque(maxlen=4096)
        # label -> (width, height) of the frame the client was sent. A click
        # comes back in those pixels, which are the observation's own.
        self._sent_frame: dict[str, tuple[int, int]] = {}
        self._clients: set[_Slot] = set()
        self._latest: UISnapshot | None = None
        self._latest_lock = threading.Lock()
        self._broadcast_scheduled = False
        self._last_image_ns = 0

        host, _, port = bind.rpartition(":")
        #: Where the server actually listens, for whoever has to connect to it.
        self.host = host or "127.0.0.1"
        self.port = int(port)

        self._loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="solarian-ui-external", daemon=True
        )
        self._thread.start()
        if not self._started.wait(timeout=15.0):
            raise RuntimeError(f"External UI failed to bind {bind} within 15s")
        if self._error is not None:
            raise self._error

    # -- ControllerUI ------------------------------------------------------
    def poll(self) -> list[UICommand]:
        commands = []
        while self._commands:
            commands.append(self._commands.popleft())
        return commands

    def render(self, snapshot: UISnapshot) -> None:
        with self._latest_lock:
            self._latest = snapshot
            if self._broadcast_scheduled:
                # A broadcast is already queued and will pick up this snapshot,
                # so the control loop never pays for a frame twice.
                return
            self._broadcast_scheduled = True
        self._loop.call_soon_threadsafe(self._broadcast)

    def tick(self, fps: int) -> None:
        # The event loop owns its own timing; this only paces the control loop.
        time.sleep(1.0 / fps if fps else 0.01)

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    # -- event loop thread -------------------------------------------------
    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start_server())
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            self._error = exc
            self._started.set()
            return
        self._started.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    async def _start_server(self) -> None:
        # compression=None: snapshot payloads are mostly JPEG, which deflate
        # cannot shrink and would only cost latency to try.
        self._server = await serve(
            self._serve_client,
            self.host,
            self.port,
            compression=None,
            max_size=1 << 20,
        )
        logger.info(
            "External UI listening on ws://%s:%s (token %s)",
            self.host,
            self.port,
            "required" if self._token else "not set",
        )

    async def _serve_client(self, connection) -> None:
        auth = await self._handshake(connection)
        if auth is None:
            return
        try:
            await connection.send(msgpack.packb({"type": "hello", "data": self._hello}))
        except websockets.exceptions.ConnectionClosed:
            # A readiness probe connects, learns the socket is up and hangs up
            # again. Letting that reach the library's handler wrapper would put a
            # traceback in the runner's log for a successful check.
            return

        # A client that only sends — the input stream, which belongs on its own
        # connection so 60 Hz samples never queue behind a JPEG frame — asks not
        # to be subscribed rather than receiving snapshots it would discard.
        slot = _Slot() if auth.get("snapshots", True) else None
        sender = None
        if slot is not None:
            self._clients.add(slot)
            sender = asyncio.create_task(self._send_frames(connection, slot))
        try:
            async for raw in connection:
                self._ingest(raw)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if slot is not None:
                self._clients.discard(slot)
                sender.cancel()
            logger.info("External UI client disconnected")

    async def _handshake(self, connection) -> dict | None:
        """The client's ``auth`` payload, or ``None`` if it may not connect.

        With no token configured there is nothing to wait for, and the first
        frame is a command: an empty payload means "default preferences".
        """
        if self._token is None:
            return {}
        try:
            raw = await asyncio.wait_for(connection.recv(), timeout=10.0)
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
            return None
        message = self._decode(raw)
        data = (message.get("data") or {}) if message else {}
        if message is None or message.get("type") != "auth" or data.get("token") != self._token:
            await connection.close(code=4401, reason="unauthorized")
            logger.warning("External UI rejected an unauthenticated client")
            return None
        return data

    async def _send_frames(self, connection, slot: _Slot) -> None:
        try:
            while True:
                payload = await slot.get()
                await connection.send(payload)
        except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed):
            pass

    def _broadcast(self) -> None:
        with self._latest_lock:
            snapshot = self._latest
            self._latest = None
            self._broadcast_scheduled = False
        if snapshot is None or not self._clients:
            return
        payload = msgpack.packb(
            {"type": "snapshot", "data": self._encode_snapshot(snapshot)}
        )
        for slot in self._clients:
            slot.put(payload)

    def _ingest(self, raw) -> None:
        message = self._decode(raw)
        if message is None:
            return
        try:
            command = UICommand.from_dict(message)
        except (KeyError, ValueError):
            logger.debug("External UI ignored unknown command %r", message)
            return
        if command.type is UICommandType.CLICK:
            command = self._click_in_capture_pixels(command)
        self._commands.append(command)

    def _click_in_capture_pixels(self, command: UICommand) -> UICommand:
        """Clamp a click to the frame the client was sent.

        The point is already in that frame's pixels: the JPEG is the
        observation image, not a resized copy of it.
        """
        point = command.data.get("point")
        sent = self._sent_frame.get(str(command.data.get("label") or ""))
        if sent is None or not point:
            return command
        try:
            px, py = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            return command
        width, height = sent
        return UICommand(
            type=command.type,
            data={
                **command.data,
                "point": [
                    min(width - 1, max(0, int(round(px)))),
                    min(height - 1, max(0, int(round(py)))),
                ],
            },
        )

    @staticmethod
    def _decode(raw) -> dict | None:
        try:
            message = msgpack.unpackb(raw if isinstance(raw, bytes) else raw.encode())
        except Exception:  # noqa: BLE001 - anything can arrive off a socket
            logger.debug("External UI could not decode a frame", exc_info=True)
            return None
        return message if isinstance(message, dict) else None

    # -- serialization -----------------------------------------------------
    def _encode_snapshot(self, snapshot: UISnapshot) -> dict:
        return {
            "t_ns": snapshot.t_ns,
            "fps": snapshot.fps,
            "sections": [
                [title, [list(row) for row in rows]] for title, rows in snapshot.sections
            ],
            "images": self._encode_images(snapshot.images),
            "key_help": [list(row) for row in snapshot.key_help],
            "skills": {key: list(value) for key, value in snapshot.skills.items()},
            "telemetry": dict(snapshot.telemetry),
            "recording": dict(snapshot.recording),
            "agent": snapshot.agent,
            "status": snapshot.status,
        }

    def _encode_images(self, images: list[UIImage]) -> list[dict]:
        now_ns = time.monotonic_ns()
        if self._image_period_ns and now_ns - self._last_image_ns < self._image_period_ns:
            # Dashboard frames stay at full rate; only the pixels are throttled.
            return []
        self._last_image_ns = now_ns

        encoded = []
        for image in images:
            # The robot's JPEG is the picture. Encoding ``data`` again would
            # be a second generation, at a size and quality the robot did not
            # choose. ``data`` is only encoded when the frame never was a JPEG.
            jpeg = image.jpeg
            if not jpeg:
                try:
                    jpeg = encode_jpeg(image.data)
                except RuntimeError:
                    logger.debug("Could not JPEG-encode %s", image.label, exc_info=True)
                    continue
            height, width = image.data.shape[:2]
            self._sent_frame[image.label] = (int(width), int(height))
            encoded.append(
                {
                    "label": image.label,
                    "jpeg": jpeg,
                    "width": int(width),
                    "height": int(height),
                    "bboxes": [[float(v) for v in bbox[:4]] for bbox in image.bboxes],
                    "bbox_texts": list(image.bbox_texts),
                    # Flat [x, y, ...] rings, filled even-odd: Solarian Base's TargetOutlines.
                    "outlines": [
                        ring.round().astype(int).reshape(-1).tolist()
                        for mask in image.masks
                        for ring in mask_rings(mask)
                    ],
                }
            )
        return encoded
