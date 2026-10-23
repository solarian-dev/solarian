"""End-to-end tests for the ``ui=external`` WebSocket bridge."""

from __future__ import annotations

import asyncio
import logging
import socket

import msgpack
import numpy as np
import pytest
import websockets

from solarian.core.ui.base import UICommandType, UIImage, UISnapshot
from solarian.core.ui.external import ExternalUI

HELLO = {"controller": "TestController", "skills": [{"name": "teleop"}]}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def ui():
    served = ExternalUI(
        bind=f"127.0.0.1:{_free_port()}",
        token="s3cret",
        image_fps=1000,
        hello=HELLO,
    )
    yield served
    served.close()


def _snapshot() -> UISnapshot:
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[4:20, 8:24] = 1
    return UISnapshot(
        t_ns=123,
        fps=30,
        sections=[("obs · arm", [("j0", "0.100"), ("temp", "51.0", (240, 80, 80))])],
        images=[
            UIImage(
                label="cam/rgb",
                data=np.full((48, 64, 3), 7, dtype=np.uint8),
                bboxes=[(2, 4, 6, 8)],
                masks=[mask],
            )
        ],
        key_help=[("space", "stop all skills")],
        skills={"available": ["teleop"], "running": []},
        telemetry={"Obs latency (ms)": "12"},
        status="idle",
    )


def _url(ui: ExternalUI) -> str:
    return f"ws://{ui.host}:{ui.port}"


async def _connect(ui: ExternalUI, token: str = "s3cret"):
    connection = await websockets.connect(_url(ui))
    await connection.send(msgpack.packb({"type": "auth", "data": {"token": token}}))
    return connection


def test_hello_then_snapshot_round_trip(ui):
    async def scenario():
        async with await _connect(ui) as connection:
            hello = msgpack.unpackb(await connection.recv())
            assert hello["type"] == "hello"
            assert hello["data"] == HELLO

            # Snapshots only go out to connected clients, so render after the
            # handshake rather than before it.
            await asyncio.sleep(0.1)
            ui.render(_snapshot())
            frame = msgpack.unpackb(await asyncio.wait_for(connection.recv(), 5))

        assert frame["type"] == "snapshot"
        data = frame["data"]
        assert data["t_ns"] == 123
        assert data["sections"][0][0] == "obs · arm"
        assert data["telemetry"] == {"Obs latency (ms)": "12"}
        assert data["skills"] == {"available": ["teleop"], "running": []}

        image = data["images"][0]
        assert image["label"] == "cam/rgb"
        assert image["width"] == 64, "the frame is sent at the observation's own size"
        assert image["height"] == 48
        assert image["jpeg"][:2] == b"\xff\xd8", "JPEG SOI marker"
        assert image["bboxes"] == [[2.0, 4.0, 6.0, 8.0]], "boxes stay in observation pixels"
        (ring,) = image["outlines"]
        xs, ys = ring[0::2], ring[1::2]
        assert (min(xs), max(xs), min(ys), max(ys)) == (8, 23, 4, 19), "rings stay in observation pixels"

    asyncio.run(scenario())


def test_a_jpeg_from_the_robot_is_forwarded_unchanged(ui):
    """The browser is shown the frame the robot encoded, not a second one."""
    robot_jpeg = b"\xff\xd8" + b"robot-jpeg-q90" + b"\xff\xd9"
    snapshot = _snapshot()
    snapshot.images[0].jpeg = robot_jpeg

    async def scenario():
        async with await _connect(ui) as connection:
            await connection.recv()  # hello
            await asyncio.sleep(0.1)
            ui.render(snapshot)
            frame = msgpack.unpackb(await asyncio.wait_for(connection.recv(), 5))
        image = frame["data"]["images"][0]
        assert image["jpeg"] == robot_jpeg
        assert (image["width"], image["height"]) == (64, 48)

    asyncio.run(scenario())


def test_commands_reach_the_controller(ui):
    async def scenario():
        async with await _connect(ui) as connection:
            await connection.recv()  # hello
            await connection.send(
                msgpack.packb({"type": "key_down", "data": {"key": "t"}})
            )
            await connection.send(
                msgpack.packb(
                    {"type": "input_sample", "data": {"module": "leader_arm"}}
                )
            )
            await connection.send(msgpack.packb({"type": "nonsense", "data": {}}))
            await asyncio.sleep(0.2)

    asyncio.run(scenario())

    commands = ui.poll()
    assert [command.type for command in commands] == [
        UICommandType.KEY_DOWN,
        UICommandType.INPUT_SAMPLE,
    ], "an unknown command type is dropped, not fatal"
    assert commands[0].data == {"key": "t"}
    assert ui.poll() == [], "polling drains"


def test_a_click_is_already_in_image_pixels(ui):
    """The JPEG is the observation image, so a click needs no rescaling."""

    async def scenario():
        async with await _connect(ui) as connection:
            await connection.recv()  # hello
            await asyncio.sleep(0.1)
            ui.render(_snapshot())
            await asyncio.wait_for(connection.recv(), 5)
            await connection.send(
                msgpack.packb(
                    {"type": "click", "data": {"label": "cam/rgb", "point": [16, 12]}}
                )
            )
            await connection.send(
                msgpack.packb(
                    {"type": "click", "data": {"label": "other/image", "point": [16, 12]}}
                )
            )
            await asyncio.sleep(0.2)

    asyncio.run(scenario())

    tracked, unknown = ui.poll()
    assert tracked.data["point"] == [16, 12]
    assert tracked.data["label"] == "cam/rgb"
    assert unknown.data["point"] == [16, 12], "an unseen image has no frame to clamp to"


def test_a_send_only_client_is_not_sent_snapshots(ui):
    """The input stream asks out of the snapshot subscription it would discard."""

    async def scenario():
        connection = await websockets.connect(_url(ui))
        await connection.send(
            msgpack.packb(
                {"type": "auth", "data": {"token": "s3cret", "snapshots": False}}
            )
        )
        async with connection:
            hello = msgpack.unpackb(await connection.recv())
            assert hello["type"] == "hello"

            await asyncio.sleep(0.1)
            ui.render(_snapshot())
            await connection.send(
                msgpack.packb({"type": "input_sample", "data": {"module": "leader_arm"}})
            )
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(connection.recv(), 0.5)

    asyncio.run(scenario())

    assert [command.type for command in ui.poll()] == [UICommandType.INPUT_SAMPLE], (
        "it still sends, which is the whole point of the connection"
    )


def test_a_client_that_hangs_up_mid_handshake_is_not_an_error(ui, caplog):
    """What a readiness probe does: connect, see the socket is up, leave.

    The runner's log is how a failed session gets diagnosed, so a successful
    check must not leave a traceback in it.
    """

    async def scenario():
        connection = await _connect(ui)
        await connection.close()
        await asyncio.sleep(0.2)

    with caplog.at_level(logging.ERROR):
        asyncio.run(scenario())

    assert "connection handler failed" not in caplog.text


def test_bad_token_is_rejected(ui):
    async def scenario():
        connection = await _connect(ui, token="wrong")
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await asyncio.wait_for(connection.recv(), 5)

    asyncio.run(scenario())


def test_image_frames_are_rate_limited():
    served = ExternalUI(
        bind=f"127.0.0.1:{_free_port()}",
        image_fps=1,
        hello=HELLO,
    )

    async def scenario():
        async with await websockets.connect(_url(served)) as connection:
            await connection.recv()  # hello; no token configured
            await asyncio.sleep(0.1)

            first, second = [], []
            for target in (first, second):
                served.render(_snapshot())
                frame = msgpack.unpackb(await asyncio.wait_for(connection.recv(), 5))
                target.extend(frame["data"]["images"])
                await asyncio.sleep(0.05)
            return first, second

    try:
        first, second = asyncio.run(scenario())
    finally:
        served.close()

    assert len(first) == 1
    assert second == [], "the second frame lands inside the 1 fps image budget"
