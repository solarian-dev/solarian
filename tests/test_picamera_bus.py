"""PiCameraBus reconnects after CSI stalls without crashing the module process."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from solarian.core.data import ImageData, PosData
from solarian.core.picamera_bus import PiCameraBus


def _cfg(**overrides):
    values = dict(
        port="0",
        fps=30,
        width=8,
        height=6,
        rotation=0,
        channels=3,
        color_mode="bgr",
        video_format="MJPG",
        resize_width=None,
        resize_height=None,
        optical_remap=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class FakePicamera2:
    instances: list["FakePicamera2"] = []
    fail_opens = 0
    fail_captures = 0
    empty_frames = 0
    reject_wait_timeout = False
    capture_exc_type: type[BaseException] = RuntimeError

    def __init__(self, camera_num=0):
        type(self).instances.append(self)
        if type(self).fail_opens > 0:
            type(self).fail_opens -= 1
            raise RuntimeError("camera busy")
        self.camera_num = camera_num
        self.started = False
        self.closed = False
        self.cancelled = False
        self.captures = 0
        self.config = None
        self.last_wait = None

    def create_video_configuration(self, main=None, controls=None, buffer_count=None):
        return {"main": main, "controls": controls, "buffer_count": buffer_count}

    def configure(self, config):
        self.config = config

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def cancel_all_and_flush(self):
        self.cancelled = True

    def capture_array(self, name="main", wait=None):
        self.last_wait = wait
        if type(self).reject_wait_timeout and wait not in (None, True, False):
            raise TypeError("wait must be bool or None")
        if type(self).fail_captures > 0:
            type(self).fail_captures -= 1
            raise type(self).capture_exc_type("Camera disconnected")
        if type(self).empty_frames > 0:
            type(self).empty_frames -= 1
            return None
        self.captures += 1
        return np.zeros((6, 8, 3), dtype=np.uint8)


@pytest.fixture
def fake_picamera2(monkeypatch):
    FakePicamera2.instances = []
    FakePicamera2.fail_opens = 0
    FakePicamera2.fail_captures = 0
    FakePicamera2.empty_frames = 0
    FakePicamera2.reject_wait_timeout = False
    FakePicamera2.capture_exc_type = RuntimeError
    mod = types.ModuleType("picamera2")
    mod.Picamera2 = FakePicamera2
    monkeypatch.setitem(sys.modules, "picamera2", mod)
    monkeypatch.setattr("solarian.core.picamera_bus.time.sleep", lambda *_a, **_k: None)
    return FakePicamera2


def _bus() -> PiCameraBus:
    bus = PiCameraBus(port="0", components={"camera": SimpleNamespace(cfg=_cfg())})
    bus.ae_settle_s = 0.0
    bus.reconnect_settle_s = 0.0
    bus.reconnect_min_delay_s = 0.0
    bus.reconnect_max_delay_s = 0.0
    bus.close_timeout_s = 1.0
    return bus


def test_read_returns_frame(fake_picamera2):
    bus = _bus()
    bus.connect()
    out = bus.read(ImageData, ["camera"])
    assert "camera" in out
    assert out["camera"].shape == (6, 8, 3)
    assert fake_picamera2.instances[0].captures == 1
    assert fake_picamera2.instances[0].last_wait == bus.capture_timeout_s
    assert fake_picamera2.instances[0].config["buffer_count"] == 6


def test_rejects_non_image_data(fake_picamera2):
    bus = _bus()
    bus.connect()
    with pytest.raises(ValueError, match="ImageData"):
        bus.read(PosData, ["camera"])


def test_reconnects_after_capture_error(fake_picamera2):
    bus = _bus()
    bus.connect()
    fake_picamera2.fail_captures = 1
    out = bus.read(ImageData, ["camera"])
    assert out["camera"].shape == (6, 8, 3)
    assert len(fake_picamera2.instances) == 2
    assert fake_picamera2.instances[0].closed is True
    assert fake_picamera2.instances[0].cancelled is True
    assert fake_picamera2.instances[1].captures == 1


def test_reconnects_after_empty_frame(fake_picamera2):
    bus = _bus()
    bus.connect()
    fake_picamera2.empty_frames = 1
    out = bus.read(ImageData, ["camera"])
    assert out["camera"].shape == (6, 8, 3)
    assert len(fake_picamera2.instances) == 2


def test_retries_open_until_camera_returns(fake_picamera2):
    bus = _bus()
    bus.connect()
    fake_picamera2.fail_captures = 1
    fake_picamera2.fail_opens = 2
    out = bus.read(ImageData, ["camera"])
    assert out["camera"].shape == (6, 8, 3)
    # original + 2 failed opens + 1 successful reopen
    assert len(fake_picamera2.instances) == 4
    assert fake_picamera2.instances[-1].captures == 1


def test_reconnects_after_capture_timeout(fake_picamera2):
    bus = _bus()
    bus.connect()
    fake_picamera2.capture_exc_type = TimeoutError
    fake_picamera2.fail_captures = 1
    out = bus.read(ImageData, ["camera"])
    assert out["camera"].shape == (6, 8, 3)
    assert len(fake_picamera2.instances) == 2


def test_keeps_optical_remap_across_reconnect(fake_picamera2, monkeypatch):
    loads = {"n": 0}

    def fake_load(_cfg):
        loads["n"] += 1
        return None

    monkeypatch.setattr("solarian.core.optical_remap.load_bus_remap", fake_load)
    bus = _bus()
    bus.connect()
    assert loads["n"] == 1
    fake_picamera2.fail_captures = 1
    bus.read(ImageData, ["camera"])
    assert loads["n"] == 1


def test_capture_falls_back_when_wait_timeout_unsupported(fake_picamera2):
    fake_picamera2.reject_wait_timeout = True
    bus = _bus()
    bus.connect()
    out = bus.read(ImageData, ["camera"])
    assert out["camera"].shape == (6, 8, 3)


def test_connect_is_idempotent(fake_picamera2):
    bus = _bus()
    bus.connect()
    bus.connect()
    assert len(fake_picamera2.instances) == 1


def test_port_must_be_csi_index(fake_picamera2):
    bus = PiCameraBus(
        port="/dev/video0",
        components={"camera": SimpleNamespace(cfg=_cfg(port="/dev/video0"))},
    )
    with pytest.raises(ValueError, match="CSI camera index"):
        bus.connect()
