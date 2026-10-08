"""A teleop device sampled elsewhere, seen from the consumer's side."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from solarian.core.configs import RemoteInputConfig
from solarian.core.controller import Controller
from solarian.core.event import EventType
from solarian.core.execution import ExecutionMode
from solarian.core.keyboard_module import KeyboardModule
from solarian.core.mouse_module import MouseModule
from solarian.core.remote_input_module import RemoteInputModule
from solarian.core.ui.base import UICommand, UICommandType
from solarian.utils.shared_memory.observation_buffer import ObservationBuffer


def _module(stale_after_s: float = 0.25) -> RemoteInputModule:
    module = RemoteInputModule(
        RemoteInputConfig(frequency=60.0, stale_after_s=stale_after_s), "leader_arm"
    )
    module.observation_buffer = ObservationBuffer.local()
    module.start()
    return module


def _sample(value: float, stamp: float) -> dict:
    return {"shoulder_pan": {"value": value, "timestamp": stamp}}


def test_a_sample_becomes_an_ordinary_observation():
    module = _module()
    module.accept_sample(_sample(0.25, stamp=1000.0))
    module.update_observations()

    reading = module.observation_buffer.latest()["shoulder_pan"]
    assert reading["value"] == 0.25
    # Stamped on arrival in this process's clock, not with the sampler's, whose
    # clock is not synced to the relay's.
    assert reading["timestamp"] != 1000.0
    assert abs(reading["timestamp"] - time.time()) < 5.0


def test_inline_so_the_hosts_tick_is_its_tick():
    assert RemoteInputModule.execution_mode is ExecutionMode.INLINE


def test_out_of_order_samples_are_dropped():
    module = _module()
    module.accept_sample(_sample(0.1, stamp=1000.0))
    module.accept_sample(_sample(0.9, stamp=999.0))
    module.update_observations()

    assert module.observation_buffer.latest()["shoulder_pan"]["value"] == 0.1


def test_a_stale_link_publishes_nothing_rather_than_the_last_pose():
    module = _module(stale_after_s=0.05)
    module.accept_sample(_sample(0.4, stamp=1000.0))
    module.update_observations()
    assert module.observation_buffer.latest()

    time.sleep(0.08)
    module.update_observations()
    # Empty, so a follower emits no waypoint and holds where it is instead of
    # latching a target the operator stopped commanding.
    assert module.observation_buffer.latest() == {}

    module.accept_sample(_sample(0.5, stamp=1001.0))
    module.update_observations()
    assert module.observation_buffer.latest()["shoulder_pan"]["value"] == 0.5


def test_malformed_fields_are_ignored():
    module = _module()
    module.accept_sample({"shoulder_pan": "not a reading", "gripper": {"value": "x"}})
    module.update_observations()
    assert module.observation_buffer.latest() is None

    module.accept_sample({"gripper": {"value": 1, "timestamp": 5.0}, "junk": None})
    module.update_observations()
    assert set(module.observation_buffer.latest()) == {"gripper"}


def test_calibration_belongs_to_the_sampler():
    cfg = RemoteInputConfig(frequency=60.0, calibration_file="leader_arm.json")
    assert cfg.calibration_file is None, "a controller with no bus cannot calibrate"


class InputCtrl(Controller):
    """Just the module map and command routing (no zmq, no shared memory)."""

    def __init__(self):
        self._init_skill_orchestration()
        self.cfg = SimpleNamespace(fps=30)
        self.modules = {}
        self.observation_buffers = {}
        self.events = {"exit_early": False, "stop_recording": False}
        for name, module in (
            ("keyboard", KeyboardModule(RemoteInputConfig(frequency=60.0), "keyboard")),
            ("leader_arm", _module()),
        ):
            self.modules[name] = module
            self.observation_buffers[name] = module.observation_buffer
        self.keyboard = self.modules["keyboard"]
        self.keyboard.observation_buffer = ObservationBuffer.local()
        self.keyboard.start()
        self.mouse = MouseModule(MouseModule.config_class(**MouseModule.default_config), "mouse")
        self.mouse.observation_buffer = ObservationBuffer.local()
        self.mouse.start()


def test_input_sample_reaches_the_named_module():
    ctrl = InputCtrl()
    assert ctrl.handle_ui_command(
        UICommand(
            UICommandType.INPUT_SAMPLE,
            {"module": "leader_arm", "fields": _sample(0.3, stamp=1.0)},
        )
    )
    ctrl.modules["leader_arm"].update_observations()
    assert ctrl.observation_buffers["leader_arm"].latest()["shoulder_pan"]["value"] == 0.3


def test_a_sample_for_a_locally_wired_module_is_refused(caplog):
    ctrl = InputCtrl()
    with caplog.at_level("WARNING"):
        ctrl.handle_ui_command(
            UICommand(UICommandType.INPUT_SAMPLE, {"module": "keyboard", "fields": {}})
        )
    assert "not a remote-input module" in caplog.text


def test_ui_keys_become_keyboard_events():
    ctrl = InputCtrl()
    ctrl.handle_ui_command(UICommand(UICommandType.KEY_DOWN, {"key": "T"}))
    ctrl.handle_ui_command(UICommand(UICommandType.KEY_UP, {"key": "T"}))

    events = ctrl.keyboard.drain_events()
    assert [(int(event.type), event.data["key"]) for event in events] == [
        (int(EventType.KEY_DOWN), "t"),
        (int(EventType.KEY_UP), "t"),
    ], "a browser keypress is the same event a local one is"


def test_quit_stops_the_loop():
    ctrl = InputCtrl()
    assert ctrl.handle_ui_command(UICommand(UICommandType.STOP_ALL)) is True
    assert ctrl.handle_ui_command(UICommand(UICommandType.QUIT)) is False


def test_a_click_on_an_image_becomes_a_mouse_event():
    ctrl = InputCtrl()
    ctrl.handle_ui_command(
        UICommand(
            UICommandType.CLICK,
            {"label": "right_wrist_camera/image", "point": [12, 34], "button": 1},
        )
    )
    (event,) = ctrl.mouse.drain_events()
    assert int(event.type) == int(EventType.CLICK)
    assert event.data == {
        "point": (12, 34),
        "module": "right_wrist_camera",
        "field": "image",
        "button": 1,
    }


def test_a_click_off_any_image_is_ignored():
    ctrl = InputCtrl()
    ctrl.handle_ui_command(UICommand(UICommandType.CLICK, {"label": "", "point": [1, 2]}))
    assert ctrl.mouse.drain_events() == []
