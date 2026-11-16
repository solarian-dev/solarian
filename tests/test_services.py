"""Services and server-hosted skills: routing, transport, and the processor."""

import logging
import sys
import threading
import time

import numpy as np
import pytest
import zmq

from solarian.core.action import ActionType
from solarian.core.clock_sync import LatencyEma
from solarian.core.data import BoundingBoxData, ImageData, PointData
from solarian.core.execution import ExecutionPlace
from solarian.core.message import Message
from solarian.core.observation import CalculatedField
from solarian.core.robot import Robot
from solarian.core.server import RelayServer
from solarian.core.server_skill_host import ServerSkillHost
from solarian.core.skill import Skill
from solarian.services import ServiceProcessor
from solarian.utils.log_buffer import LogRecord


class _Socket:
    def __init__(self, inbox=()):
        self.inbox = [m.serialize() for m in inbox]
        self.sent: list[Message] = []

    def send(self, raw, flags=0):
        self.sent.append(Message.deserialize(raw))

    def recv(self, flags=0):
        if not self.inbox:
            raise zmq.Again()
        return self.inbox.pop(0)


class _Proc:
    def __init__(self):
        self.calls = []

    def start(self, params):
        self.calls.append(("start", params))

    def stop(self):
        self.calls.append(("stop",))


class _Host:
    def __init__(self, running=()):
        self.running = set(running)
        self.started, self.stopped = [], []

    def runs(self, name):
        return name in self.running

    def start_named_skill(self, data):
        self.started.append(data)

    def stop_named_skill(self, data):
        self.stopped.append(data["skill"])


def _relay(processors=None, host=None):
    relay = RelayServer.__new__(RelayServer)
    relay._processors = processors or {}
    relay.skill_host = host
    relay._logged_absent_robot = False
    relay.robot_cmd_sender = _Socket()
    return relay


def _controller_msg(action_type, data):
    return Message(data={"action": {"action_type": int(action_type), "data": data}})


def test_relay_applies_a_service_start_for_a_field_it_computes():
    proc = _Proc()
    relay = _relay({("cam", "bbox"): proc})
    msg = _controller_msg(
        ActionType.START_SERVICE, {"module": "cam", "field": "bbox", "params": {"point": [3, 4]}}
    )

    relay._relay_controller_message(msg.serialize())

    assert proc.calls == [("start", {"point": [3, 4]})]
    (sent,) = relay.robot_cmd_sender.sent
    assert sent.data == {"heartbeat": True}, "the robot's watchdog still hears the controller"


def test_relay_forwards_a_service_action_for_a_robot_field():
    relay = _relay({("cam", "bbox"): _Proc()})
    msg = _controller_msg(ActionType.STOP_SERVICE, {"module": "cam", "field": "depth"})

    relay._relay_controller_message(msg.serialize())

    (sent,) = relay.robot_cmd_sender.sent
    assert sent.data["action"]["action_type"] == int(ActionType.STOP_SERVICE)


def test_relay_hosts_skills_placed_on_the_server_only():
    host = _Host(running={"umi_pnp"})
    relay = _relay(host=host)

    for place in ("server", "robot"):
        relay._relay_controller_message(
            _controller_msg(
                ActionType.START_SKILL, {"skill": "umi_pnp", "execution_place": place}
            ).serialize()
        )
    relay._relay_controller_message(
        _controller_msg(ActionType.STOP_SKILL, {"skill": "umi_pnp"}).serialize()
    )
    relay._relay_controller_message(
        _controller_msg(ActionType.STOP_SKILL, {"skill": "umi"}).serialize()
    )

    assert [d["execution_place"] for d in host.started] == ["server"]
    assert host.stopped == ["umi_pnp"]
    forwarded = [m.data.get("action", {}).get("data", {}).get("skill") for m in relay.robot_cmd_sender.sent]
    assert forwarded == [None, "umi_pnp", None, "umi"]


class _ServerSkill(Skill):
    config_name = "server_wave"

    def start(self):
        self.stop_requested = False

    def is_alive(self):
        return not self.stop_requested

    def request_stop(self):
        self.stop_requested = True

    def control_step(self, dt):
        return None


def _server_skill_host():
    host = ServerSkillHost.__new__(ServerSkillHost)
    host.observation_buffers, host.action_queues, host.skills_status_buffer = {}, {}, None
    host._local_skills, host._stopping, host._deferred_starts = {}, {}, {}
    return host


def test_the_server_starts_a_skill_by_path_with_its_run_id():
    host = _server_skill_host()

    host.start_named_skill(
        {"skill": "server_wave", "path": f"{__name__}:_ServerSkill", "run_id": 9}
    )

    skill = host._local_skills["server_wave"]
    assert skill.skill_id == 9 and skill.execution_place is ExecutionPlace.SERVER


def test_the_server_ignores_a_retried_stop_for_another_run():
    host = _server_skill_host()
    host.start_named_skill(
        {"skill": "server_wave", "path": f"{__name__}:_ServerSkill", "run_id": 2}
    )

    host.stop_named_skill({"skill": "server_wave", "run_id": 1})
    assert "server_wave" in host._local_skills

    host.stop_named_skill({"skill": "server_wave", "run_id": 2})
    assert "server_wave" not in host._local_skills


def test_robot_reads_every_queued_message_and_batches_in_order():
    robot = Robot.__new__(Robot)
    robot._action_latency = LatencyEma()
    robot._last_command_ns = None
    move = {"action_type": int(ActionType.MODULE_ACTION), "data": {"arm": {"n": 0}}}
    batch = Message(
        data={"actions": [{**move, "data": {"arm": {"n": i}}} for i in (1, 2)]}
    )
    robot.action_socket = _Socket([Message(data={"action": move}), batch])

    actions = robot.get_actions()

    assert [a.data["arm"]["n"] for a in actions] == [0, 1, 2]
    assert robot.get_actions() == []


def test_a_relay_batch_alone_does_not_feed_the_watchdog():
    robot = Robot.__new__(Robot)
    robot._action_latency = LatencyEma()
    robot._last_command_ns = None
    robot.action_socket = _Socket([Message(data={"actions": []})])
    robot.get_actions()
    assert robot._last_command_ns is None


def test_calculated_field_reads_its_execution_place_from_the_conf(tmp_path):
    conf = tmp_path / "tracker.yaml"
    conf.write_text("execution_place: robot\n", encoding="utf-8")
    field = CalculatedField(
        inputs=[], processor=ServiceProcessor, output=BoundingBoxData(), conf=str(conf)
    )
    assert field.execution_place is ExecutionPlace.ROBOT

    conf.write_text("server_url: x\n", encoding="utf-8")
    missing = CalculatedField(
        inputs=[], processor=ServiceProcessor, output=BoundingBoxData(), conf=str(conf)
    )
    with pytest.raises(ValueError, match="execution_place"):
        missing.execution_place


def test_bounding_box_is_an_int32_array():
    box = BoundingBoxData().duplicate({"value": [1, 2, 3, 4]})
    assert box.value.dtype == np.int32
    assert box.value.tolist() == [1, 2, 3, 4]
    assert BoundingBoxData().value.tolist() == [0, 0, 0, 0]


def test_a_point_is_drawn_as_a_square_centred_on_it():
    point = PointData(box_px=10).duplicate({"value": [100.4, 50.0, 87.0]})
    assert point.value.dtype == np.float32
    assert point.overlay_box().tolist() == [95, 45, 105, 55]
    assert point.overlay_text() == "point 87%"
    assert PointData().overlay_box().tolist() == [0, 0, 0, 0]
    assert PointData().overlay_text() is None


class _Echo(ServiceProcessor):
    """Returns ``[n, n, n, n]`` for the n-th frame since the start."""

    def __init__(self):
        super().__init__(inputs=[], output=BoundingBoxData())
        self._input_field_names = ["image"]
        self.n = 0
        self.stopped = threading.Event()

    def on_start(self, frame, params):
        self.n = int(params["seed"])
        return np.full(4, self.n)

    def on_frame(self, frame):
        self.n += 1
        return np.full(4, self.n)

    def on_stop(self):
        self.stopped.set()


def _frame(ts):
    image = ImageData(value=np.zeros((4, 4, 3), dtype=np.uint8))
    image.timestamp = ts
    return {"image": image}


def _wait_for(proc, ts, deadline_s=2.0):
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        out = proc(_frame(ts))
        if out.value.any():
            return out
        time.sleep(0.01)
    raise AssertionError("service never produced a result")


def test_service_processor_is_zeros_until_started_then_stamps_the_source_frame():
    proc = _Echo()
    assert proc(_frame(1.0)).value.tolist() == [0, 0, 0, 0]

    proc.start({"seed": 7})
    out = _wait_for(proc, 2.0)
    assert out.value.tolist()[0] >= 7
    assert out.timestamp == 2.0, "a result carries the time of the frame it came from"

    proc.stop()
    assert proc.stopped.wait(2.0)
    assert proc(_frame(3.0)).value.tolist() == [0, 0, 0, 0]


def test_a_message_without_the_image_keeps_the_last_result():
    """The robot's preview throttle drops the image but still sends the module."""
    proc = _Echo()
    proc.start({"seed": 7})
    _wait_for(proc, 2.0)
    assert proc({}).value.tolist()[0] >= 7


def test_log_panel_keeps_the_exception_of_a_logged_traceback():
    try:
        {}["image"]
    except KeyError:
        record = logging.LogRecord("relay", logging.ERROR, __file__, 1, "Error in state relay", (), None)
        record.exc_info = sys.exc_info()
    message = LogRecord.from_logging_record(record).message
    assert message.startswith("Error in state relay: KeyError: 'image' (test_services.py:")


def test_display_click_maps_back_to_capture_pixels():
    from solarian.core.controller_gui import display_to_image

    # A 640x480 frame drawn at half size, offset (10, 20).
    rect = (10, 20, 320, 240)
    assert display_to_image((10, 20), rect, (640, 480)) == (0, 0)
    assert display_to_image((170, 140), rect, (640, 480)) == (320, 240)
    assert display_to_image((329, 259), rect, (640, 480)) == (638, 478)
    assert display_to_image((5, 20), rect, (640, 480)) is None


def test_preview_rate_is_configurable():
    from solarian.core.configs import RobotConfig

    assert RobotConfig.model_fields["preview_image_hz"].default == 30.0
