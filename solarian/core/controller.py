import os
import queue
import threading
import time
import zmq
import math
import numpy as np
from abc import ABC
from dataclasses import dataclass, field
from collections.abc import Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any
from multiprocessing.managers import SharedMemoryManager

from solarian.utils import busy_wait
from solarian.core import curve
from solarian.core import Action
from solarian.core.action import ActionType
from solarian.core import Observation
from solarian.core.conf_paths import project_path
from solarian.core.configs import ControllerConfig
from solarian.core.message import Message
from solarian.core.module import Module
from solarian.core.remote_input_module import RemoteInputModule
from solarian.core.clock_sync import (
    sync_with_server,
    get_offset,
    synced_time_ns,
    synced_time,
    latency_ms_from_stamp_ns,
    LatencyEma,
    format_latency_status,
)
from solarian.core.event import (
    Event,
    EventHost,
    EventType,
    event_handler,
    event_inbox_for,
)
from solarian.core.execution import ExecutionMode, ExecutionPlace
from solarian.core.keyboard_module import KeyboardModule
from solarian.core.mouse_module import MouseModule
from solarian.core.ui import build_ui
from solarian.core.ui.base import (
    ControllerUI,
    UICommand,
    UICommandType,
    UIImage,
    UISnapshot,
    recording_status,
)
from solarian.core.skill import Skill, SkillFailed, SkillStatus, Tool, failure_reason
from solarian.utils import profile, manual_profile, get_class, StatusBoard
from solarian.conf import settings
from solarian.core.data import ImageData, BoundingBoxData, MaskData, PointData
from solarian.perception.processor import field_name_for_input
from solarian.utils.shared_memory.observation_buffer import ObservationBuffer
from solarian.utils.shared_memory.action_queue import ModuleActionQueue
from solarian.utils.shared_memory.skill_status_buffer import SkillStatusBuffer
from solarian.utils.configs import load_config_yaml

if TYPE_CHECKING:
    from solarian.core.recording import ControllerRecorder
    from solarian.core.recording.episode_log import EpisodeLog

import logging
logger = logging.getLogger(__name__)

# Seconds between re-requests of a remote ``STOP_SKILL``, and how long to keep
# re-requesting before assuming the skill is gone. Long enough that a robot
# busy finishing an inference is not spammed, short enough that the operator
# does not wait on it.
_REMOTE_STOP_RETRY_S = 0.2
_REMOTE_STOP_TIMEOUT_S = 3.0

# How long to keep re-requesting an episode boundary before giving up. Resent
# every tick rather than on a slower timer: teleop is unusable while it is
# pending, and an episode with a late boundary is worth less than one tick.
_RECORD_REQUEST_TIMEOUT_S = 3.0


@dataclass
class SkillRecord:
    """A skill the controller believes is running, local or remote."""

    name: str
    execution_place: ExecutionPlace
    execution_mode: ExecutionMode = ExecutionMode.PROCESS
    status: SkillStatus = SkillStatus.RUNNING
    skill_id: int | None = None
    modules: set[str] = field(default_factory=set)


def _skill_spec(cls) -> dict:
    """Registry entry for a skill, read off the class without constructing it.

    A declared skill class may be an expensive one that only ever runs on the
    robot, so nothing here instantiates it.
    """
    skill_id = cls.__dict__.get("skill_id")
    return {
        "cls": cls,
        "execution_place": ExecutionPlace(cls.execution_place),
        "execution_mode": ExecutionMode(cls.execution_mode),
        "skill_id": int(skill_id) if isinstance(skill_id, int) and skill_id else None,
        "docstring": (cls.__doc__ or "").strip(),
    }


def _deep_merge_module_actions(base: dict, overlay: dict) -> dict:
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            nested = dict(base[key])
            if "data" in nested and "data" in value and isinstance(nested["data"], dict) and isinstance(value["data"], dict):
                merged_data = dict(nested["data"])
                merged_data.update(value["data"])
                nested.update(value)
                nested["data"] = merged_data
                base[key] = nested
            else:
                _deep_merge_module_actions(nested, value)
                base[key] = nested
        else:
            base[key] = value
    return base


class Controller(EventHost, ABC):

    # Every controller has a keyboard: it is how an operator stops everything.
    # Inline, so its events land on the control thread rather than on the OS
    # listener's.
    keyboard = KeyboardModule()
    # Clicks on displayed camera images, reported by whichever front end runs.
    mouse = MouseModule()

    # Skill classes this controller can run. Only a registry: it is what the
    # think process picks from and what a ``START_SKILL`` name resolves to,
    # not a set of ready-made instances — a key handler constructs the skill
    # it wants with the arguments it wants.
    skills: tuple[type[Skill], ...] = ()

    #: What the agent may call, by tool name: a skill class and the params this
    #: controller binds (module names, which policy config). The agent fills in
    #: the rest from the skill's :meth:`~Skill.tool_spec`. One skill can back
    #: several tools — one policy skill, a pick config and a place config.
    tools: dict[str, Tool] = {}

    # Deprecated: which front end runs is deployment configuration, not a
    # property of the controller class, so it belongs in ``conf/controller.yaml``
    # as ``ui:``. Still honoured — ``use_gui = False`` selects ``ui=opencv`` —
    # for subclasses that set it before ``ui`` existed.
    use_gui = True

    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.context = zmq.Context()

        try:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        except ImportError:
            self.device = 'cpu'
        
        self._image_field_keys: list[tuple[str, str]] | None = None
        #: ``(module, image field)`` -> the box and mask fields computed from it.
        self._overlay_field_map: dict[tuple[str, str], list[str]] | None = None
        #: The front end, built by :meth:`start`. A controller that is only
        #: connected to (to read sensors, say) never has one.
        self.ui: ControllerUI | None = None
        
        # action socket
        self.action_socket = self.context.socket(zmq.PUSH)
        self.action_socket.setsockopt(zmq.CONFLATE, 1)
        self.action_socket.setsockopt(zmq.LINGER, 0)
        curve.apply_client(self.action_socket, self.cfg.connection, "controller actions")
        self.action_socket.connect(f"tcp://{self.cfg.connection.ip}:{self.cfg.connection.action_receiver_port}")
        
        # Image subscription
        # Not CONFLATE: each message carries only the modules that advanced,
        # so keeping only the newest would drop frames behind joint updates.
        self.state_socket = self.context.socket(zmq.SUB)
        self.state_socket.setsockopt(zmq.RCVHWM, 8)
        self.state_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.state_socket.setsockopt(zmq.LINGER, 0)
        curve.apply_client(self.state_socket, self.cfg.connection, "controller state")
        self.state_socket.connect(f"tcp://{self.cfg.connection.ip}:{self.cfg.connection.state_sender_port}")

        # Verify connection
        # self._verify_connection()
        
        # self.last_observation = None
        self.last_action = None
        self.obs_none_count = 0
        # self.last_msg = None
        self.observation_lock = threading.Lock()  # Thread-safe image access
        self.running = False
        self.connected = False
        self._thread_error = None
        self._obs_latency = LatencyEma()
        self._obs_age_latency = LatencyEma()
        #: Pre-formatted telemetry rows, mirrored into every UI snapshot so a
        #: front end that cannot read the status board still gets them.
        self.telemetry: dict[str, str] = {}
        #: When the last state message arrived, for the robot-link age row.
        self._last_state_ns: int | None = None
        #: When this controller last put anything on the action socket, so the
        #: heartbeat only fills real silence.
        self._last_send_ns = time.monotonic_ns()

        self.modules = {}

        # Create modules from class-level attributes. Walks the MRO so that
        # modules declared on a base controller -- the keyboard, for one --
        # exist on every subclass, matching how Robot collects modules.
        remote_input = set(self.cfg.remote_input_modules)
        for klass in type(self).__mro__:
            for name, val in klass.__dict__.items():
                if name in self.modules or not isinstance(val, Module):
                    continue
                # A device the UI samples keeps its name and loses its driver:
                # the consumers read it by name, so swapping the implementation
                # here is the whole of what "the leader arm is on the operator's
                # laptop" means to the rest of the controller.
                module_class = (
                    RemoteInputModule if name in remote_input else val.__class__
                )
                module_cfg = self.load_module_config(name, module_class)
                if module_class is val.__class__:
                    self.modules[name] = val.bind(module_cfg, name)
                else:
                    self.modules[name] = module_class(module_cfg, name)
                setattr(self, name, self.modules[name])
                remote_input.discard(name)
        if remote_input:
            raise ValueError(
                f"controller.remote_input_modules names {sorted(remote_input)}, "
                f"which {type(self).__name__} does not declare; it has "
                f"{sorted(self.modules)}"
            )

        # Create events dictionary directly instead of using init_keyboard_listener
        self.events = {
            "exit_early": False,
            "rerecord_episode": False,
            "track all": False,
            "verbose": False,
            "prompt": None,
            "is_prompting": False,
            "reset_policy": False
        }

        self.last_observation = Observation({})

        self._init_recording()

        # Shared-memory infrastructure mirroring what Robot owns on its side.
        # The controller keeps a per-module ObservationBuffer so downstream
        # consumers (e.g. the agent and skill processes) can read the
        # latest observation without draining the buffer; the buffer is a
        # ring and simply overwrites itself on each new ``put``.
        # ``skill_status_buffer`` collects status transitions the robot
        # forwards inside its state messages.
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        self.skill_status_buffer = SkillStatusBuffer(
            shm_manager=self.shm_manager,
            get_max_k=64,
            get_time_budget=0.2,
            put_desired_frequency=60,
        )

        # Build observation buffers eagerly — same pattern as Robot.__init__.
        # We import the Robot class via ``settings.ROBOT_CLASS`` and
        # instantiate its modules (schema-only, no hardware connections) so
        # that class-level data descriptors (e.g. ``OpenCVCamera.rgb_image.shape``)
        # are populated before we call ``Observation.zeros()``.
        self.observation_buffers: dict[str, ObservationBuffer] = {}
        self._setup_remote_observation_schema()
        self._create_observation_buffers()
        self._create_local_observation_buffers()

        self._init_skill_orchestration()
        from solarian.utils.shared_memory.action_queue import ActionQueue
        #: Control-plane inbox: START_SKILL / STOP_SKILL from the legacy think
        #: process. Skills command modules through their own queues, not here.
        self.skill_command_queue = ActionQueue(
            shm_manager=self.shm_manager,
            buffer_size=256,
            max_payload_bytes=8192,
        )
        self.agent = self._build_agent()

    def _build_agent(self):
        """The supervising agent, when ``agent:`` is configured; started by :meth:`start`."""
        if self.cfg.agent is None:
            return None
        from solarian.core.ai.agent import AgentHarness

        if not self.tools:
            logger.warning("agent is configured but %s declares no tools", type(self).__name__)
        return AgentHarness(host=self, cfg=self.cfg.agent, tools=self.tools)

    def load_module_config(self, module_name: str, module_class: type):
        """Load module config from YAML using the module's own config class.

        Modules with nothing to describe -- an input device has no ports,
        calibration, or limits -- declare a ``default_config`` and are used
        without the project shipping an empty YAML file for them.
        """
        from solarian.core.conf_paths import module_config_path

        yaml_path = str(module_config_path(module_name))
        default_config = getattr(module_class, "default_config", None)
        if default_config is not None and not os.path.exists(yaml_path):
            return module_class.config_class(**default_config)
        module_cfg = load_config_yaml(yaml_path, module_class.config_class)
        return module_cfg

    def _setup_remote_observation_schema(self) -> None:
        """Instantiate the Robot's modules schema-only so that class-level
        data descriptors are populated with correct shapes.

        The Controller does not own the remote Robot's modules, but it needs
        to know their observation schemas to create matching shared-memory
        buffers.  Instantiating a module with its YAML config populates
        class-level descriptors such as ``OpenCVCamera.rgb_image.shape`` —
        which is all that ``Observation.zeros()`` needs downstream.

        No hardware is opened here; ``Module.connect()`` is never called.
        """
        try:
            robot_cls = get_class(settings.ROBOT_CLASS)
        except Exception as e:
            logger.warning(f"Could not resolve ROBOT_CLASS: {e}")
            return

        for name, attr in list(robot_cls.__dict__.items()):
            if not isinstance(attr, Module):
                continue
            if name in self.modules:
                continue
            module_class = type(attr)
            try:
                module_cfg = self.load_module_config(name, module_class)
            except FileNotFoundError:
                logger.warning(
                    f"No config file for remote module '{name}', "
                    f"observation schema for it will be inferred from "
                    f"the first incoming observation"
                )
                continue
            # Schema-only: instantiate for its side-effect of mutating
            # class-level data descriptors.  The instance itself is
            # discarded.
            attr.bind(module_cfg, name)

    def _create_observation_buffers(self) -> None:
        """Create a shared-memory ``ObservationBuffer`` for every module
        declared in the active ``Observation`` subclass."""
        example_observation = Observation.zeros()
        for name, module_obs in example_observation.items():
            if name in self.observation_buffers:
                continue
            try:
                self.observation_buffers[name] = ObservationBuffer.create_from_example(
                    shm_manager=self.shm_manager,
                    example_observation=module_obs,
                    get_time_budget=0.2,
                    put_desired_frequency=120,
                )
            except Exception as e:
                logger.warning(
                    f"Could not create observation buffer for '{name}': {e}"
                )

    def _create_local_observation_buffers(self) -> None:
        """Give each controller-side module a buffer and register it.

        Local rather than shared-memory: these modules run on this process's
        own loop, so there is no fork to cross, and no reason to require them
        to appear in the robot's ``Observation`` schema or to flatten what
        they report. Registering them in the same dict is what lets a skill
        read any module by name without knowing which side it sits on.
        """
        for name, module in self.modules.items():
            if name in self.observation_buffers:
                logger.debug(
                    f"Controller module '{name}' shadows a robot module of "
                    f"the same name in the observation schema"
                )
            buffer = ObservationBuffer.local()
            self.observation_buffers[name] = buffer
            module.observation_buffer = buffer

    def _verify_connection(self, timeout=5000):
        """Verify that we can connect to the relay server"""
        logger.info(f"Verifying connection to relay server at {self.cfg.connection.ip}...")
        
        # Create a temporary PUSH socket to test connection
        test_socket = self.context.socket(zmq.PUSH)
        test_socket.setsockopt(zmq.LINGER, 0)
        print(f"Attempting to connect to tcp://{self.cfg.connection.ip}:{self.cfg.connection.action_receiver_port}...")
        test_socket.connect(f"tcp://{self.cfg.connection.ip}:{self.cfg.connection.action_receiver_port}")
        
        try:
            # Try to send a message
            logger.info("Attempting to send test message...")
            test_socket.send_string("test", zmq.NOBLOCK)
            logger.info("Test message sent successfully")
            time.sleep(1)  # Give it a moment to ensure the message was sent
            return True
        except zmq.ZMQError as e:
            logger.error(f"ZMQ error occurred: {e}")
            raise ConnectionError(f"ZMQ error while connecting to relay server: {e}")
        finally:
            logger.info("Cleaning up test socket...")
            test_socket.close()
        
        logger.info("Successfully connected to relay server")

    def _publish_telemetry(self, label: str, value) -> None:
        """Put one telemetry row on the status board and in the UI snapshot.

        Two consumers with different reach: the board is process-wide and always
        on, the snapshot is per-frame and may cross a link to a browser that
        cannot see this process's terminal at all.
        """
        StatusBoard.set(label, value)
        self.telemetry[label] = str(value)

    @profile
    def get_state(self):
        bytes = self.state_socket.recv(zmq.NOBLOCK)
        msg = Message.deserialize(bytes)
        self._last_state_ns = time.time_ns()

        latency_ms = latency_ms_from_stamp_ns(msg.robot_timestamp)
        if latency_ms is not None:
            self._publish_telemetry(
                "Obs latency (ms)",
                format_latency_status(self._obs_latency, latency_ms),
            )

        if "observation" in msg.data:
            raw_obs = msg.data["observation"]
            observation = Observation(raw_obs)

            age_ms = self._observation_age_ms(observation)
            if age_ms is not None:
                self._publish_telemetry(
                    "Obs age (ms)",
                    format_latency_status(self._obs_age_latency, age_ms),
                )

            for module_name, module_observation in observation.items():
                self.last_observation[module_name] = module_observation
                self._put_module_observation(module_name, module_observation)

            StatusBoard.set("Observation", self.last_observation)

            recorder = self.controller_recorder
            if recorder is not None:
                recorder.feed(observation)

        if "recording" in msg.data:
            self.robot_recording = msg.data["recording"]

        if "skill_status" in msg.data:
            recs = msg.data["skill_status"]
            self._ingest_skill_status_records(recs)
            for rec in recs:
                try:
                    self.skill_status_buffer.put(
                        skill_id=int(rec["skill_id"]),
                        status=int(rec["status"]),
                        timestamp=float(rec["timestamp"]),
                        wait=False,
                    )
                except Exception as e:
                    logger.warning(f"skill_status_buffer.put failed: {e}")

    def _observation_age_ms(self, observation: Observation) -> float | None:
        """Age of the newest sensor timestamp in the observation (ms)."""
        newest = None
        for _module_name, module_obs in observation.items():
            items = getattr(module_obs, "items", None)
            if not callable(items):
                continue
            for _fname, field in items():
                ts = getattr(field, "timestamp", None)
                if ts is None:
                    continue
                newest = float(ts) if newest is None else max(newest, float(ts))
        if newest is None:
            return None
        # Sensor stamps use robot local wall time; after clock sync both sides
        # share the server time base via offset, so this is a useful combined
        # capture→controller age signal (includes WiFi + compress + queueing).
        return (synced_time() - newest) * 1000.0

    @profile
    def _send_action(self, action: Action):
        """Send control actions to robot"""
        msg = Message()
        msg.controller_timestamp = synced_time_ns()
        msg.data = {"action": action.to_dict()}
        self._send_message(msg)
        recorder = self.controller_recorder
        if recorder is not None:
            recorder.note_sent(action)

    def _send_message(self, msg: Message) -> None:
        """Put *msg* on the action socket and note that the link was just used.

        Every send goes through here so the heartbeat below can tell real
        silence from the gap between two messages. This socket is CONFLATE: a
        heartbeat sent on top of a command not yet transmitted would replace
        it.
        """
        self.action_socket.send(msg.serialize())
        self._last_send_ns = time.monotonic_ns()

    def _send_heartbeat_if_idle(self) -> None:
        """Tell the robot the link is alive while there is nothing to send.

        The robot's watchdog cannot tell an idle controller from a dead one, and
        every other message on this socket is event-driven, so the silence has
        to be filled from here.
        """
        heartbeat_s = self.cfg.connection.heartbeat_s
        if (time.monotonic_ns() - self._last_send_ns) / 1e9 < heartbeat_s:
            return
        msg = Message()
        msg.controller_timestamp = synced_time_ns()
        msg.data = {"heartbeat": True}
        self._send_message(msg)

    def _put_module_observation(self, module_name: str, module_observation) -> None:
        """Push *module_observation* into its ``ObservationBuffer``.

        The buffer is a ring and overwrites itself, so reads by the think
        process or skills never drain it — they all see the latest entry.
        """
        buffer = self.observation_buffers.get(module_name)
        if buffer is None:
            logger.debug(
                f"No observation buffer for module '{module_name}' "
                f"(not declared in Observation schema); dropping."
            )
            return

        try:
            buffer.put(module_observation, wait=False)
        except Exception as e:
            logger.warning(
                f"observation_buffers['{module_name}'].put failed: {e}"
            )

    def _state_receiver(self):
        """Receive and decode image streams"""
        try:
            while self.running:
                try:
                    self.get_state() 
                except zmq.Again:
                    time.sleep(0.001)
        except Exception as e:
            if self._thread_error is None:
                self._thread_error = e
        finally:
            self.running = False

    def connect(self):
        """Start receiver threads"""
        self.sync_clock()

        for module_name, module in self.modules.items():
            if not module.needs_motor_calibration():
                # Input devices and other calibration-free modules.
                continue

            module.load_calibration()

            module.connect()

            if module.calibration is None:
                module.calibrate()

            calibration = module.require_calibration()
            if not module.is_calibrated():
                module.write_calibration()

            module.disconnect()
            module.connect(calibration)

        self.connected = True
        logger.info("Controller connected")

    def sync_clock(self) -> None:
        """Estimate wall-clock offset vs the relay server (soft-fail)."""
        try:
            result = sync_with_server(
                self.cfg.connection.ip,
                self.cfg.connection.clock_sync_port,
                context=self.context,
                connection=self.cfg.connection,
            )
            self._publish_telemetry("Clock offset (ms)", result.offset_s * 1000.0)
            self._publish_telemetry("Clock sync RTT (ms)", result.rtt_s * 1000.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Clock sync failed (%s); using local wall clock (offset=%.1f ms)",
                exc,
                get_offset() * 1000.0,
            )

    def is_connected(self):
        return self.connected
        
    def initialize(self):
        if not self.is_connected():
            raise Exception("Controller not connected, cannot initialize, must call connect() first.")
        # initialize modules
        logger.info("Initializing modules")
        for module_name, module in self.modules.items():
            module.initialize()
            # A controller-side module publishes into a local ObservationBuffer,
            # which a fork does not carry, so in PROCESS mode every reader on
            # this side would poll an empty buffer forever.
            if module.execution_mode is ExecutionMode.PROCESS:
                raise ValueError(
                    f"Controller module '{module_name}' has execution_mode "
                    f"'process'; a controller-side module must be 'inline' or "
                    f"'thread'. Set execution_mode on the class, or "
                    f"'execution:' in conf/{module_name}.yaml."
                )
            # Started here rather than at the first tick so a module can emit
            # before the control loop asks it for anything.
            module.start()
        logger.info("Controller initialized")

    def disconnect(self):
        """Clean shutdown"""
        self._close_controller_recorder()
        if not self.is_connected():
            return
        self.running = False
        self.action_socket.close()
        self.state_socket.close()
        self.context.term()
        self.connected = False

    def stop(self):
        self.running = False

    def _build_display_schema(self):
        """Build and cache the image fields and the box/mask fields drawn over each."""
        obs_spec = Observation.get_module_obs_spec()
        image_keys: list[tuple[str, str]] = []
        overlay_map: dict[tuple[str, str], list[str]] = {}

        for module_name, module_obs_instance in obs_spec.items():
            module_obs_cls = type(module_obs_instance)
            sensor_fields = module_obs_cls.get_sensor_fields_spec()
            calculated_fields = module_obs_cls.get_calculated_fields_spec()

            all_fields = {**sensor_fields, **calculated_fields}
            for field_name, field_spec in all_fields.items():
                if field_spec.data_class is ImageData:
                    image_keys.append((module_name, field_name))

            for field_name, cf in calculated_fields.items():
                if not issubclass(cf.data_class, (BoundingBoxData, PointData, MaskData)):
                    continue
                for inp in cf.inputs:
                    input_field = field_name_for_input(
                        inp, sensor_fields, calculated_fields
                    )
                    if input_field:
                        overlay_map.setdefault((module_name, input_field), []).append(
                            field_name
                        )
                        break

        self._image_field_keys = image_keys
        self._overlay_field_map = overlay_map

    # -- UI snapshot -------------------------------------------------------
    @profile
    def _collect_display_images(self) -> list[UIImage]:
        """Every image field in the latest observation, with its boxes and masks.

        One collection for every front end: a window grid, a browser and a JPEG
        encoder all want the same pixels, and the calculated boxes and masks
        belong to the frame they were computed from rather than to whoever draws
        them.
        """
        if self._image_field_keys is None or self._overlay_field_map is None:
            self._build_display_schema()
        assert self._image_field_keys is not None and self._overlay_field_map is not None

        images: list[UIImage] = []
        for module_name, field_name in self._image_field_keys:
            try:
                module_obs = self.last_observation[module_name]
            except (KeyError, TypeError):
                continue
            if module_obs is None:
                continue
            try:
                field_data = module_obs[field_name]
            except KeyError:
                continue
            if field_data is None or field_data.value is None:
                continue

            bboxes = []
            bbox_texts = []
            masks = []
            for overlay_field in self._overlay_field_map.get((module_name, field_name), ()):
                try:
                    overlay = module_obs[overlay_field]
                except KeyError:
                    continue
                if overlay is None or overlay.value is None:
                    continue
                if isinstance(overlay, MaskData):
                    # A mask segmented on another resolution would misalign.
                    if np.shape(overlay.value) == field_data.value.shape[:2]:
                        masks.append(overlay.value)
                else:
                    bboxes.append(overlay.overlay_box())
                    bbox_texts.append(overlay.overlay_text())

            images.append(
                UIImage(
                    label=f"{module_name}/{field_name}",
                    data=field_data.value,
                    bboxes=bboxes,
                    masks=masks,
                    bbox_texts=bbox_texts,
                    jpeg=field_data._jpeg,
                )
            )
        return images

    @staticmethod
    def _gui_format_value(value) -> str:
        if isinstance(value, dict):
            value = value.get("value")
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        if isinstance(value, (int, np.integer, bool)):
            return str(value)
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return f"{float(value):.3f}"
            if value.ndim == 1 and value.size <= 8:
                return "[" + ", ".join(f"{float(x):.2f}" for x in value) + "]"
            return f"<{value.shape}>"
        if value is None:
            return "-"
        return str(value)

    def _gui_field_spec(self, module_name: str, field_name: str):
        """Return the SensorField / CalculatedField spec for an obs field, if any."""
        try:
            module_obs_instance = Observation.get_module_obs_spec()[module_name]
        except (KeyError, TypeError):
            return None
        module_obs_cls = type(module_obs_instance)
        spec = module_obs_cls.get_sensor_fields_spec().get(field_name)
        if spec is None:
            spec = module_obs_cls.get_calculated_fields_spec().get(field_name)
        return spec

    def _gui_dashboard_sections(self) -> list[tuple[str, list[tuple]]]:
        """Build ``(title, [(field, value, color?), ...])`` panels from the latest
        observation and action, skipping image fields."""
        from solarian.core.controller_gui import LEVEL_COLORS
        from solarian.core.observation import SensorField

        sections: list[tuple[str, list[tuple]]] = []

        try:
            obs_items = list(self.last_observation.items())
        except Exception:
            obs_items = []
        for module_name, module_obs in obs_items:
            if module_obs is None:
                continue
            try:
                payload = module_obs.to_dict()
            except Exception:
                continue
            rows: list[tuple] = []
            for field, val in payload.items():
                raw = val.get("value") if isinstance(val, dict) else val
                if isinstance(raw, (bytes, bytearray)):
                    continue
                if isinstance(raw, np.ndarray) and raw.ndim >= 2:
                    continue  # image / large tensor -> shown in the camera grid
                display = self._gui_format_value(val)
                color = None
                spec = self._gui_field_spec(module_name, field)
                if isinstance(spec, SensorField):
                    level = spec.threshold_level(raw)
                    if level is not None:
                        color = LEVEL_COLORS[level]
                row = (field, display, color) if color is not None else (field, display)
                rows.append(row)
            if rows:
                sections.append((f"obs · {module_name}", rows))

        action = self.last_action
        if action is not None:
            try:
                adict = action.to_dict()
                data = adict.get("data", {})
                if isinstance(data, dict):
                    for module_name, payload in data.items():
                        fields = payload.get("data", {}) if isinstance(payload, dict) else {}
                        rows = [
                            (field, self._gui_format_value(val))
                            for field, val in fields.items()
                        ]
                        if rows:
                            sections.append((f"act · {module_name}", rows))
            except Exception:
                pass
        return sections

    def gui_key_help(self) -> list[tuple[str, str]]:
        """Key-binding rows ``(keys, description)`` shown in the GUI keys panel.

        Only the one row the framework can know about, the universal stop.
        Which keys start which skills is application policy living in a
        subclass's ``@event_handler``, so a subclass that binds keys should
        extend this with its own rows.
        """
        return [
            ("space", "stop all skills"),
            ("r", "start / stop recording"),
            ("delete", "discard this episode and re-record"),
        ]

    def ui_snapshot(self) -> UISnapshot:
        """Everything the front end needs to draw one frame.

        Built here and nowhere else, so a browser on the far side of a socket
        sees the same numbers as a local window rather than a thinner view of
        them.
        """
        telemetry = dict(self.telemetry)
        if self._last_state_ns is not None:
            age_ms = (time.time_ns() - self._last_state_ns) / 1e6
            telemetry["Robot link age (ms)"] = f"{age_ms:.0f}"
        running = sorted(self.running_skills)
        images = self._collect_display_images()
        agent = None
        if self.agent is not None:
            agent = self.agent.ui_state()
            view = self.agent.view_image()
            if view is not None:
                images.append(view)
        recording = self.recording_state
        return UISnapshot(
            t_ns=synced_time_ns(),
            fps=self.cfg.fps or None,
            sections=self._gui_dashboard_sections(),
            images=images,
            agent=agent,
            key_help=self.gui_key_help(),
            skills={
                "available": sorted(self._skill_registry),
                "running": running,
            },
            telemetry=telemetry,
            recording=recording,
            # Recording first: a window with the episode number in it is the
            # difference between noticing a session was never armed and
            # finding out when the dataset is empty.
            status=" · ".join(
                (recording_status(recording),
                 ", ".join(running) if running else "idle")
            ),
        )

    def ui_hello(self) -> dict:
        """One-shot description of this controller for a UI built without it.

        A front end on the far side of a socket cannot import the project, so
        what it may start, which keys mean what and what an observation contains
        all have to arrive as data.
        """
        return {
            "controller": type(self).__name__,
            "fps": self.cfg.fps,
            "key_help": [list(row) for row in self.gui_key_help()],
            "skills": [
                {
                    "name": name,
                    "execution_place": spec["execution_place"].value,
                    "execution_mode": spec["execution_mode"].value,
                    "docstring": spec["docstring"],
                }
                for name, spec in sorted(self._skill_registry.items())
            ],
            "observation_schema": self._ui_observation_schema(),
            "agent": None if self.agent is None else {
                "planner": self.agent.planner.model,
                "monitor": self.agent.monitor.model if self.agent.monitor else None,
                "tools": [
                    {"name": name, "description": str(spec["description"]).strip()}
                    for name, spec in self.agent.specs.items()
                ],
            },
        }

    @staticmethod
    def _ui_observation_schema() -> dict[str, dict[str, str]]:
        """``{module: {field: data class}}`` for the declared observation."""
        schema: dict[str, dict[str, str]] = {}
        for module_name, module_obs in Observation.get_module_obs_spec().items():
            module_obs_cls = type(module_obs)
            fields = {
                **module_obs_cls.get_sensor_fields_spec(),
                **module_obs_cls.get_calculated_fields_spec(),
            }
            schema[module_name] = {
                field_name: spec.data_class.__name__
                for field_name, spec in fields.items()
            }
        return schema

    def handle_ui_command(self, command: UICommand) -> bool:
        """Apply one command from the front end; ``False`` means stop the loop.

        Key commands go through the keyboard module rather than to a handler
        directly, so every existing ``@event_handler(EventType.KEY_DOWN)`` fires
        for a browser keypress exactly as it does for a local one, and lands on
        the control thread either way.
        """
        data = command.data
        match command.type:
            case UICommandType.KEY_DOWN:
                self.keyboard.on_press(str(data["key"]).lower())
            case UICommandType.KEY_UP:
                self.keyboard.on_release(str(data["key"]).lower())
            case UICommandType.CLICK:
                # ``label`` is the ``module/field`` every UIImage carries.
                module_name, _, field_name = str(data.get("label") or "").partition("/")
                point = data.get("point")
                if point and field_name:
                    self.mouse.on_click(
                        point, module_name, field_name, int(data.get("button") or 1)
                    )
            case UICommandType.START_SKILL:
                self.start_skill_by_name(
                    str(data.get("skill") or ""), data.get("params") or {}
                )
            case UICommandType.STOP_SKILL:
                name = str(data.get("skill") or "")
                if name:
                    self._operator_took_over(f"stopped {name}")
                    self.stop_skill(name, reason="stopped by the operator")
            case UICommandType.STOP_ALL:
                self.stop_all_skills()
            case UICommandType.RECORD_START:
                self.start_recording(data.get("task"))
            case UICommandType.RECORD_STOP:
                self.stop_recording()
            case UICommandType.RECORD_REDO:
                self.redo_recording()
            case UICommandType.RECORD_TARGETS:
                self.set_recording_targets(
                    controller=data.get("controller"), robot=data.get("robot")
                )
            case UICommandType.INPUT_SAMPLE:
                self._dispatch_input_sample(data)
            case UICommandType.AGENT_MESSAGE:
                if self.agent is not None:
                    self.agent.send_user_message(str(data.get("text") or ""))
            case UICommandType.AGENT_CANCEL:
                if self.agent is not None:
                    self.agent.cancel()
            case UICommandType.AGENT_RESET:
                if self.agent is not None:
                    self.agent.reset()
            case UICommandType.QUIT:
                return False
        return True

    def _dispatch_input_sample(self, data: dict) -> None:
        """Hand a teleop reading from the UI to the module that stands in for it.

        ``accept_sample`` is polymorphic — only a
        :class:`~solarian.core.remote_input_module.RemoteInputModule` has one —
        so this is where a sample sampled in a browser tab becomes an ordinary
        observation, and where a sample addressed to a locally-wired module is
        refused rather than silently dropped.
        """
        module_name = str(data.get("module") or "")
        accept_sample = getattr(self.modules.get(module_name), "accept_sample", None)
        if accept_sample is None:
            logger.warning(
                "Input sample for '%s', which is not a remote-input module here; "
                "this controller has %s",
                module_name,
                sorted(self.modules),
            )
            return
        accept_sample(data.get("fields") or {})

    def control_loop(self):
        try:
            while self.running:
                start_loop_t = time.time_ns()

                # Events and local state are refreshed by the loop, not by
                # control_step, so a controller that overrides control_step
                # still gets its event handlers run and its modules read.
                events = self._drain_events()
                self._tick_local_modules()
                self._fan_out_events(events)
                # Here rather than in control_step, which a subclass may
                # override: the controller's episode must still open.
                self._open_pending_controller_episode()

                self.last_action = self.control_step()

                if self.last_action is not None:
                    StatusBoard.set("Action", self.last_action)
                    self._send_action(self.last_action)
                else:
                    self._send_heartbeat_if_idle()

                # Wait for next frame
                if self.cfg.fps is not None:
                    dt_s = (time.time_ns() - start_loop_t)/1e9
                    busy_wait((1.0 / self.cfg.fps) - dt_s)
                
                manual_profile("control loop latency", (time.time_ns() - start_loop_t))
        except Exception as e:
            if self._thread_error is None:
                self._thread_error = e
        finally:
            self.running = False


    def on_start(self):
        """Hook method called after control threads start but before display loop.
        
        Override this in subclasses to start additional threads or perform
        setup that should happen right before the main display loop begins.
        """
        pass

    # -- Keyboard handling ---------------------------------------------------
    # Key input arrives as KEY_DOWN / KEY_UP events from the keyboard module
    # and is dispatched on the control thread. Whether those events come from
    # a global OS listener or from the GUI window is the controller's call,
    # made here; the module does not decide for itself.

    def _start_keyboard_listener(self) -> None:
        """Listen for global OS key events (no window of our own)."""
        keyboard = self.modules.get("keyboard")
        if keyboard is not None:
            keyboard.start_listener()

    def _stop_keyboard_listener(self) -> None:
        keyboard = self.modules.get("keyboard")
        if keyboard is not None:
            keyboard.stop_listener()

    def _collect_controller_skills(self) -> dict:
        """Registry of this controller's ``skills`` plus plugin entry points.

        Keyed by :meth:`Skill.registry_name`, the same identity a
        ``START_SKILL`` action carries, so a name that reaches the controller
        from the think process resolves to the same class the robot would
        resolve it to.
        """
        from solarian.core.skill_registry import discover_available_skills

        skills: dict = {}
        for cls in (*type(self).skills, *(tool.skill for tool in type(self).tools.values())):
            skills[cls.registry_name()] = _skill_spec(cls)
        try:
            discovered = discover_available_skills()
        except Exception:
            logger.exception("Skill plugin discovery failed")
            discovered = {}
        for name, spec in discovered.items():
            if name in skills:
                if not skills[name].get("skill_id"):
                    skills[name]["skill_id"] = spec.get("skill_id")
            else:
                skills[name] = spec
        return skills

    # ========= observations ============
    def _tick_local_modules(self) -> None:
        """Advance every inline controller-side module once per tick.

        An inline module has no loop of its own, so the host's tick *is* its
        tick: this is where its ``update_observations`` runs and its buffer
        gains a sample. One read per module per tick, shared by every skill —
        two skills both following the leader arm must not turn into two
        blocking serial reads, nor see the arm at two different instants.
        """
        for name, module in self.modules.items():
            if module.execution_mode is ExecutionMode.INLINE:
                # The runner logs and retires a module that raises, so one
                # unplugged device cannot take the control loop down with it.
                module.runner.tick()

            # ``get_state`` puts the robot's observation on the board; a
            # controller-side module publishes only into its own buffer, so
            # its sample is surfaced here or not at all.
            latest = self.observation_buffers[name].latest()
            if not isinstance(latest, dict):
                continue
            # A local buffer holds whatever its module put there: actuator
            # reads are ``{field: {value, timestamp}}``, a keyboard's held-key
            # set is not, and only the former has numeric rows to show.
            values = {
                field: reading["value"]
                for field, reading in latest.items()
                if isinstance(reading, dict) and "value" in reading
            }
            if values:
                StatusBoard.set(name, values)

    @property
    def observations(self) -> dict:
        """Every module's observation buffer, keyed by module name.

        The robot's modules and the controller's own are one mapping rather
        than two, because a skill should not have to know which side of the
        link a module sits on to read it — and module names are unique across
        both, so there is nothing to disambiguate.

        Buffers rather than values: a skill that wants the latest sample
        calls ``.latest()``, one that needs history calls ``.last_k(n)``, and
        one that cares about neither pays for neither.
        """
        return self.observation_buffers

    # ========= skill orchestration ============
    def _init_skill_orchestration(self) -> None:
        self._local_skills: dict[str, Skill] = {}
        self._pending_actions: list[Action] = []
        self._remote_running: dict[str, SkillRecord] = {}
        #: Remote skills asked to stop but not yet confirmed stopped, as
        #: ``name -> (first_request_t, last_request_t)`` in monotonic seconds.
        self._remote_stopping: dict[str, tuple[float, float]] = {}
        self._skill_registry: dict[str, dict[str, Any]] = (
            self._collect_controller_skills()
        )
        #: Each start of a skill is one invocation; this maps a running skill's
        #: name to its invocation, so whoever started it can tell its end from
        #: the end of an earlier run of the same skill.
        self._invocations: dict[str, int] = {}
        self._invocation_seq = 0
        #: Why each skill being stopped is stopping, reported when it ends.
        self._stop_reasons: dict[str, str] = {}
        #: ``(fn, future)`` to run on the control thread; see :meth:`call_on_control_thread`.
        self._control_calls: queue.SimpleQueue = queue.SimpleQueue()
        #: True while running a call the agent queued, so what it starts and
        #: stops is not mistaken for the operator taking over.
        self._agent_dispatching = False
        self.agent = None
        self._init_events()

    @property
    def held_keys(self) -> set[str]:
        """Keys currently held, as reported by the keyboard module(s).

        This is state, not a stream of events, and it only changes when the
        control loop drains — so a skill reading it mid-tick sees a stable
        snapshot rather than a set an input thread is mutating underneath it.
        """
        held: set[str] = set()
        for module in self.modules.values():
            held |= set(getattr(module, "held_keys", ()))
        return held

    def _module_action_queue(self, skill, module_name: str):
        """Build the queue a skill pushes ``module_name``'s actions into.

        Shared memory only when the skill runs in its own process; otherwise
        a plain in-process ring, since producer and consumer are both on this
        loop and there is no fork for the flattening to serve.

        Returns ``None`` when the module has no action schema to size shared
        memory from, matching how :meth:`_setup_remote_observation_schema`
        skips a module with no config.
        """
        if not skill.execution_mode.crosses_process_boundary:
            return ModuleActionQueue.local()
        try:
            example = Action.zeros()[module_name]
        except (KeyError, TypeError):
            example = None
        if example is None:
            logger.warning(
                "No action schema for module '%s'; skill '%s' cannot command it "
                "from another process",
                module_name,
                skill.skill_name,
            )
            return None
        return ModuleActionQueue(
            shm_manager=self.shm_manager,
            example_action=example,
            buffer_size=1024,
        )

    def _modules_overlap(
        self, a: "Skill | SkillRecord", b: "Skill | SkillRecord"
    ) -> bool:
        def names(x) -> set[str]:
            return x.commanded_module_names() if isinstance(x, Skill) else set(x.modules)

        return bool(names(a) & names(b))

    @property
    def running_skills(self) -> dict[str, SkillRecord]:
        out: dict[str, SkillRecord] = {}
        for name, skill in self._local_skills.items():
            out[name] = SkillRecord(
                name=name,
                execution_place=skill.execution_place,
                execution_mode=skill.execution_mode,
                status=SkillStatus.RUNNING,
                skill_id=int(skill.skill_id) if skill.skill_id else None,
                modules=skill.commanded_module_names(),
            )
        for name, rec in self._remote_running.items():
            if name not in out:
                out[name] = rec
        return out

    @event_handler(EventType.KEY_DOWN)
    def stop_all_on_space(self, event: Event) -> None:
        """Space is the operator's stop-everything, on every controller.

        The only key the framework claims. Which other keys start which skills
        is application policy, so subclasses bind those in their own
        ``@event_handler`` — theirs runs alongside this one, not instead of it.
        """
        if event.data.get("key") in (" ", "space"):
            self.stop_all_skills(reason="operator pressed space")

    # ========= recording ============
    def _init_recording(self) -> None:
        #: Recording state as the robot last reported it. The robot owns its
        #: episodes, so this is only ever echoed back. ``request`` is the id
        #: of the last boundary the robot acted on.
        self.robot_recording: dict = {
            "enabled": False,
            "active": False,
            "episode": 0,
            "request": 0,
            "session": None,
        }

        #: The GUI's two switches: which sides record the next episode.
        cfg = self.cfg.recording
        self.recording_targets = {"controller": bool(cfg.controller), "robot": bool(cfg.robot)}
        #: The targets the open episode started with, so a switch flipped
        #: mid-take still lets stop and drop reach the side that is recording.
        self._episode_targets = dict(self.recording_targets)

        #: The controller's own dataset, built on the first episode it records.
        self.controller_recorder: "ControllerRecorder | None" = None
        #: Controller episode boundaries run here, in order, because they block
        #: on encoders: spawning them on the first start, draining them on a
        #: seal. On the control loop that would be seconds of frozen teleop.
        self._controller_recording_queue: queue.Queue = queue.Queue()
        self._controller_recording_thread: threading.Thread | None = None
        #: With both targets on, the controller's episode waits for the robot
        #: to open its own and then takes the robot's session and index, so the
        #: two recordings line up. ``(request id, deadline, start kwargs)``.
        self._pending_controller_start: tuple[int, float, dict] | None = None

        #: Set when the episode in progress ends, whoever ended it. A recording
        #: schedule waits on this rather than sleeping out its timer, so the
        #: operator pressing delete cuts the wait short instead of being
        #: ignored until the clock runs out.
        self.episode_over = threading.Event()

        #: The episode boundary the robot has not confirmed yet, resent every
        #: tick by :meth:`_retry_record_request`.
        self._record_request: Action | None = None
        self._record_request_id = 0
        self._record_request_deadline = 0.0

    @event_handler(EventType.KEY_DOWN)
    def toggle_recording_on_r(self, event: Event) -> None:
        """``r`` toggles the episode, ``delete`` throws the current one away.

        Both names for delete: the key a laptop labels *delete* is backspace,
        and only a full-size keyboard has the forward one.
        """
        key = event.data.get("key")
        if key == "r":
            if self.recording_state["active"]:
                self.stop_recording()
            else:
                self.start_recording()
        elif key in ("delete", "backspace"):
            self.redo_recording()

    @property
    def recording_state(self) -> dict:
        """Both sides' recording, as UIs and the agent read it.

        ``enabled`` says whether starting an episode would record anything;
        ``active`` whether either side is recording now. ``episode`` and
        ``session`` are the controller's while it records, else the robot's.
        """
        robot = dict(self.robot_recording)
        recorder = self.controller_recorder
        controller = {
            "active": recorder is not None and recorder.is_recording,
            "episode": recorder.episode_index if recorder is not None else 0,
            "session": recorder.session_dir_name if recorder is not None else None,
        }
        targets = dict(self.recording_targets)
        if controller["active"] or (not robot["active"] and targets["controller"]):
            side = controller
        else:
            side = robot
        return {
            "targets": targets,
            "enabled": targets["controller"] or (targets["robot"] and bool(robot["enabled"])),
            "active": bool(robot["active"]) or controller["active"],
            "episode": int(side["episode"]),
            "session": side["session"],
            "robot": robot,
            "controller": controller,
        }

    @property
    def episode_log(self) -> "EpisodeLog | None":
        """The log of the episode the controller records, or last recorded; ``None`` before any."""
        recorder = self.controller_recorder
        return recorder.log if recorder is not None else None

    def set_recording_targets(
        self, controller: bool | None = None, robot: bool | None = None
    ) -> None:
        """Flip the GUI's switches. The episode in progress keeps its own."""
        if controller is not None:
            self.recording_targets["controller"] = bool(controller)
        if robot is not None:
            self.recording_targets["robot"] = bool(robot)

    def start_recording(self, task: str | None = None, session: str | None = None) -> None:
        """Open an episode on whichever sides the switches select.

        The robot's side is a request sent down the ordinary action path, and
        the state shown back to the operator is whatever the robot reports
        having done with it. The controller's side records what reaches it.

        Args:
            task: Stamped on the episode. Defaults to the controller's ``task``.
            session: Name of the session this episode is filed under. ``None``
                leaves each side on whichever session it is already writing.
        """
        self.episode_over.clear()
        targets = dict(self.recording_targets)
        self._episode_targets = targets
        start = {"task": task if task is not None else self.cfg.task, "session": session}
        if targets["robot"]:
            self._request_recording(ActionType.RECORD_START, start)
        if not targets["controller"]:
            return
        if targets["robot"]:
            self._pending_controller_start = (
                self._record_request_id,
                time.monotonic() + _RECORD_REQUEST_TIMEOUT_S,
                start,
            )
        else:
            self._start_controller_episode(**start)

    def stop_recording(self) -> None:
        if self._episode_targets["robot"]:
            self._request_recording(ActionType.RECORD_STOP, {})
        if self._episode_targets["controller"]:
            self._pending_controller_start = None
            self._on_controller_recorder(lambda recorder: recorder.stop())
        self.episode_over.set()

    def redo_recording(self) -> None:
        """Throw the episode in progress away and cut its timer short.

        ``solarian record`` sees ``rerecord_episode`` and starts another
        without advancing its counter, so the next take gets a full-length
        episode rather than the remainder of the one just discarded.
        """
        if self._episode_targets["robot"]:
            self._request_recording(ActionType.RECORD_DROP, {})
        if self._episode_targets["controller"]:
            self._pending_controller_start = None
            self._on_controller_recorder(lambda recorder: recorder.drop())
        self.events["rerecord_episode"] = True
        self.episode_over.set()

    def _start_controller_episode(
        self,
        task: str | None,
        session: str | None,
        dir_name: str | None = None,
        index: int | None = None,
    ) -> None:
        if self.controller_recorder is None:
            # Imported here: it pulls in PyAV and pyarrow, which a controller
            # that never records should not pay for at import.
            from solarian.core.recording import ControllerRecorder

            if self._overlay_field_map is None:
                self._build_display_schema()
            cfg = self.cfg.recording
            self.controller_recorder = ControllerRecorder(
                root=project_path(cfg.root),
                cfg=cfg,
                fps=cfg.fps if cfg.fps is not None else self.cfg.fps,
                task=self.cfg.task,
                session=type(self).__name__,
                shm_manager=self.shm_manager,
                overlays=self._overlay_field_map,
            )
        self._on_controller_recorder(
            lambda recorder: recorder.start(
                self.last_observation,
                session=session,
                task=task,
                dir_name=dir_name,
                index=index,
            )
        )

    def _open_pending_controller_episode(self) -> None:
        """Start the controller's episode once the robot has opened its own.

        The robot echoes a request it has acted on, but sealing the previous
        take can hold the new one back, so an echo alone is not an open
        episode. A robot that cannot record, or never opens one in time,
        leaves the controller recording on its own numbering.
        """
        pending = self._pending_controller_start
        if pending is None:
            return
        request_id, deadline, start = pending
        robot = self.robot_recording
        acknowledged = robot["request"] >= request_id
        if acknowledged and robot["active"]:
            self._pending_controller_start = None
            self._start_controller_episode(
                **start, dir_name=robot["session"], index=int(robot["episode"])
            )
            return
        disabled = acknowledged and not robot["enabled"]
        if not disabled and time.monotonic() < deadline:
            return
        self._pending_controller_start = None
        logger.warning(
            "The robot did not open an episode; the controller records this "
            "one in a session of its own"
        )
        self._start_controller_episode(**start)

    def _on_controller_recorder(self, command: "Callable[[ControllerRecorder], Any]") -> None:
        """Run *command* on the controller recording thread, after earlier ones."""
        if self.controller_recorder is None:
            return
        if self._controller_recording_thread is None:
            self._controller_recording_thread = threading.Thread(
                target=self._run_controller_recording,
                name="ControllerRecording",
                daemon=True,
            )
            self._controller_recording_thread.start()
        self._controller_recording_queue.put(command)

    def _run_controller_recording(self) -> None:
        while True:
            command = self._controller_recording_queue.get()
            if command is None:
                return
            try:
                command(self.controller_recorder)
            except Exception:
                logger.exception("Controller recording failed")

    def _close_controller_recorder(self) -> None:
        """Seal whatever the controller is recording and stop its encoders."""
        if self._controller_recording_thread is None:
            return
        self._on_controller_recorder(lambda recorder: recorder.close())
        self._controller_recording_queue.put(None)
        # The seal waits on every encoder's flush, which is bounded by
        # VideoWriter's own timeout.
        self._controller_recording_thread.join(timeout=60.0)
        self._controller_recording_thread = None

    def _request_recording(self, action_type: int, data: dict) -> None:
        """Queue an episode boundary and keep asking until the robot echoes it.

        An episode boundary is the one action that cannot be dropped: a lost
        RECORD_STOP leaves the next take appended to this one. The action
        socket is CONFLATE and the control loop sends at fps, so a boundary
        put straight on the socket is very likely to be overwritten before the
        robot reads it. Only the last request matters, so a newer one replaces
        an unconfirmed older one rather than queueing behind it.
        """
        self._record_request_id += 1
        self._record_request = Action(
            action_type=action_type,
            data={**data, "request": self._record_request_id},
        )
        self._record_request_deadline = time.monotonic() + _RECORD_REQUEST_TIMEOUT_S

    def _retry_record_request(self) -> Action | None:
        """The pending episode boundary, or ``None`` once the robot confirms it."""
        request = self._record_request
        if request is None:
            return None
        if self.robot_recording["request"] >= self._record_request_id:
            self._record_request = None
            return None
        if time.monotonic() > self._record_request_deadline:
            logger.error(
                "Robot never acknowledged %s after %.1f s; the episode "
                "boundary is lost",
                ActionType.name_of(int(request.action_type)),
                _RECORD_REQUEST_TIMEOUT_S,
            )
            self._record_request = None
            return None
        return request

    def start_skill(self, skill: Skill) -> str:
        """Run a skill the caller built, preempting whatever overlaps it.

        The caller constructs the skill with the arguments it wants, so this
        needs no spec to rebuild one from; all it decides is *where* the skill
        runs. A remote skill is not launched here at all — the robot or server
        owns it, and gets a ``START_SKILL`` naming the class and its params.
        """
        name = skill.skill_name
        self._log_episode_event("skill_started", skill=name, params=skill.params)
        if not self._agent_dispatching:
            self._operator_took_over(f"started {name}")
        preempted = f"preempted by {name}"
        if name in self.running_skills:
            # Starting is not toggling: pressing the same key twice re-runs the
            # skill rather than leaving nothing commanding the modules.
            self.stop_skill(name, reason=preempted)
        for other in list(self._local_skills.values()):
            if self._modules_overlap(skill, other):
                self.stop_skill(other.skill_name, reason=preempted)
        for other_name, rec in list(self.running_skills.items()):
            if other_name in self._local_skills:
                continue
            if rec.execution_place.is_remote and self._modules_overlap(skill, rec):
                self.stop_skill(other_name, reason=preempted)
        self._invocation_seq += 1
        self._invocations[name] = self._invocation_seq
        self._stop_reasons.pop(name, None)
        if skill.execution_place.is_remote:
            self._pending_actions.append(
                Action(
                    action_type=ActionType.START_SKILL,
                    data={
                        "skill": type(skill).registry_name(),
                        "params": skill.params,
                        # The relay starts the ones placed on it and passes
                        # the rest on, so one skill can run on either host
                        # by configuration alone.
                        "execution_place": skill.execution_place.value,
                    },
                )
            )
            self._remote_running[name] = SkillRecord(
                name=name,
                execution_place=skill.execution_place,
                execution_mode=skill.execution_mode,
                status=SkillStatus.RUNNING,
                skill_id=int(skill.skill_id) if skill.skill_id else None,
                modules=skill.commanded_module_names(),
            )
        else:
            # Every locally-run skill reads its inputs off itself and writes
            # its actions the same way, so wire both here whichever way it
            # turns out to be scheduled.
            skill.observations = self.observation_buffers
            skill.event_inbox = event_inbox_for(skill)
            skill.action_queues = {
                module_name: queue
                for module_name, queue in (
                    (n, self._module_action_queue(skill, n))
                    for n in skill.commanded_module_names()
                )
                if queue is not None
            }
            if skill.execution_mode is ExecutionMode.INLINE:
                # No runner: the controller's own loop steps this skill.
                skill.begin()
            else:
                skill.skills_status_buffer = self.skill_status_buffer
                skill.start()
        self._local_skills[name] = skill
        logger.info(
            "Started skill '%s' execution_place=%s execution=%s",
            name,
            skill.execution_place.value,
            skill.execution_mode.value,
        )
        return name

    def start_skill_by_name(self, name: str, params: dict) -> None:
        """Start a registered skill by the name a ``START_SKILL`` carries.

        The registry is the one the think process picks from, so a name arriving
        from a UI, from the skill command queue or from the robot all resolve to
        the same class.
        """
        spec = self._skill_registry.get(name)
        if spec is None:
            logger.warning(
                "Cannot start unknown skill '%s'; known: %s",
                name,
                sorted(self._skill_registry),
            )
            return
        try:
            self.start_skill(spec["cls"](params=params))
        except Exception:
            logger.exception("Could not start skill '%s'", name)

    def stop_skill(
        self, name: str, reason: str = "", status: SkillStatus = SkillStatus.STOPPED
    ) -> None:
        """Stop a running skill.

        Args:
            reason: Why, passed to whoever started it when it ends.
            status: How a local skill ended: ``STOPPED`` when someone stops it,
                ``COMPLETED`` / ``FAILED`` when the tick retires one that ended
                itself. A remote skill's end is whatever the robot reports.
        """
        rec = self.running_skills.get(name)
        if rec is None:
            return
        if status is SkillStatus.STOPPED:
            self._log_episode_event("skill_stop", skill=name, reason=reason)
        skill = self._local_skills.pop(name, None)
        if skill is not None and not skill.execution_place.is_remote:
            self._remote_running.pop(name, None)
            if skill.execution_mode is ExecutionMode.INLINE:
                skill.teardown()
                # What teardown dispatched, e.g. a stop for the motion the
                # skill started, has to go out: the skill is no longer
                # drained by the tick.
                last_words: list[dict] = []
                self._drain_skill_actions(skill, last_words)
                self._pending_actions.extend(
                    Action(action_type=ActionType.MODULE_ACTION, data=slot)
                    for slot in last_words
                    if slot
                )
            elif skill.is_alive():
                skill.request_stop()
            logger.info("Stopped skill '%s'", name)
            self._skill_ended(name, status, reason)
            return

        # A remote skill is stopped by asking, so it is not stopped until the
        # robot says so. Its record stays in ``_remote_running`` until then:
        # dropping it here would also drop the module set that makes it
        # preemptable, and a status row can only restore the record, not the
        # set — so a lost stop would leave a skill nothing can ever displace.
        remote_name = type(skill).registry_name() if skill is not None else rec.name
        if reason:
            self._stop_reasons[remote_name] = reason
        now = time.monotonic()
        self._remote_stopping[remote_name] = (now, now)
        self._pending_actions.append(
            Action(action_type=ActionType.STOP_SKILL, data={"skill": remote_name})
        )
        logger.info("Stopping remote skill '%s'", remote_name)

    def _retry_remote_stops(self) -> None:
        """Keep asking the robot to stop a skill until its status says it did.

        The controller's action socket is ``ZMQ_CONFLATE``, so one ``STOP_SKILL``
        can be replaced by whatever the next tick sends — which, when the
        operator preempts a policy with a move, is that move. The
        confirmation travels the same way and can be lost too, hence the
        deadline: this gives up rather than asking forever.
        """
        now = time.monotonic()
        for name, (first, last) in list(self._remote_stopping.items()):
            if now - first > _REMOTE_STOP_TIMEOUT_S:
                logger.warning(
                    "Robot never confirmed skill '%s' stopped after %.1f s; "
                    "giving up and assuming it is gone",
                    name,
                    now - first,
                )
                self._remote_stopping.pop(name, None)
                self._remote_running.pop(name, None)
                self._skill_ended(name, SkillStatus.STOPPED, "stop never confirmed by the robot")
                continue
            if now - last < _REMOTE_STOP_RETRY_S:
                continue
            self._remote_stopping[name] = (first, now)
            self._pending_actions.append(
                Action(action_type=ActionType.STOP_SKILL, data={"skill": name})
            )

    def stop_all_skills(self, reason: str = "stopped by the operator") -> None:
        if not self._agent_dispatching:
            self._operator_took_over("stopped all skills")
        for name in list(self.running_skills):
            self.stop_skill(name, reason=reason)

    def _skill_ended(self, name: str, status: SkillStatus, reason: str = "") -> None:
        """One invocation of ``name`` is over; tell the agent if it started it.

        Called from the control thread for local skills and from the state
        receiver for remote ones, whose end is only known when the robot says.
        """
        invocation = self._invocations.pop(name, None)
        stored = self._stop_reasons.pop(name, "")
        log = self.episode_log
        if log is not None:
            log.event("skill_ended", skill=name, status=status.name.lower(), reason=reason or stored)
        if invocation is None:
            return
        if self.agent is not None:
            self.agent.on_skill_ended(invocation, status, reason or stored)

    def _operator_took_over(self, description: str) -> None:
        if self.agent is not None:
            self.agent.on_operator_action(description)

    def _log_episode_event(self, kind: str, **data) -> None:
        """Log a command in the episode, ``by`` whoever gave it.

        Only for the control thread: elsewhere ``_agent_dispatching`` says
        nothing about who is calling.
        """
        log = self.episode_log
        if log is not None:
            log.event(kind, by="agent" if self._agent_dispatching else "operator", **data)

    # ========= agent host ============
    def call_on_control_thread(self, fn: Callable[[], Any]) -> Future:
        """Run ``fn`` on the control thread at the next tick; its result lands in the future.

        Skill orchestration is single-threaded by design — every start, stop
        and preemption happens between two ticks — so a thread that wants to
        command the robot (the agent) asks rather than reaching in.
        """
        future: Future = Future()
        if not self.running:
            future.set_exception(RuntimeError("the control loop is not running"))
            return future
        self._control_calls.put((fn, future))
        return future

    def _run_control_calls(self) -> None:
        while True:
            try:
                fn, future = self._control_calls.get_nowait()
            except queue.Empty:
                return
            self._agent_dispatching = True
            try:
                future.set_result(fn())
            except Exception as exc:  # noqa: BLE001 - the caller decides what a failure means
                future.set_exception(exc)
            finally:
                self._agent_dispatching = False

    def skill_invocation(self, name: str) -> int | None:
        """The invocation of ``name`` running now, or ``None``."""
        return self._invocations.get(name)

    def skill_active(self, name: str) -> bool:
        """Running, or asked to stop and not yet confirmed stopped.

        Membership tests only: the agent asks from its own thread, and these
        dicts change under it.
        """
        return (
            name in self._local_skills
            or name in self._remote_running
            or name in self._remote_stopping
        )

    def field_value(self, module: str, field_name: str):
        """The latest ``module.field`` reading this controller has, or ``None``."""
        try:
            return self.last_observation[module][field_name]
        except (KeyError, TypeError):
            return None

    def agent_inputs(self) -> tuple[list[UIImage], dict[str, dict[str, str]]]:
        """What the agent observes: camera images with their overlays, and readings.

        Readings are the dashboard's, already formatted, so the agent reads
        the same numbers the operator does. ``agent.state_fields`` narrows them.
        """
        wanted = self.cfg.agent.state_fields if self.cfg.agent is not None else {}
        state: dict[str, dict[str, str]] = {}
        for title, rows in self._gui_dashboard_sections():
            module = title.removeprefix("obs · ")
            if module == title or (wanted and module not in wanted):
                continue
            fields = wanted.get(module)
            state[module] = {
                str(row[0]): str(row[1]) for row in rows if not fields or row[0] in fields
            }
        return self._collect_display_images(), state

    def on_agent_idle(self) -> None:
        """Called on the control thread when an agent run ends.

        Override to bring back whatever should run between tasks — teleop,
        say — so the operator is never left with nothing commanding the arm.
        """

    def check_agent_action(self, action_id: str, tool: str, elapsed_s: float) -> tuple | None:
        """A deterministic check on a running agent action, polled with the monitor.

        Returns ``("finish" | "fail" | "escalate", reason)`` to stop the action
        as succeeded or failed, or wake the planner, or ``None``. A finish or
        fail can add a third element, a tool to start without arguments once
        the action has stopped: ``("fail", "the arm stalled", "home")``. For
        what a reading shows better than a camera — an arm that stopped moving,
        a joint at its limit. ``action_id`` is new for every action, so state
        kept across polls can be reset when it changes.
        """
        return None

    # ========= services ============
    def start_service(self, module: str, field: str, **params) -> None:
        """Start the processor that computes ``module.field``, with *params*.

        Addressed by field rather than by host: the relay applies it when it
        computes the field, otherwise it reaches the robot. Which one does is
        the field's ``execution_place``, so moving a tracker between them is a config
        change and not a code change.
        """
        self._pending_actions.append(
            Action(
                action_type=ActionType.START_SERVICE,
                data={"module": module, "field": field, "params": params},
            )
        )
        self._log_episode_event("service_started", module=module, field=field, params=params)
        logger.info("Starting service %s/%s %s", module, field, params)

    def stop_service(self, module: str, field: str) -> None:
        self._log_episode_event("service_stopped", module=module, field=field)
        self._pending_actions.append(
            Action(
                action_type=ActionType.STOP_SERVICE,
                data={"module": module, "field": field},
            )
        )
        logger.info("Stopping service %s/%s", module, field)

    def _ingest_skill_status_records(self, records) -> None:
        """Update remote-running table from robot/server ``skill_status`` rows."""
        id_to_name = {
            int(spec["skill_id"]): name
            for name, spec in self._skill_registry.items()
            if spec.get("skill_id") is not None
        }
        for rec in records:
            # Deserialized off the wire (rows) or read from shared memory
            # (records), and only the robot's rows carry a reason.
            if isinstance(rec, dict):
                skill_id = int(rec.get("skill_id", 0))
                status = int(rec.get("status", 0))
                reason = str(rec.get("reason") or "")
            else:
                skill_id = int(getattr(rec, "skill_id", 0))
                status = int(getattr(rec, "status", 0))
                reason = ""
            name = id_to_name.get(skill_id)
            if name is None:
                continue
            if status == int(SkillStatus.RUNNING):
                spec = self._skill_registry.get(name, {})
                existing = self._remote_running.get(name)
                self._remote_running[name] = SkillRecord(
                    name=name,
                    execution_place=spec.get("execution_place", ExecutionPlace.ROBOT),
                    execution_mode=spec.get(
                        "execution_mode", ExecutionMode.PROCESS
                    ),
                    status=SkillStatus.RUNNING,
                    skill_id=skill_id,
                    # Status rows carry no params; keep what start_skill stored.
                    modules=existing.modules if existing is not None else set(),
                )
            elif status in (
                int(SkillStatus.COMPLETED),
                int(SkillStatus.FAILED),
                int(SkillStatus.STOPPED),
            ):
                was_running = self._remote_running.pop(name, None) is not None
                self._local_skills.pop(name, None)
                self._remote_stopping.pop(name, None)
                if was_running:
                    self._skill_ended(name, SkillStatus(status), reason)

    def _drain_skill_actions(self, skill, slots: list[dict]) -> None:
        """Merge everything *skill* queued into *slots*, one per queue position."""
        for module_name, module_queue in skill.action_queues.items():
            if module_queue is None or module_queue.empty():
                continue
            try:
                queued_actions = module_queue.get_all()
            except Exception:
                logger.exception(
                    "Could not drain '%s' actions from skill '%s'",
                    module_name,
                    skill.skill_name,
                )
                continue
            for index, module_action in enumerate(queued_actions):
                while len(slots) <= index:
                    slots.append({})
                _deep_merge_module_actions(
                    slots[index], {module_name: module_action.to_dict()}
                )

    def _tick_skills(self) -> Action | None:
        # Before the skills: an episode boundary that loses a race with a
        # teleop action is a corrupt episode, not a late one.
        record_request = self._retry_record_request()
        if record_request is not None:
            return record_request

        self._retry_remote_stops()
        self._run_control_calls()
        if self._pending_actions:
            return self._pending_actions.pop(0)

        queue = self.skill_command_queue
        if not queue.empty():
            try:
                queued = queue.get()
            except Exception:
                queued = None
            if queued is not None:
                qtype = int(queued.action_type)
                if qtype == int(ActionType.START_SKILL):
                    self.start_skill_by_name(
                        queued.data.get("skill"), queued.data.get("params", {})
                    )
                elif qtype == int(ActionType.STOP_SKILL):
                    name = queued.data.get("skill")
                    if name:
                        self.stop_skill(str(name))
                else:
                    logger.warning(
                        "Ignoring %s on the skill command queue, which carries "
                        "START_SKILL and STOP_SKILL only",
                        ActionType.name_of(qtype),
                    )
                return self._tick_skills()

        # Inline skills get stepped the way a runner steps the others, so
        # both end up writing into their own per-module queues rather than
        # one of them taking a shortcut the merge below cannot see.
        finished: list[tuple[str, SkillStatus, str]] = []
        for name, skill in list(self._local_skills.items()):
            if skill.execution_place.is_remote:
                continue
            if skill.execution_mode is not ExecutionMode.INLINE:
                # A skill in its own process or thread ends on its own loop;
                # what it leaves behind is how it ended.
                if not skill.is_alive():
                    error = skill.get_error()
                    if error is None:
                        finished.append((name, SkillStatus.COMPLETED, ""))
                    else:
                        finished.append((name, SkillStatus.FAILED, failure_reason(error[1])))
                continue
            try:
                skill.dispatch(skill.step())
            except SkillFailed as exc:
                logger.warning("Inline skill '%s' failed: %s", name, exc)
                finished.append((name, SkillStatus.FAILED, str(exc)))
                continue
            except Exception as exc:
                logger.exception("Inline skill '%s' crashed", name)
                finished.append((name, SkillStatus.FAILED, f"{type(exc).__name__}: {exc}"))
                continue
            if skill.finished:
                finished.append((name, SkillStatus.COMPLETED, ""))

        # One slot per position in what each skill queued. Slot 0 goes out
        # now; the rest queue up behind it. Merging index-wise rather than
        # flattening keeps both properties that matter: skills stepping on
        # the same tick stay simultaneous, and a skill that queued a chunk
        # of successive waypoints does not collapse to its last one.
        slots: list[dict] = []
        for skill in list(self._local_skills.values()):
            if skill.execution_place.is_remote:
                continue
            self._drain_skill_actions(skill, slots)
        for name, status, reason in finished:
            self.stop_skill(name, reason=reason, status=status)

        slots = [slot for slot in slots if slot]
        if not slots:
            return None
        self._pending_actions.extend(
            Action(action_type=ActionType.MODULE_ACTION, data=slot)
            for slot in slots[1:]
        )
        return Action(action_type=ActionType.MODULE_ACTION, data=slots[0])

    def control_step(self):
        """Tick the running skills and merge what they emit into one action.

        This is the declarative path: the loop has already dispatched this
        tick's events and ticked the local modules, so all that is left is to
        let the skills act. Overriding it is still supported — the
        control loop sends whatever it returns — and an override loses only
        skill orchestration, not event handling.
        """
        return self._tick_skills()

    def start(self, session: Callable[[], None] | None = None):
        """Run the controller until the operator quits or *session* is done.

        Args:
            session: A schedule driven from outside the UI loop — ``solarian
                record`` passes the one that brackets episodes. It runs on its
                own thread once the loop is live, and returning from it ends
                the run.
        """
        self.running = True
        self._thread_error = None
        # StatusBoard.set() during connect defers the UI thread; start it now.
        StatusBoard.start_ui()
        logger.info("Starting state receiver")

        threading.Thread(target=self._state_receiver, daemon=True).start()

        # spin control loop in a separate thread
        logger.info("Starting control loop")
        t = threading.Thread(target=self.control_loop, daemon=True)
        t.start()

        self.ui = build_ui(self)
        # A front end with no window or terminal of its own cannot see key
        # events, so the global OS listener is started for it and only for it.
        if not self.ui.captures_keys:
            self._start_keyboard_listener()

        # Hook for subclasses to start additional threads, and to bring up
        # whatever skill the application wants running from the start.
        self.on_start()
        if self.agent is not None:
            self.agent.start()

        # After on_start, so an episode opened here contains whatever skill
        # the application brought up with it. A session brackets its own
        # episodes and would only have to close this one first.
        if self.cfg.record and session is None:
            self.start_recording()

        if session is not None:
            threading.Thread(target=self._run_session, args=(session,), daemon=True).start()

        logger.info("Starting UI loop (%s)", type(self.ui).__name__)
        fps = self.cfg.fps or 30
        try:
            while self.running:
                start_loop_t = time.time_ns()

                quit_requested = False
                for command in self.ui.poll():
                    if not self.handle_ui_command(command):
                        quit_requested = True
                        break
                if quit_requested:
                    break

                self.ui.render(self.ui_snapshot())

                if self.events["exit_early"]:
                    break

                self.ui.tick(fps)
                manual_profile("ui loop latency", (time.time_ns() - start_loop_t))
        finally:
            if self.agent is not None:
                self.agent.close()
            # Sealed here rather than left to the robot's shutdown, which can
            # only mark the episode interrupted.
            if self.recording_state["active"]:
                self.stop_recording()
            self.running = False
            t.join(timeout=5.0)
            self._close_controller_recorder()
            self.events["exit_early"] = False
            self.ui.close()
            self._stop_keyboard_listener()

        if self._thread_error is not None:
            raise self._thread_error

    def _run_session(self, session: Callable[[], None]) -> None:
        try:
            session()
        except Exception as e:
            if self._thread_error is None:
                self._thread_error = e
        finally:
            # The schedule is the run: when it is over, so is the UI loop.
            self.running = False


    def is_calibrated(self):
        for module_name, module in self.modules.items():
            if not module.is_calibrated():
                return False
        return True

    def calibrate(self):
        for module_name, module in self.modules.items():
            module.calibrate()

    def write_calibration(self):
        for module_name, module in self.modules.items():
            module.write_calibration()


    # def get_sync_obs(self) -> dict:
    #     """
    #     Timestamp alignment policy
    #     We assume the cameras used for obs are always [0, k - 1], where k is the number of robots
    #     All other cameras, find corresponding frame with the nearest timestamp
    #     All low-dim observations, interpolate with respect to 'current' time
    #     """

    #     "observat`ion dict"
    #     # assert self.is_ready

    #     # get data
    #     # 60 Hz, camera_calibrated_timestamp
    #     k = math.ceil(
    #         self.cfg.camera_obs_horizon * self.cfg.camera_down_sample_steps \
    #         * (60 / self.frequency)) + 2 # here 2 is adjustable, typically 1 should be enough
    #     # print('==>k  ', k, self.camera_obs_horizon, self.camera_down_sample_steps, self.frequency)
    #     self.last_camera_data = self.camera.get(
    #         k=k, 
    #         out=self.last_camera_data)

    #     # both have more than n_obs_steps data
    #     last_robots_data = list()
    #     last_grippers_data = list()
    #     # 125/500 hz, robot_receive_timestamp
    #     for robot in self.robots:
    #         last_robots_data.append(robot.get_all_state())
    #     # 30 hz, gripper_receive_timestamp
    #     for gripper in self.grippers:
    #         last_grippers_data.append(gripper.get_all_state())

    #     # select align_camera_idx
    #     num_obs_cameras = len(self.robots)
    #     align_camera_idx = None
    #     running_best_error = np.inf
   
    #     for camera_idx in range(num_obs_cameras):
    #         this_error = 0
    #         this_timestamp = self.last_camera_data[camera_idx]['timestamp'][-1]
    #         for other_camera_idx in range(num_obs_cameras):
    #             if other_camera_idx == camera_idx:
    #                 continue
    #             other_timestep_idx = -1
    #             while True:
    #                 if self.last_camera_data[other_camera_idx]['timestamp'][other_timestep_idx] < this_timestamp:
    #                     this_error += this_timestamp - self.last_camera_data[other_camera_idx]['timestamp'][other_timestep_idx]
    #                     break
    #                 other_timestep_idx -= 1
    #         if align_camera_idx is None or this_error < running_best_error:
    #             running_best_error = this_error
    #             align_camera_idx = camera_idx

    #     last_timestamp = self.last_camera_data[align_camera_idx]['timestamp'][-1]
    #     dt = 1 / self.frequency

    #     # align camera obs timestamps
    #     camera_obs_timestamps = last_timestamp - (
    #         np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt)
    #     camera_obs = dict()
    #     for camera_idx, value in self.last_camera_data.items():
    #         this_timestamps = value['timestamp']
    #         this_idxs = list()
    #         for t in camera_obs_timestamps:
    #             nn_idx = np.argmin(np.abs(this_timestamps - t))
    #             # if np.abs(this_timestamps - t)[nn_idx] > 1.0 / 120 and camera_idx != 3:
    #             #     print('ERROR!!!  ', camera_idx, len(this_timestamps), nn_idx, (this_timestamps - t)[nn_idx-1: nn_idx+2])
    #             this_idxs.append(nn_idx)
    #         # remap key
    #         camera_obs[f'camera{camera_idx}_rgb'] = value['color'][this_idxs]

    #     # obs_data to return (it only includes camera data at this stage)
    #     obs_data = dict(camera_obs)

    #     # include camera timesteps
    #     obs_data['timestamp'] = camera_obs_timestamps

    #     # align robot obs
    #     robot_obs_timestamps = last_timestamp - (
    #         np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)
    #     for robot_idx, last_robot_data in enumerate(last_robots_data):
    #         robot_pose_interpolator = PoseInterpolator(
    #             t=last_robot_data['robot_timestamp'], 
    #             x=last_robot_data['ActualTCPPose'])
    #         robot_pose = robot_pose_interpolator(robot_obs_timestamps)
    #         robot_obs = {
    #             f'robot{robot_idx}_eef_pos': robot_pose[...,:3],
    #             f'robot{robot_idx}_eef_rot_axis_angle': robot_pose[...,3:]
    #         }
    #         # update obs_data
    #         obs_data.update(robot_obs)

    #     # align gripper obs
    #     gripper_obs_timestamps = last_timestamp - (
    #         np.arange(self.gripper_obs_horizon)[::-1] * self.gripper_down_sample_steps * dt)
    #     for robot_idx, last_gripper_data in enumerate(last_grippers_data):
    #         # align gripper obs
    #         gripper_interpolator = get_interp1d(
    #             t=last_gripper_data['gripper_timestamp'],
    #             x=last_gripper_data['gripper_position'][...,None]
    #         )
    #         gripper_obs = {
    #             f'robot{robot_idx}_gripper_width': gripper_interpolator(gripper_obs_timestamps)
    #         }

    #         # update obs_data
    #         obs_data.update(gripper_obs)

    #     # accumulate obs
    #     if self.obs_accumulator is not None:
    #         for robot_idx, last_robot_data in enumerate(last_robots_data):
    #             self.obs_accumulator.put(
    #                 data={
    #                     f'robot{robot_idx}_eef_pose': last_robot_data['ActualTCPPose'],
    #                     f'robot{robot_idx}_joint_pos': last_robot_data['ActualQ'],
    #                     f'robot{robot_idx}_joint_vel': last_robot_data['ActualQd'],
    #                 },
    #                 timestamps=last_robot_data['robot_timestamp']
    #             )

    #         for robot_idx, last_gripper_data in enumerate(last_grippers_data):
    #             self.obs_accumulator.put(
    #                 data={
    #                     f'robot{robot_idx}_gripper_width': last_gripper_data['gripper_position'][...,None]
    #                 },
    #                 timestamps=last_gripper_data['gripper_timestamp']
    #             )

    #     return obs_data
