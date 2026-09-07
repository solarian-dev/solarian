import os
import select
import sys
import threading
import time
import zmq
import cv2
import random
import math
import string
import numpy as np
from abc import ABC
from dataclasses import dataclass, field
from typing import Any
from multiprocessing.managers import SharedMemoryManager

from solarian.utils import busy_wait
from solarian.core import Action
from solarian.core.action import ActionType
from solarian.core import Observation
from solarian.core.configs import ControllerConfig, ActionValue
from solarian.core.message import Message
from solarian.core.module import Module
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
from solarian.core.ai.system1.skill import Skill, SkillStatus
from solarian.utils import profile, manual_profile, get_class, StatusBoard
from solarian.conf import settings
from solarian.core.data import ImageData, BoundingBoxData
from solarian.perception.processor import field_name_for_input
from solarian.utils.shared_memory.observation_buffer import ObservationBuffer
from solarian.utils.shared_memory.action_queue import ModuleActionQueue
from solarian.utils.shared_memory.skill_status_buffer import SkillStatusBuffer
from solarian.utils.configs import load_config_yaml


import logging
logger = logging.getLogger(__name__)

# Monitor geometry used to center cv2 windows. Resolved lazily on first
# use via ``_get_screen_geometry()`` because ``screeninfo`` on macOS
# imports ``pyobjc`` (AppKit / Foundation / Cocoa) at module load, which
# then breaks any subsequent fork() in the parent process (Apple's
# AppKit fork-safety net; not bypassed by
# ``OBJC_DISABLE_INITIALIZE_FORK_SAFETY``). Importing this module in a
# parent that later forks workers (e.g. UMI's eval_real.py on macOS)
# would therefore poison every camera/visualizer worker. Defaults
# (0,0,0,0) center against the top-left corner, which the code already
# tolerated as the "no screeninfo" fallback.
sw, sh, sx, sy = 0, 0, 0, 0
_SCREEN_GEOMETRY_RESOLVED = False


def _get_screen_geometry():
    """Populate ``sw, sh, sx, sy`` lazily from ``screeninfo.get_monitors``.

    Safe to call repeatedly; the ``screeninfo`` import (and the Cocoa init
    it triggers on macOS) happens at most once, and only AFTER the caller
    has chosen to do so -- typically inside a controller method that runs
    well after any fork()ed worker processes have been started.
    """
    global sw, sh, sx, sy, _SCREEN_GEOMETRY_RESOLVED
    if _SCREEN_GEOMETRY_RESOLVED:
        return sw, sh, sx, sy
    _SCREEN_GEOMETRY_RESOLVED = True
    try:
        from screeninfo import get_monitors
        monitors = get_monitors()
        m = monitors[0]
        sw, sh, sx, sy = m.width, m.height, m.x, m.y
    except Exception:
        pass
    return sw, sh, sx, sy


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

    # Skill classes this controller can run. Only a registry: it is what the
    # think process picks from and what a ``START_SKILL`` name resolves to,
    # not a set of ready-made instances — a key handler constructs the skill
    # it wants with the arguments it wants.
    skills: tuple[type[Skill], ...] = ()

    # When True and not running headless, use the single-window pygame GUI
    # (dashboard + camera grid + on-window key capture) instead of the OpenCV
    # camera windows plus the global pynput listener. The GUI captures keys via
    # window focus, so it needs no OS accessibility / input-monitoring
    # permission. Enabled by default so the GUI is always shown when not
    # headless; set ``use_gui = False`` on a subclass to explicitly opt out.
    use_gui = True

    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.context = zmq.Context()
        
        self.clicked_point = None
        self.clicked_id = None

        try:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        except ImportError:
            self.device = 'cpu'
        
        # Initialize OpenCV windows once
        self.windows_initialized = []
        self._image_field_keys: list[tuple[str, str]] | None = None
        self._bbox_field_map: dict[tuple[str, str], list[tuple[str, str]]] | None = None
        
        # action socket
        self.action_socket = self.context.socket(zmq.PUSH)
        self.action_socket.setsockopt(zmq.CONFLATE, 1)
        self.action_socket.setsockopt(zmq.LINGER, 0)
        self.action_socket.connect(f"tcp://{self.cfg.connection.ip}:{self.cfg.connection.action_receiver_port}")
        
        # Image subscription
        self.state_socket = self.context.socket(zmq.SUB)
        self.state_socket.setsockopt(zmq.CONFLATE, 1)
        self.state_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.state_socket.setsockopt(zmq.LINGER, 0)
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

        self.modules = {}

        # Create modules from class-level attributes. Walks the MRO so that
        # modules declared on a base controller -- the keyboard, for one --
        # exist on every subclass, matching how Robot collects modules.
        for klass in type(self).__mro__:
            for name, val in klass.__dict__.items():
                if name in self.modules or not isinstance(val, Module):
                    continue
                module_class = val.__class__
                module_cfg = self.load_module_config(name, module_class)
                self.modules[name] = module_class(module_cfg, name)
                setattr(self, name, self.modules[name])

        # Create events dictionary directly instead of using init_keyboard_listener
        self.events = {
            "exit_early": False,
            "rerecord_episode": False, 
            "stop_recording": False,
            "track all": False,
            "verbose": False,
            "prompt": None,
            "is_prompting": False,
            "reset_policy": False
        }

        self.last_observation = Observation({})

        # Shared-memory infrastructure mirroring what Robot owns on its side.
        # The controller keeps a per-module ObservationBuffer so downstream
        # consumers (e.g. ThinkProcess and skill processes) can read the
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
        #: Control-plane inbox: START_SKILL / STOP_SKILL from the think
        #: process. Skills command modules through their own queues, not here.
        self.skill_command_queue = ActionQueue(
            shm_manager=self.shm_manager,
            buffer_size=256,
            max_payload_bytes=8192,
        )

    def load_module_config(self, module_name: str, module_class: type):
        """Load module config from YAML using the module's own config class.

        Modules with nothing to describe -- an input device has no ports,
        calibration, or limits -- declare a ``default_config`` and are used
        without the project shipping an empty YAML file for them.
        """
        conf_path = os.environ.get("SOLARIAN_CONF_PATH", os.path.join(os.getcwd(), "conf"))
        yaml_path = os.path.join(conf_path, f"{module_name}.yaml")
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
            module_class(module_cfg, name)

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

    @profile
    def get_state(self):
        bytes = self.state_socket.recv(zmq.NOBLOCK)
        msg = Message.deserialize(bytes)

        latency_ms = latency_ms_from_stamp_ns(msg.robot_timestamp)
        if latency_ms is not None:
            StatusBoard.set(
                "Obs latency (ms)",
                format_latency_status(self._obs_latency, latency_ms),
            )

        if "observation" in msg.data:
            raw_obs = msg.data["observation"]
            observation = Observation(raw_obs)

            age_ms = self._observation_age_ms(observation)
            if age_ms is not None:
                StatusBoard.set(
                    "Obs age (ms)",
                    format_latency_status(self._obs_age_latency, age_ms),
                )

            for module_name, module_observation in observation.items():
                self.last_observation[module_name] = module_observation
                self._put_module_observation(module_name, module_observation)

            StatusBoard.set("Observation", self.last_observation)

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

        if msg.data.get("click_ack", None) and msg.data["click_ack"] == self.clicked_id:
            logger.info(f"Click acknowledged {self.clicked_id}")
            self.clicked_id = None
            self.clicked_point = None

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

        if self.clicked_point is not None:
            msg.data["click"] = {
                "point": self.clicked_point,
                "id": self.clicked_id
            }
        self.action_socket.send(msg.serialize())

    def send_stop_tracking(self):
        msg = Message()
        msg.controller_timestamp = synced_time_ns()
        msg.data = {"stop_tracking": True}
        self.action_socket.send(msg.serialize())

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
            if module.cfg.calibration_file is None:
                # Input devices and other calibration-free modules.
                continue

            module.load_calibration()

            module.connect()

            if module.calibration is None:
                module.calibrate()

            if not module.is_calibrated():
                module.write_calibration()

            calibration = module.get_calibration()
            if calibration:
                module.disconnect()
                module.connect(calibration)
            else:
                module.disconnect()

        self.connected = True
        logger.info("Controller connected")

    def sync_clock(self) -> None:
        """Estimate wall-clock offset vs the relay server (soft-fail)."""
        try:
            result = sync_with_server(
                self.cfg.connection.ip,
                self.cfg.connection.clock_sync_port,
                context=self.context,
            )
            StatusBoard.set("Clock offset (ms)", result.offset_s * 1000.0)
            StatusBoard.set("Clock sync RTT (ms)", result.rtt_s * 1000.0)
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
        if not self.is_connected():
            return
        self.running = False
        self.action_socket.close()
        self.state_socket.close()
        self.context.term()
        self.connected = False

    def stop(self):
        self.running = False

    # def record_frame(self, dataset, action, task):

    #     # "action": nparray shape 6
    #     # "observation.state": nparray shape 6
    #     # "observation.images.front": nparray shape (480, 640, 3)
    #     # "observation.images.wrist": nparray shape (480, 640, 3)
    #     # "task": string

    #     self.last_observation = self.observation_buffer.get()
        
    #     frame = {**self.last_observation.to_lerobot_format(), **action.to_lerobot_format()}

    #     # resize images if necessary
    #     for key in frame:
    #         if key.startswith("observation.images."):
    #             img = frame[key]
    #             if img.shape[0] != self.cfg.video_res[0] or img.shape[1] != self.cfg.video_res[1]:
    #                 frame[key] = cv2.resize(img, (self.cfg.video_res[1], self.cfg.video_res[0]), interpolation=cv2.INTER_AREA)
        
    #     dataset.add_frame(frame, task)
        
    
    def _headless_loop(self, episode_time_s):
        """Main loop for headless mode — no cv2 GUI, reads keypresses from stdin."""
        import tty
        import termios

        timestamp = 0
        start_episode_t = time.perf_counter()

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            logger.info("Headless mode — press 'q' to quit")

            while timestamp < episode_time_s and self.running:
                start_loop_t = time.time_ns()
                timestamp = time.perf_counter() - start_episode_t

                if self.events["exit_early"]:
                    break

                if select.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1).lower()
                    # Headless cbreak mode has no release event, so every
                    # character read is treated as a full click (down+up).
                    self.keyboard.on_press(ch)
                    self.keyboard.on_release(ch)
                    if ch == 'q':
                        break

                dt_s = (time.time_ns() - start_loop_t) / 1e9
                sleep_s = (1.0 / self.cfg.fps) - dt_s if self.cfg.fps else 0.01
                if sleep_s > 0:
                    time.sleep(sleep_s)

                manual_profile("headless loop latency", (time.time_ns() - start_loop_t))
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def display_loop(self, episode_time_s):
        timestamp = 0
        start_episode_t = time.perf_counter()
        try:
            while timestamp < episode_time_s and self.running:
                start_loop_t = time.time_ns()

                result = self.display()

                if result is False:
                    break

                timestamp = time.perf_counter() - start_episode_t
                if self.events["exit_early"]:
                    break

                dt_s = (time.time_ns() - start_loop_t) / 1e9
                if self.cfg.fps is not None:
                    wait_ms = max(1, int(((1.0 / self.cfg.fps) - dt_s) * 1000))
                else:
                    wait_ms = 1
                # cv2.waitKey is still needed to pump the cv2 event loop
                # (without it, imshow windows go unresponsive).  Key events
                # themselves are delivered via the pynput listener started
                # in :meth:`start`, so we only use the return value to
                # notice the 'q' quit shortcut while a cv2 window is focused.
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord('q'):
                    break

                manual_profile("display loop latency", (time.time_ns() - start_loop_t))
        finally:
            if self.windows_initialized:
                for name in self.windows_initialized:
                    try:
                        cv2.destroyWindow(name)
                    except:
                        pass
                cv2.destroyAllWindows()
                self.windows_initialized = []
            
            
            
    
    # Mouse callback function
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            logger.info(f"Clicked at (x={x}, y={y})")
            self.clicked_point = (x, y)
            # generate random alphanumeric id
            self.clicked_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    
    def _build_display_schema(self):
        """Build and cache the mapping of image/bbox fields per module."""
        obs_spec = Observation.get_module_obs_spec()
        image_keys: list[tuple[str, str]] = []
        bbox_map: dict[tuple[str, str], list[tuple[str, str]]] = {}

        for module_name, module_obs_instance in obs_spec.items():
            module_obs_cls = type(module_obs_instance)
            sensor_fields = module_obs_cls.get_sensor_fields_spec()
            calculated_fields = module_obs_cls.get_calculated_fields_spec()

            all_fields = {**sensor_fields, **calculated_fields}
            for field_name, field_spec in all_fields.items():
                if field_spec.data_class is ImageData:
                    image_keys.append((module_name, field_name))

            for field_name, cf in calculated_fields.items():
                if cf.data_class is not BoundingBoxData:
                    continue
                for inp in cf.inputs:
                    input_field = field_name_for_input(
                        inp, sensor_fields, calculated_fields
                    )
                    if input_field:
                        img_key = (module_name, input_field)
                        bbox_map.setdefault(img_key, []).append(
                            (module_name, field_name)
                        )
                        break

        self._image_field_keys = image_keys
        self._bbox_field_map = bbox_map

    @profile
    def display(self):
        if self._image_field_keys is None or self._bbox_field_map is None:
            self._build_display_schema()
        assert self._image_field_keys is not None and self._bbox_field_map is not None

        images = {}
        image_bboxes = {}

        for module_name, field_name in self._image_field_keys:
            module_obs = self.last_observation[module_name]
            if module_obs is None:
                continue
            try:
                field_data = module_obs[field_name]
            except KeyError:
                continue
            if field_data is not None and field_data.value is not None:
                images[(module_name, field_name)] = field_data.value

        for img_key, bbox_fields in self._bbox_field_map.items():
            if img_key not in images:
                continue
            module_obs = self.last_observation[img_key[0]]
            if module_obs is None:
                continue
            for bbox_module, bbox_field in bbox_fields:
                try:
                    field_data = module_obs[bbox_field]
                except KeyError:
                    continue
                if field_data is not None and field_data.value is not None:
                    image_bboxes.setdefault(img_key, []).append(field_data.value)

        if not self.windows_initialized:
            # Resolve screen geometry lazily (see module-level
            # `_get_screen_geometry` for the fork-safety rationale).
            _sw, _sh, _sx, _sy = _get_screen_geometry()
            for (module_name, field_name), img in images.items():
                window_name = f"{module_name}/{field_name}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                height, width = img.shape[:2]
                cv2.resizeWindow(window_name, width, height)
                cv2.setMouseCallback(window_name, self.mouse_callback)
                cv2.moveWindow(window_name, _sx + (_sw - width)//2, _sy + (_sh - height)//2)
                self.windows_initialized.append(window_name)

        BBOX_COLOR = (255, 0, 0)

        for (module_name, field_name), img in images.items():
            img_key = (module_name, field_name)
            window_name = f"{module_name}/{field_name}"

            bbox_list = image_bboxes.get(img_key, [])
            if bbox_list:
                display_img = img.copy()
                for bbox in bbox_list:
                    if all(v == 0 for v in bbox[:4]):
                        continue
                    x1, y1, x2, y2 = bbox[:4]
                    cv2.rectangle(display_img,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                BBOX_COLOR, 2)
            else:
                display_img = img

            cv2.imshow(window_name, np.ascontiguousarray(display_img[:, :, ::-1]))

    # -- single-window GUI (dashboard + cameras + key capture) -------------
    def _collect_display_images(self) -> list[tuple[str, "np.ndarray"]]:
        """Return ``(label, rgb_image)`` for every image field in the latest
        observation, for rendering in the GUI camera grid."""
        if self._image_field_keys is None:
            self._build_display_schema()
        assert self._image_field_keys is not None

        images: list[tuple[str, np.ndarray]] = []
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
            if field_data is not None and field_data.value is not None:
                images.append((f"{module_name}/{field_name}", field_data.value))
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
        return [("space", "stop all skills")]

    def _gui_loop(self, episode_time_s):
        """Main loop for GUI mode: render the single-window dashboard/cameras
        and feed this window's key events into the keyboard module."""
        from solarian.core.controller_gui import ControllerGUI

        try:
            gui = ControllerGUI(title="Solarian")
        except Exception:
            logger.exception(
                "Failed to start pygame GUI; falling back to OpenCV display "
                "and the global pynput keyboard listener."
            )
            self._start_keyboard_listener()
            return self.display_loop(episode_time_s)

        fps = self.cfg.fps or 30
        timestamp = 0
        start_episode_t = time.perf_counter()
        try:
            while timestamp < episode_time_s and self.running:
                start_loop_t = time.time_ns()

                # The window owns key events here, so it feeds the keyboard
                # module directly and no OS-level listener is needed.
                quit_requested = gui.poll(
                    on_press=self.keyboard.on_press,
                    on_release=self.keyboard.on_release,
                )
                if quit_requested:
                    break

                gui.render(
                    sections=self._gui_dashboard_sections(),
                    images=self._collect_display_images(),
                    key_help=self.gui_key_help(),
                    fps=self.cfg.fps if self.cfg.fps else None,
                )

                timestamp = time.perf_counter() - start_episode_t
                if self.events["exit_early"]:
                    break

                gui.tick(fps)
                manual_profile("gui loop latency", (time.time_ns() - start_loop_t))
        finally:
            gui.close()

    def control_loop(self, dataset=None, episode_time_s=120, single_task=None):
        try:
            timestamp = 0
            start_episode_t = time.perf_counter()
            self.send_stop_tracking()
            while timestamp < episode_time_s:
                start_loop_t = time.time_ns()
                timestamp = time.perf_counter() - start_episode_t

                # Events and local state are refreshed by the loop, not by
                # control_step, so a controller that overrides control_step
                # still gets its event handlers run and its modules read.
                events = self._drain_events()
                self._tick_local_modules()
                self._fan_out_events(events)

                self.last_action = self.control_step()

                if self.last_action is not None:
                    StatusBoard.set("Action", self.last_action)
                    self._send_action(self.last_action)

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
        from solarian.core.ai.system1.registry import discover_available_skills

        skills: dict = {}
        for cls in type(self).skills:
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
        self._skill_registry: dict[str, dict[str, Any]] = (
            self._collect_controller_skills()
        )
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
            self.stop_all_skills()

    def start_skill(self, skill: Skill) -> str:
        """Run a skill the caller built, preempting whatever overlaps it.

        The caller constructs the skill with the arguments it wants, so this
        needs no spec to rebuild one from; all it decides is *where* the skill
        runs. A remote skill is not launched here at all — the robot or server
        owns it, and gets a ``START_SKILL`` naming the class and its params.
        """
        name = skill.skill_name
        if name in self.running_skills:
            # Starting is not toggling: pressing the same key twice re-runs the
            # skill rather than leaving nothing commanding the modules.
            self.stop_skill(name)
        for other in list(self._local_skills.values()):
            if self._modules_overlap(skill, other):
                self.stop_skill(other.skill_name)
        for other_name, rec in list(self.running_skills.items()):
            if other_name in self._local_skills:
                continue
            if rec.execution_place.is_remote and self._modules_overlap(skill, rec):
                self.stop_skill(other_name)
        if skill.execution_place.is_remote:
            self._pending_actions.append(
                Action(
                    action_type=ActionType.START_SKILL,
                    data={
                        "skill": type(skill).registry_name(),
                        "params": skill.params,
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

    def stop_skill(self, name: str) -> None:
        rec = self.running_skills.get(name)
        if rec is None:
            return
        skill = self._local_skills.pop(name, None)
        self._remote_running.pop(name, None)
        if skill is not None and not skill.execution_place.is_remote:
            if skill.execution_mode is ExecutionMode.INLINE:
                skill.teardown()
            else:
                skill.request_stop()
        else:
            remote_name = (
                type(skill).registry_name() if skill is not None else rec.name
            )
            self._pending_actions.append(
                Action(action_type=ActionType.STOP_SKILL, data={"skill": remote_name})
            )
        logger.info("Stopped skill '%s'", name)

    def stop_all_skills(self) -> None:
        for name in list(self.running_skills):
            self.stop_skill(name)

    def _ingest_skill_status_records(self, records) -> None:
        """Update remote-running table from robot/server ``skill_status`` rows."""
        id_to_name = {
            int(spec["skill_id"]): name
            for name, spec in self._skill_registry.items()
            if spec.get("skill_id") is not None
        }
        for rec in records:
            if isinstance(rec, dict):
                skill_id = int(rec.get("skill_id", 0))
                status = int(rec.get("status", 0))
            else:
                skill_id = int(getattr(rec, "skill_id", 0))
                status = int(getattr(rec, "status", 0))
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
                self._remote_running.pop(name, None)
                self._local_skills.pop(name, None)

    def _tick_skills(self) -> Action | None:
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
                    name = queued.data.get("skill")
                    params = queued.data.get("params", {})
                    spec = self._skill_registry.get(name)
                    if spec is None:
                        logger.warning(
                            "Cannot start unknown skill '%s'; known: %s",
                            name,
                            sorted(self._skill_registry),
                        )
                    else:
                        try:
                            self.start_skill(spec["cls"](params=params))
                        except Exception:
                            logger.exception("Could not start skill '%s'", name)
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
        finished: list[str] = []
        for name, skill in list(self._local_skills.items()):
            if skill.execution_place.is_remote:
                continue
            if skill.execution_mode is not ExecutionMode.INLINE:
                continue
            try:
                skill.dispatch(skill.step())
            except Exception:
                logger.exception("Inline skill '%s' crashed", name)
                finished.append(name)
                continue
            if skill.finished:
                finished.append(name)

        # One slot per position in what each skill queued. Slot 0 goes out
        # now; the rest queue up behind it. Merging index-wise rather than
        # flattening keeps both properties that matter: skills stepping on
        # the same tick stay simultaneous, and a skill that queued a chunk
        # of successive waypoints does not collapse to its last one.
        slots: list[dict] = []
        for skill in list(self._local_skills.values()):
            if skill.execution_place.is_remote:
                continue
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
        for name in finished:
            self.stop_skill(name)

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

    def start(self, dataset=None, episode_time_s=None, single_task=None):
        """Implements the control loop
        
        Args:
            dataset (Dataset, optional): Dataset to record data. Defaults to None.
            events (dict, optional): Events to trigger. Defaults to None.
            episode_time_s (float, optional): Episode time in seconds. Defaults to None.
            single_task (str, optional): Single task to run. Defaults to None.
        """

        if episode_time_s is None:
            episode_time_s = float("inf")

        self.running = True
        self._thread_error = None
        # StatusBoard.set() during connect defers the UI thread; start it now.
        StatusBoard.start_ui()
        logger.info("Starting state receiver")

        threading.Thread(target=self._state_receiver, daemon=True).start()

        # spin control loop in a separate thread
        logger.info("Starting control loop")
        t = threading.Thread(target=self.control_loop, args=(dataset, episode_time_s + 1, single_task), daemon=True)
        t.start()

        gui_mode = (not self.cfg.headless) and self.use_gui

        # Global OS key events, needed only when nothing else is capturing
        # them: headless mode reads stdin, and the GUI window captures keys
        # itself without needing OS accessibility permission.
        if not self.cfg.headless and not gui_mode:
            self._start_keyboard_listener()

        # Hook for subclasses to start additional threads, and to bring up
        # whatever skill the application wants running from the start.
        self.on_start()

        try:
            if self.cfg.headless:
                logger.info("Starting headless loop (no display)")
                try:
                    self._headless_loop(episode_time_s)
                finally:
                    self.running = False
                    t.join(timeout=5.0)
                    self.events["exit_early"] = False
            elif gui_mode:
                logger.info("Starting GUI loop (single-window dashboard)")
                try:
                    self._gui_loop(episode_time_s)
                finally:
                    self.running = False
                    t.join(timeout=5.0)
                    self.events["exit_early"] = False
            else:
                logger.info("Starting display loop")
                try:
                    self.display_loop(episode_time_s)
                finally:
                    self.running = False
                    t.join(timeout=5.0)
                    self.events["exit_early"] = False
        finally:
            self._stop_keyboard_listener()

        if self._thread_error is not None:
            raise self._thread_error


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
