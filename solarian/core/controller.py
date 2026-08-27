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
from abc import ABC, abstractmethod
from typing import Any
from multiprocessing.managers import SharedMemoryManager

# pynput is imported lazily inside ``_start_keyboard_listener`` instead of
# at module load. On macOS ``pynput.keyboard`` transitively initializes
# ``Quartz`` / ``AppKit`` / ``AVFoundation`` (``AVCaptureDevice``,
# ``NSResponder``, ...) the moment it's imported. Once those Obj-C classes
# have been ``+initialize``'d in a process, any subsequent ``fork()`` of
# that process aborts the child the moment it touches Cocoa (Apple's
# hard-coded fork-safety net; not bypassed by
# ``OBJC_DISABLE_INITIALIZE_FORK_SAFETY``). Importing this module in a
# parent process that later forks camera / robot / gripper workers (as
# UMI's eval_real.py does on macOS) therefore breaks those forks unless
# the pynput load is deferred until after the forks. ``None`` here is the
# "not yet loaded" sentinel; the lazy loader below replaces it with the
# real module the first time ``_start_keyboard_listener`` runs, or with
# ``False`` if pynput is not installed / fails to import.
_pynput_keyboard = None
_PYNPUT_TRIED = False


def _get_pynput_keyboard():
    """Return ``pynput.keyboard`` (cached), or ``None`` if unavailable.

    Deferred so that merely importing this module does NOT initialize
    Cocoa in the parent process; see comment above.
    """
    global _pynput_keyboard, _PYNPUT_TRIED
    if _PYNPUT_TRIED:
        return _pynput_keyboard
    _PYNPUT_TRIED = True
    try:
        from pynput import keyboard as _kb
        _pynput_keyboard = _kb
    except Exception:  # pragma: no cover - pynput is optional / may fail on some platforms
        _pynput_keyboard = None
    return _pynput_keyboard



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
from solarian.utils import profile, manual_profile, get_class, StatusBoard
from solarian.conf import settings
from solarian.core.data import ImageData, BoundingBoxData
from solarian.perception.processor import field_name_for_input
from solarian.utils.shared_memory.observation_ring_buffer import ObservationRingBuffer
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


class Controller(ABC):

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

        # Create modules from class-level attributes
        for name in self.__class__.__dict__:
            val = getattr(self, name, None)
            if isinstance(val, Module):
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
        # The controller keeps a per-module ObservationRingBuffer so downstream
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
        self.observation_buffers: dict[str, ObservationRingBuffer] = {}
        self._setup_remote_observation_schema()
        self._create_observation_buffers()

        # Global keyboard listener (pynput) that delivers real OS key-down
        # / key-up events to the on_key_press / on_key_release hooks.
        # Started in :meth:`start`, stopped when ``start`` returns.
        self._keyboard_listener = None
        # Tracks which chars pynput currently reports as held, so we can
        # filter out OS auto-repeat and only fire :meth:`on_key_press`
        # once per physical press.
        self._keyboard_held: set[str] = set()

    def load_module_config(self, module_name: str, module_class: type):
        """Load module config from YAML using the module's own config class"""
        conf_path = os.environ.get("SOLARIAN_CONF_PATH", os.path.join(os.getcwd(), "conf"))
        yaml_path = os.path.join(conf_path, f"{module_name}.yaml")
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
        """Create a shared-memory ``ObservationRingBuffer`` for every module
        declared in the active ``Observation`` subclass."""
        example_observation = Observation.zeros()
        for name, module_obs in example_observation.items():
            if name in self.observation_buffers:
                continue
            try:
                self.observation_buffers[name] = ObservationRingBuffer.create_from_example(
                    shm_manager=self.shm_manager,
                    example_observation=module_obs,
                    get_time_budget=0.2,
                    put_desired_frequency=120,
                )
            except Exception as e:
                logger.warning(
                    f"Could not create observation buffer for '{name}': {e}"
                )

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
            for rec in msg.data["skill_status"]:
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
        """Push *module_observation* into its ``ObservationRingBuffer``.

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
            module.load_calibration()

            if module.cfg.calibration_file is None:
                logger.error(f"No calibration required for module {module_name}")
                continue

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
                    self._dispatch_key_press(ch)
                    self._dispatch_key_release(ch)
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

        Override in keyboard-teleop controllers. Keys listed here (split on
        whitespace) are highlighted while held.
        """
        return []

    def _gui_loop(self, episode_time_s):
        """Main loop for GUI mode: render the single-window dashboard/cameras
        and route this window's key events to on_key_press / on_key_release."""
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

                quit_requested = gui.poll(
                    on_press=self._dispatch_key_press,
                    on_release=self._dispatch_key_release,
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
    # We use pynput for global OS-level key-down / key-up events instead of
    # cv2.waitKey synthesis.  cv2 reports key events inconsistently across
    # platforms (especially on macOS with the cocoa backend, where we saw
    # per-frame bursts of synthesised releases), so bypassing it entirely
    # gives reliable one-press-one-release semantics.

    @staticmethod
    def _pynput_char(key) -> str | None:
        """Return the lowercase character for a pynput key, or ``None`` for
        non-character keys (arrows, modifiers, function keys, ...)."""
        try:
            char = key.char
        except AttributeError:
            return None
        if not char:
            return None
        return char.lower()

    def _on_pynput_press(self, key) -> None:
        char = self._pynput_char(key)
        logger.debug("pynput press: key=%r char=%r held=%r", key, char, self._keyboard_held)
        if char is None:
            return
        # Filter OS auto-repeat: only fire on the rising edge.
        if char in self._keyboard_held:
            return
        self._keyboard_held.add(char)
        self._dispatch_key_press(char)

    def _on_pynput_release(self, key) -> None:
        char = self._pynput_char(key)
        logger.debug("pynput release: key=%r char=%r held=%r", key, char, self._keyboard_held)
        if char is None:
            return
        # We deliberately do NOT gate release on having seen a matching
        # press: pynput on macOS occasionally reports slightly different
        # ``key.char`` values for the same physical key across press and
        # release (e.g. during IME / dead-key / layout transitions), and
        # swallowing those releases is worse than letting a stray one
        # through.  ``_keyboard_held`` is still used to suppress auto-
        # repeat presses.
        self._keyboard_held.discard(char)
        self._dispatch_key_release(char)

    def _start_keyboard_listener(self) -> None:
        """Start a daemon pynput listener that routes global keyboard
        events through :meth:`on_key_press` / :meth:`on_key_release`.
        """
        if self._keyboard_listener is not None:
            return
        pynput_keyboard = _get_pynput_keyboard()
        if pynput_keyboard is None:
            logger.warning(
                "pynput is not available; keyboard hooks (on_key_press / "
                "on_key_release) will not be invoked in this controller."
            )
            return
        try:
            listener = pynput_keyboard.Listener(
                on_press=self._on_pynput_press,
                on_release=self._on_pynput_release,
            )
            listener.daemon = True
            listener.start()
            self._keyboard_listener = listener
            logger.info("pynput keyboard listener started")
        except Exception:
            logger.exception("Failed to start pynput keyboard listener")
            self._keyboard_listener = None

    def _stop_keyboard_listener(self) -> None:
        listener = self._keyboard_listener
        if listener is None:
            return
        self._keyboard_listener = None
        try:
            listener.stop()
        except Exception:
            logger.debug("pynput listener.stop() raised", exc_info=True)
        self._keyboard_held.clear()

    def _dispatch_key_press(self, char: str) -> None:
        logger.debug("dispatching on_key_press(%r)", char)
        try:
            self.on_key_press(char)
        except Exception:
            logger.exception("on_key_press(%r) raised", char)

    def _dispatch_key_release(self, char: str) -> None:
        logger.debug("dispatching on_key_release(%r)", char)
        try:
            self.on_key_release(char)
        except Exception:
            logger.exception("on_key_release(%r) raised", char)

    def on_key_press(self, char: str):
        """Called once when *char* transitions from released to pressed.

        Override in subclasses.  OS auto-repeat while the key is held does
        not re-fire this hook.
        """
        pass

    def on_key_release(self, char: str):
        """Called once when *char* transitions from pressed to released.

        Override in subclasses.
        """
        pass

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

        # Global keyboard listener — provides real OS key-down / key-up
        # events to the on_key_press / on_key_release hooks.  Skipped in
        # headless mode (we read stdin) and in GUI mode (the window captures
        # keys itself, needing no OS accessibility permission).
        if not self.cfg.headless and not gui_mode:
            self._start_keyboard_listener()

        # Hook for subclasses to start additional threads
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


    @abstractmethod
    def control_step(self) -> Action:
        """Implements the control step
        
        Args:
        
        Returns:
            Action: The action to be sent to the robot
        """
        pass

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
