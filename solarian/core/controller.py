import os
import threading
import torch
import time
import zmq
import cv2
import random
import math
import string
import numpy as np
from abc import ABC, abstractmethod
from multiprocessing.managers import SharedMemoryManager


from solarian.utils import busy_wait
from solarian.core import Action
from solarian.core import Observation
from solarian.core.configs import ControllerConfig, ActionValue
from solarian.core.message import Message
from solarian.utils.shared_memory.observation_ring_buffer import ObservationRingBuffer
from solarian.core.module import Module
from solarian.utils import profile, manual_profile, get_class, StatusBoard
from solarian.conf import settings
from solarian.core.data import ImageData, BoundingBoxData


from screeninfo import get_monitors

import logging
logger = logging.getLogger(__name__)

try:
    monitors = get_monitors()
    m = monitors[0]
    sw, sh, sx, sy = m.width, m.height, m.x, m.y
except:
    print("No monitors found.")


class Controller(ABC):
    
    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.context = zmq.Context()
        
        self.clicked_point = None
        self.clicked_id = None

        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
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

        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        # Observation buffers are lazily initialized on the first received
        # observation so the schema (including image shapes) is derived from
        # real data rather than from Observation.zeros(), which may have None
        # payloads for fields whose shapes are only known at runtime.
        self.module_observation_buffers = {}
        self.last_observation = {}

        # cv2 keyboard simulation state (char → monotonic timestamp)
        self._held_keys: dict[str, float] = {}
        self._held_keys_last_seen: dict[str, float] = {}

    def load_module_config(self, module_name: str, module_class: type):
        """Load module config from YAML using the module's own config class"""
        from solarian.utils.configs import load_config_yaml
        conf_path = os.environ.get("SOLARIAN_CONF_PATH", os.path.join(os.getcwd(), "conf"))
        yaml_path = os.path.join(conf_path, f"{module_name}.yaml")
        module_cfg = load_config_yaml(yaml_path, module_class.config_class)
        return module_cfg

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
        raw_obs = msg.data["observation"]
        observation = Observation(raw_obs)
        
        for module_name, module_observation in observation.items():
            self.last_observation[module_name] = module_observation

            if module_name not in self.module_observation_buffers:
                self.module_observation_buffers[module_name] = ObservationRingBuffer.create_from_example(
                    shm_manager=self.shm_manager,
                    example_observation=module_observation,
                    get_time_budget=0.01,
                    put_desired_frequency=200
                )
            try:
                self.module_observation_buffers[module_name].put(module_observation, wait=False)
            except TimeoutError:
                pass
            StatusBoard.set(f"{module_name} Observation", module_observation)
            StatusBoard.set(f"{module_name} Observation Buffer Size", self.module_observation_buffers[module_name].count)

        if msg.data.get("click_ack", None) and msg.data["click_ack"] == self.clicked_id:
            logger.info("Click acknowledged", self.clicked_id)
            self.clicked_id = None
            self.clicked_point = None

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

    @profile
    def send_action(self, action: Action):
        """Send control actions to robot"""
        msg = Message()
        msg.controller_timestamp = time.time_ns()
        msg.data = {"action": action.to_dict()}

        if self.clicked_point is not None:
            msg.data["click"] = {
                "point": self.clicked_point,
                "id": self.clicked_id
            }
        self.action_socket.send(msg.serialize())

    def send_stop_tracking(self):
        msg = Message()
        msg.controller_timestamp = time.time_ns()
        msg.data = {"stop_tracking": True}
        self.action_socket.send(msg.serialize())

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
                key = cv2.waitKey(wait_ms) & 0xFF
                self._process_cv2_key(key)
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

            source_to_field = {}
            for fname, sf in sensor_fields.items():
                source_to_field[id(sf.source)] = fname
            for fname, cf in calculated_fields.items():
                source_to_field[id(cf.output)] = fname

            for field_name, cf in calculated_fields.items():
                if cf.data_class is not BoundingBoxData:
                    continue
                for inp in cf.inputs:
                    input_field = source_to_field.get(id(inp))
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
            module_obs = self.last_observation.get(module_name)
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
            module_obs = self.last_observation.get(img_key[0])
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
            for (module_name, field_name), img in images.items():
                window_name = f"{module_name}/{field_name}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                height, width = img.shape[:2]
                cv2.resizeWindow(window_name, width, height)
                cv2.setMouseCallback(window_name, self.mouse_callback)
                cv2.moveWindow(window_name, sx + (sw - width)//2, sy + (sh - height)//2)
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
                    self.send_action(self.last_action)

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

    # -- Keyboard simulation from cv2.waitKey --------------------------------
    # cv2.waitKey only reports which key is down on a given frame; it has no
    # release event.  We synthesise on_key_press / on_key_release by tracking
    # timing and accounting for the OS initial key-repeat delay.

    # Long grace window that covers the OS repeat delay (~400-500 ms on macOS).
    _KEY_INITIAL_GRACE_S = 0.6
    # Once OS repeat is flowing, release quickly after the last event.
    _KEY_REPEAT_TIMEOUT_S = 0.15

    def _process_cv2_key(self, key_code: int):
        """Called every display-loop frame with the raw cv2.waitKey result."""
        now = time.monotonic()

        # Register key-down for detected key
        if key_code != 0xFF:
            try:
                char = chr(key_code).lower()
            except (ValueError, OverflowError):
                char = None
            if char is not None:
                if char not in self._held_keys:
                    self._held_keys[char] = now
                    self.on_key_press(char)
                self._held_keys_last_seen[char] = now

        # Check for releases
        for char in list(self._held_keys):
            first = self._held_keys[char]
            last = self._held_keys_last_seen[char]
            held_duration = now - first
            since_last_event = now - last
            if held_duration < self._KEY_INITIAL_GRACE_S:
                continue
            if since_last_event > self._KEY_REPEAT_TIMEOUT_S:
                del self._held_keys[char]
                del self._held_keys_last_seen[char]
                self.on_key_release(char)

    def on_key_press(self, char: str):
        """Called once when a key is first pressed. Override in subclasses."""
        pass

    def on_key_release(self, char: str):
        """Called once when a key is considered released. Override in subclasses."""
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
        logger.info("Starting state receiver")

        threading.Thread(target=self._state_receiver, daemon=True).start()

        # spin control loop in a separate thread
        logger.info("Starting control loop")
        t = threading.Thread(target=self.control_loop, args=(dataset, episode_time_s + 1, single_task), daemon=True)
        t.start()
        
        # Hook for subclasses to start additional threads
        self.on_start()
        
        # run display loop in main thread
        logger.info("Starting display loop")
        try:
            self.display_loop(episode_time_s)
        finally:
            self.running = False
            t.join(timeout=5.0)
            self.events["exit_early"] = False

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
