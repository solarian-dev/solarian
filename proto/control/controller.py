import threading
import time
import zmq
import cv2
import numpy as np
from abc import ABC, abstractmethod

from proto.common.utils import busy_wait
from proto.robots import Action
from proto.common.configs import ControllerConfig, ValueName
from proto.control import Message
from proto.robots.observation import Observation
from proto.robots.module import Module
from proto.robots.component_bus import ComponentBus
from proto.common.utils import profile, manual_profile, get_class
from proto.conf import settings


class Controller(ABC):
    
    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.context = zmq.Context()
        
        self.is_connected = False
        self.clicked_point = None
        self.clicked_id = None
        
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
        self._verify_connection()
        
        self.last_observation = None
        self.last_msg = None
        self.observation_lock = threading.Lock()  # Thread-safe image access
        self.running = False
        self.modules = {}
        self.component_bus = None
        self.create_components()

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

    def create_components(self):
        self.component_bus = ComponentBus()
        for module_name, module_cfg in self.cfg.modules.items():
            module = get_class(settings.MODULE_CLASSES[module_name])(module_cfg, module_name, self.component_bus)
            self.modules[module_name] = module

    def _verify_connection(self, timeout=5000):
        """Verify that we can connect to the relay server"""
        print(f"Verifying connection to relay server at {self.cfg.connection.ip}...")
        
        # Create a temporary PUSH socket to test connection
        test_socket = self.context.socket(zmq.PUSH)
        test_socket.setsockopt(zmq.LINGER, 0)
        print(f"Attempting to connect to tcp://{self.cfg.connection.ip}:{self.cfg.connection.action_receiver_port}...")
        test_socket.connect(f"tcp://{self.cfg.connection.ip}:{self.cfg.connection.action_receiver_port}")
        
        try:
            # Try to send a message
            print("Attempting to send test message...")
            test_socket.send_string("test", zmq.NOBLOCK)
            print("Test message sent successfully")
            time.sleep(1)  # Give it a moment to ensure the message was sent
            return True
        except zmq.ZMQError as e:
            print(f"ZMQ error occurred: {e}")
            raise ConnectionError(f"ZMQ error while connecting to relay server: {e}")
        finally:
            print("Cleaning up test socket...")
            test_socket.close()
        
        print("Successfully connected to relay server")

    @profile
    def get_state(self):
        bytes = self.state_socket.recv(zmq.NOBLOCK)
        msg = Message.deserialize(bytes)
        observation = Observation.model_validate(msg.data["observation"])

        # Decode images
        for module_name in observation.data.keys():
            for sensor_name in observation.data[module_name].keys():
                if observation.data[module_name][sensor_name].get(ValueName.IMAGE) is not None:
                    img = observation.data[module_name][sensor_name][ValueName.IMAGE]
                    img = cv2.imdecode(
                        np.frombuffer(img, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    observation.data[module_name][sensor_name][ValueName.IMAGE] = img
        
        if msg.data.get("click_ack", None) and msg.data["click_ack"] == self.clicked_id:
            print("Click acknowledged", self.clicked_id)
            self.clicked_id = None
            self.clicked_point = None

        # if observation.get("start_pos_ack", None):
        #     print("Start position acknowledged")
        #     self.start_pos = False
                    
        # Thread-safe update observation
        with self.observation_lock:
            self.last_observation = observation
            self.last_msg = msg

    def _state_receiver(self):
        """Receive and decode image streams"""
        while self.running:
            try:
                self.get_state() 
            except zmq.Again:
                time.sleep(0.001)
            except Exception as e:
                print(f"[ERROR] Image receiver failed: {e}")

    def connect(self):
        """Start receiver threads"""
        self.calibration = {}
        for module_name, module in self.modules.items():
            self.calibration.update(module.get_calibration())

        if self.component_bus is not None:
            self.component_bus.connect(self.calibration)
        print("Controller connected")
        
    def setup(self):
        if not self.is_calibrated():
            self.calibrate()

        # setup modules
        for module_name, module in self.modules.items():
            module.setup()

        self.is_connected = True
        self.running = True
        threading.Thread(target=self._state_receiver, daemon=True).start()
        print("Controller setup")

    def disconnect(self):
        """Clean shutdown"""
        self.running = False
        self.action_socket.close()
        self.state_socket.close()
        self.context.term()
        self.is_connected = False

    @profile
    def send_action(self, action: Action):
        """Send control actions to robot"""
        msg = Message()
        msg.controller_timestamp = time.time_ns()
        msg.data = {"action": action.model_dump()}

        if self.clicked_point is not None:
            msg.data["click"] = {
                "point": self.clicked_point,
                "id": self.clicked_id
            }
        self.action_socket.send(msg.serialize())

    def record_frame(self, dataset, action, task):

        # "action": nparray shape 6
        # "observation.state": nparray shape 6
        # "observation.images.front": nparray shape (480, 640, 3)
        # "observation.images.wrist": nparray shape (480, 640, 3)
        # "task": string
        
        frame = {**self.last_observation.to_lerobot_format(), **action.to_lerobot_format()}

        # resize images if necessary
        for key in frame:
            if key.startswith("observation.images."):
                img = frame[key]
                if img.shape[0] != self.cfg.video_res[0] or img.shape[1] != self.cfg.video_res[1]:
                    frame[key] = cv2.resize(img, (self.cfg.video_res[1], self.cfg.video_res[0]), interpolation=cv2.INTER_AREA)
        
        dataset.add_frame(frame, task)
        
    def display_state_loop(self, episode_time_s):
        timestamp = 0
        start_episode_t = time.perf_counter()
        while timestamp < episode_time_s:
            start_loop_t = time.perf_counter()
            self.display_state()
            dt_s = time.perf_counter() - start_loop_t
            time.sleep(max(0, (1.0 / 45) - dt_s))

            timestamp = time.perf_counter() - start_episode_t
            if self.events["exit_early"]:
                break

    def display_state(self):
        pass

    def control_loop_steps(self, dataset=None, episode_time_s=None, single_task=None):
        try:
            timestamp = 0
            start_episode_t = time.perf_counter()
            while timestamp < episode_time_s:
                start_loop_t = time.time_ns()

                # if "reset_policy" in events and events["reset_policy"]:
                #     events["reset_policy"] = False
                #     log_say(f"Resetting policy", True)
                #     self.reset()
                #     while self.start_pos:
                #         time.sleep(0.033)
                #     time.sleep(2)
                #     log_say(f"Policy reset", True)
                
                action = self.control_step(self.last_observation)

                self.send_action(action)

                # Add frame to dataset if recording
                if dataset is not None:
                    self.record_frame(dataset, action, single_task)

                timestamp = time.perf_counter() - start_episode_t
                if self.events["exit_early"]:
                    # events["exit_early"] = False
                    break

                # Wait for next frame
                if self.cfg.fps is not None:
                    dt_s = (time.time_ns() - start_loop_t)/1e9
                    busy_wait((1.0 / self.cfg.fps) - dt_s)
                
                manual_profile("control loop latency", (time.time_ns() - start_loop_t))
                # print(f"Control loop time elapsed: {(time.time_ns() - start_loop_t)/1e9:.6f} seconds")
                
        except KeyboardInterrupt:
            print("\nShutting down teleop client...")


    def control_loop(self, dataset=None, episode_time_s=None, single_task=None):
        """Implements the control loop
        
        Args:
            dataset (Dataset, optional): Dataset to record data. Defaults to None.
            events (dict, optional): Events to trigger. Defaults to None.
            episode_time_s (float, optional): Episode time in seconds. Defaults to None.
            single_task (str, optional): Single task to run. Defaults to None.
        """

        # if events is None:
        #     events = {"exit_early": False}

        if episode_time_s is None:
            episode_time_s = float("inf")

        # spin control loop in a separate thread
        t = threading.Thread(target=self.control_loop_steps, args=(dataset, episode_time_s + 1, single_task), daemon=True)
        t.start()
        
        # run display loop in main thread
        self.display_state_loop(episode_time_s)

        t.join()
        self.events["exit_early"] = False


    @abstractmethod
    def control_step(self, observation: Observation) -> Action:
        """Implements the control step
        
        Args:
            observation (Observation): The last observation received from the robot.
        
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
