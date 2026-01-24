import threading
import torch
import time
import zmq
import cv2
import random
import string
import numpy as np
from abc import ABC, abstractmethod


from proto.common.utils import busy_wait
from proto.robots import Action
from proto.common.configs import ControllerConfig, ValueName, ActionValue
from proto.control import Message
from proto.robots.observation import Observation
from proto.robots.module import Module
from proto.robots.component_bus import ComponentBus
from proto.common.utils import profile, manual_profile, get_class, StatusBoard
from proto.conf import settings


from screeninfo import get_monitors

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
        
        self.is_connected = False
        self.clicked_point = None
        self.clicked_id = None

        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Initialize OpenCV windows once
        self.windows_initialized = []
        
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
        self.last_action = None
        self.obs_none_count = 0
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
                for image_type in [ValueName.IMAGE, ValueName.DEPTH_IMAGE]:
                    if observation.data[module_name][sensor_name].get(image_type, None) is not None:
                        img = observation.data[module_name][sensor_name][image_type]
                        img = cv2.imdecode(
                            np.frombuffer(img, dtype=np.uint8),
                            cv2.IMREAD_COLOR
                        )
                        observation.data[module_name][sensor_name][image_type] = img
        
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
        try:
            while self.running:
                try:
                    self.get_state() 
                except zmq.Again:
                    time.sleep(0.001)
                except KeyboardInterrupt:
                    print("\nShutting down state receiver")
                    break
                # except Exception as e:
                #     print(f"[ERROR] State receiver failed: {e}")
        except KeyboardInterrupt:
            print("\nShutting down state receiver")

    def connect(self):
        """Start receiver threads"""
        self.calibration = {}
        for module_name, module in self.modules.items():
            mod_calib = module.get_calibration()
            if mod_calib is None:
                continue
            self.calibration.update(mod_calib)

        if self.component_bus is not None:
            self.component_bus.connect(self.calibration)

        self.is_connected = True
        print("Controller connected")
        
    def initialize(self):
        # initialize modules
        for module_name, module in self.modules.items():
            module.initialize()
        
        print("Controller initialized")

    def disconnect(self):
        """Clean shutdown"""
        if not self.is_connected:
            return
        self.running = False
        self.action_socket.close()
        self.state_socket.close()
        self.context.term()
        self.is_connected = False

    def stop(self):
        self.running = False

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

    def send_stop_tracking(self):
        msg = Message()
        msg.controller_timestamp = time.time_ns()
        msg.data = {"stop_tracking": True}
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
        
    
    def display_loop(self, episode_time_s):
        timestamp = 0
        start_episode_t = time.perf_counter()
        try:
            while timestamp < episode_time_s and self.running:
                start_loop_t = time.time_ns()
                result = self.display()
                
                # If display returns False, exit the loop
                if result is False:
                    break
                
                timestamp = time.perf_counter() - start_episode_t
                if self.events["exit_early"]:
                    break

                # Wait for next frame
                if self.cfg.fps is not None:
                    dt_s = (time.time_ns() - start_loop_t)/1e9
                    busy_wait((1.0 / self.cfg.fps) - dt_s)
                
                manual_profile("display loop latency", (time.time_ns() - start_loop_t))
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt: Shutting down display loop...")
        except Exception as e:
            print(f"\nError in display loop: {e}")
        finally:
            # Clean up OpenCV windows
            if self.windows_initialized:
                print("Cleaning up OpenCV windows...")
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
            print(f"Clicked at (x={x}, y={y})")
            self.clicked_point = (x, y)
            # generate random alphanumeric id
            self.clicked_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    
    @profile
    def display(self):

        StatusBoard.set("Action", self.last_action)

        if self.last_observation is None:
            return

        images = {}
        # add all images in observation
        for module_name, module_data in self.last_observation.data.items():
            for sensor_name, sensor_data in module_data.items():
                if sensor_data.get(ValueName.IMAGE) is not None or sensor_data.get(ValueName.DEPTH_IMAGE) is not None:
                    images[sensor_name] = sensor_data
                
        # Initialize windows once if not already done
        if not self.windows_initialized:
            for name, image in images.items():
                for img_type in [ValueName.IMAGE, ValueName.DEPTH_IMAGE]:
                    if not img_type in image:
                        continue
                    window_name = f"{name} {img_type}"

                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    # Get the resolution of the image
                    height, width, _ = image[img_type].shape
                    # Set the size of the window to the resolution of the image
                    cv2.resizeWindow(window_name, width, height)
                    cv2.setMouseCallback(window_name, self.mouse_callback)
                    cv2.moveWindow(window_name, sx + (sw - width)//2, sy + (sh - height)//2)
                    self.windows_initialized.append(window_name)
            

        COLOR = (255, 0, 0)

        # board = StatusBoard()
        # board.set("Observation", self.last_observation)
        StatusBoard.set("Observation", self.last_observation)
        

        for name, image in images.items():

            for img_type in [ValueName.IMAGE, ValueName.DEPTH_IMAGE]:
                if not img_type in image:
                    continue

                img = image[img_type]

                # print("Image: ", image)
                
                # Only modify the image if bounding box exists
                if image.get(ValueName.BOUNDING_BOX, None) is not None and image.get(ValueName.BOUNDING_BOX)[0] != -1:
                    # Create a copy only when drawing on the image
                    display_img = img.copy()
                    # print("bounding box", image[ValueName.BOUNDING_BOX])
                    x1, y1, x2, y2, depth = image[ValueName.BOUNDING_BOX]
                    cv2.rectangle(display_img,
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                COLOR, 2)
                    cv2.putText(display_img, f"Depth: {depth:.3f}", 
                                (int(x1), int(y1) + 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.9, (0, 255, 0), 2)
                else:
                    display_img = img
                    
                display_img_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
                # Convert color only once before display
                # if img_type == ValueName.IMAGE:
                #     display_img_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
                # else:
                #     display_img_bgr = colorize_depth(display_img, True)
                
                # Show frames
                cv2.imshow(name + " " + str(img_type), display_img_bgr)
            
            # No need to call getWindowProperty every time since windows are created once
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False

    def control_loop(self, dataset=None, episode_time_s=None, single_task=None):
        try:
            timestamp = 0
            start_episode_t = time.perf_counter()
            self.send_stop_tracking()
            while timestamp < episode_time_s:
                start_loop_t = time.time_ns()
                timestamp = time.perf_counter() - start_episode_t

                

                # if "reset_policy" in events and events["reset_policy"]:
                #     events["reset_policy"] = False
                #     log_say(f"Resetting policy", True)
                #     self.reset()
                #     while self.start_pos:
                #         time.sleep(0.033)
                #     time.sleep(2)
                #     log_say(f"Policy reset", True)

                if True or self.last_observation is not None:
                    
                
                    self.last_action = self.control_step(self.last_observation)

                    if self.last_action == ActionValue.STOP:
                        self.running = False
                        break

                    self.send_action(self.last_action)

                    # Add frame to dataset if recording
                    if dataset is not None and self.last_observation is not None:
                        self.record_frame(dataset, self.last_action, single_task)
                else:
                    # self.obs_none_count += 1
                    # if self.obs_none_count > 1:
                    #     raise ValueError("Last observation is None")
                    print("Last observation is None")

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
            print("\nShutting down control loop...")
        finally:
            self.running = False


    def on_start(self):
        """Hook method called after control threads start but before display loop.
        
        Override this in subclasses to start additional threads or perform
        setup that should happen right before the main display loop begins.
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

        # if events is None:
        #     events = {"exit_early": False}

        if episode_time_s is None:
            episode_time_s = float("inf")

        self.running = True
        print("Starting state receiver")
        threading.Thread(target=self._state_receiver, daemon=True).start()

        # spin control loop in a separate thread
        print("Starting control loop")
        t = threading.Thread(target=self.control_loop, args=(dataset, episode_time_s + 1, single_task), daemon=True)
        t.start()
        
        # Hook for subclasses to start additional threads
        self.on_start()
        
        # run display loop in main thread
        print("Starting display loop")
        self.display_loop(episode_time_s)

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
