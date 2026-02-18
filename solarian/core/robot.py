import zmq
import time
import threading
import cv2
import numpy as np
import os
from abc import ABC, abstractmethod
from multiprocessing.managers import SharedMemoryManager

from solarian.core.configs import ValueName, RobotConfig
# from .observation import Observation
from solarian.core.message import Message
from solarian.core.action import Action
from .module import Module
from .module_bus import ModuleBus
from solarian.core import Observation
from solarian.utils import profile, manual_profile, get_class, busy_wait, StatusBoard
from solarian.core.data import DataKind
from solarian.utils.shared_memory.observation_queue import ObservationQueue
from solarian.utils.shared_memory.action_queue import ActionQueue
from solarian.conf import settings
from solarian.utils import get_class


class Robot(ABC):

    _instance = None

    # Override in subclasses to use a custom config class (must extend ModuleConfig)
    config_class: type = RobotConfig
    

    # def __init_subclass__(cls):
    #     super().__init_subclass__()

    #     # 1) enforce required fields + types
    #     req = getattr(Robot, "__annotations__", {})
    #     for field, expected_type in req.items():
    #         if not hasattr(cls, field):
    #             raise TypeError(f"{cls.__name__} must declare '{field}: {expected_type}'")
    #         value = getattr(cls, field)
    #         if not isinstance(value, expected_type):
    #             raise TypeError(
    #                 f"{cls.__name__}.{field} must be {expected_type.__name__}, "
    #                 f"got {type(value).__name__}"
                # )

        # # 2) enforce at least one Module
        # modules = {
        #     name: value for name, value in cls.__dict__.items()
        #     if isinstance(value, Module)
        # }
        # if not modules:
        #     raise TypeError(f"{cls.__name__} must declare at least one Module")


        

    def __new__(cls, *args, **kwargs):
        if Robot._instance is None:
            Robot._instance = super().__new__(cls)
        return Robot._instance

    """Base class with common robot functionality"""
    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        # self.cfg.validate_with_robot_schema(settings.ROBOT_SCHEMA)
        self.context = zmq.Context()
        
        # action receiver (PULL)
        self.action_socket = self.context.socket(zmq.PULL)
        self.action_socket.setsockopt(zmq.CONFLATE, 1)
        self.action_socket.connect(f"tcp://{self.cfg.connection.ip}:{self.cfg.connection.action_sender_port}")
        
        # State publisher (PUB)
        self.state_socket = self.context.socket(zmq.PUB)
        self.state_socket.setsockopt(zmq.CONFLATE, 1)
        self.state_socket.connect(f"tcp://{self.cfg.connection.ip}:{self.cfg.connection.state_receiver_port}")

        #self.control_lock = threading.Lock()
        self.observation_lock = threading.Lock()
        self.last_observation = None
        self.windows_initialized = False

        # Verify connection
        # self._verify_connection()
        
        self.current_action = None
        self.current_velocity = None
        self.current_arm_state = None
        
        self.running = False
        self.verbose = True
        self.captured_start_pos = False        
        self.connected = False

        # create component bus
        self.component_bus = ModuleBus()

        self.modules = {}
        self.observation_queues = {}
        self.action_queues = {}

        example_observation = get_class(settings.OBSERVATION_CLASS).zeros()
        example_action = get_class(settings.ACTION_CLASS).zeros()

        # IMPORTANT: Store the manager so it doesn't get garbage collected
        # (which would clean up all shared memory before subprocess can access it)
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        # Create modules from class-level attributes
        for name in self.__class__.__dict__:
            val = getattr(self, name, None)
            if isinstance(val, Module):
                module_class = val.__class__
                module_cfg = self.load_module_config(name, module_class)

                # create module's observation queue
                self.observation_queues[name] = ObservationQueue(
                    shm_manager=self.shm_manager,
                    example_obs=example_observation[name],
                    buffer_size=1024,
                )
                # create module's action queue
                self.action_queues[name] = ActionQueue(
                    shm_manager=self.shm_manager,
                    example_action=example_action[name],
                    buffer_size=1024,
                )
                # create module
                self.modules[name] = module_class(module_cfg, name, self.component_bus, self.observation_queues[name], self.action_queues[name])
                # override class-level attribute with module instance
                setattr(self, name, self.modules[name])
        

    def load_module_config(self, module_name: str, module_class: type):
        """Load module config from YAML using the module's own config class"""
        from solarian.utils.configs import load_config_yaml
        conf_path = os.environ.get("SOLARIAN_CONF_PATH", os.path.join(os.getcwd(), "conf"))
        yaml_path = os.path.join(conf_path, f"{module_name}.yaml")
        module_cfg = load_config_yaml(yaml_path, module_class.config_class)
        return module_cfg
    
    
    def _verify_connection(self, timeout=5000):
        """Verify that we can connect to the relay server"""
        print(f"Verifying connection to relay server at {self.cfg.connection.ip}...")
        
        # Create a temporary PUSH socket to test connection
        test_socket = self.context.socket(zmq.PUSH)
        test_socket.setsockopt(zmq.LINGER, 0)
        print(f"Attempting to connect to tcp://{self.cfg.connection.ip}:{self.cfg.connection.state_receiver_port}...")
        test_socket.connect(f"tcp://{self.cfg.connection.ip}:{self.cfg.connection.state_receiver_port}")
        
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
            print("Successfully connected to relay server")
            print("Cleaning up test socket...")
            test_socket.close()

    @profile
    def get_action(self):
        try:
            bytes = self.action_socket.recv(zmq.NOBLOCK)
            msg = Message.deserialize(bytes)

            if "action" not in msg.data:
                return

            raw_action = msg.data["action"]
            action = get_class(settings.ACTION_CLASS)(raw_action)
            self.current_action = action
            
            latency = (time.time_ns() - msg.controller_timestamp)
            manual_profile("process_action latency", latency)
            # with self.observation_lock:
            self.process_action(action)
            
        except zmq.Again:
            pass

    # def _action_receiver(self):
    #     """Thread for receiving actions"""
    #     while self.running:
    #         try:
    #             bytes = self.action_socket.recv(zmq.NOBLOCK)
    #             msg = Message.deserialize(bytes)
    #             action = Action.model_validate(msg.data["action"])
    #             self.current_action = action
    #             if self.verbose:
    #                 latency = (time.time_ns() - msg.controller_timestamp)/1e6
    #                 # print(f"Received action: {action}")
    #                 print(f"action latency: {latency:.2f}ms")
    #             with self.observation_lock:
    #                 self.process_action(action)
    #             # print("Action processed")
    #         except zmq.Again:
    #         # except Exception as e:
    #             # print(f"Error in action receiver: {e}")
    #             time.sleep(0.001)

    @profile
    def send_observation(self, observation: Observation):
        if observation is None:
            return
        msg = Message()
        msg.robot_timestamp = time.time_ns()

        raw_obs = {}
        obs_class = get_class(settings.OBSERVATION_CLASS)

        # Encode images as JPEG
        for module_name in observation.keys():
            raw_obs[module_name] = {}
            for sensor_name in observation[module_name].keys():
                raw_obs[module_name][sensor_name] = {}
                for data_name in observation[module_name][sensor_name].keys():
                    
                    if obs_class[module_name][sensor_name][data_name].kind == DataKind.IMAGE:
                        img = observation[module_name][sensor_name][data_name]
                        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.cfg.img_compression_quality]
                        encoded_img = cv2.imencode('.jpeg', img, encode_params)[1].tobytes()
                        raw_obs[module_name][sensor_name][data_name] = encoded_img
                        # print image size
                        # print("-----------------------------------------------")
                        # print(f"Image size: {len(encoded_img)/1024:.2f} kb")
                        # print("Compression time: ", time.time() - start_time)
                        # print("-----------------------------------------------")
                    else:
                        raw_obs[module_name][sensor_name][data_name] = observation[module_name][sensor_name][data_name]
                
        msg.data = {"observation": raw_obs}
        self.state_socket.send(msg.serialize())

    # def _state_publisher(self):
    #     """Thread for capturing and sending state observation"""
    #     print("State publisher started")
    #     while self.running:
    #         # Get and send observation
            

    #         self.get_action()

    #         # print("Getting observation in robot.py")
    #         with self.observation_lock:
    #             observation = self.get_observation()
    #         # print("Observation:", observation)

    #         self.send_observation(observation)

    #         # print("Observation sent", observation)
    #         time.sleep(0.01)

    
    def control_loop(self):
        print("Control loop started")
        try:
            while self.running:
                start_loop_t = time.time_ns()
                # Get and send observation

                self.update()
                self.get_action()
                observation = self.get_observation()

                # board = StatusBoard()
                # board.set("Observation", observation)
                StatusBoard.set("Observation", observation)
                StatusBoard.set("Action", self.current_action)
                # print("Observation:", observation)
                # if 'arm' in observation.data and 'wrist' in observation.data['arm'] and 'image' in observation.data['arm']['wrist'] and type(observation.data['arm']['wrist']['image']) is np.ndarray:
                #     print("Image is numpy array")
                # else:
                #     print("Image is not numpy array")
                #     exit(1)

                if not observation.is_empty():
                    # print("Sending observation:", observation)
                    self.send_observation(observation)

                # # Wait for next frame
                # if self.cfg.fps is not None:
                #     dt_s = (time.time_ns() - start_loop_t)/1e9
                #     busy_wait((1.0 / self.cfg.fps) - dt_s)

                # print("Observation sent", observation)
                time.sleep(0.001)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt: Shutting down robot...")
            self.disconnect()

    def connect(self):
        self.calibration = {}
        # connect to hardware without calibration
        self.component_bus.connect(self.calibration)

        for module_name, module in self.modules.items():
            # module.load_config()
            module.load_calibration()

            # module requires calibration but no calibration file was found
            if module.calibration is None:
                # will start calibration process
                module.calibrate()
            
            # module requires calibration but not yet written to hardware
            if not module.is_calibrated():
                module.write_calibration()
                
            # get module calibration grouped by port
            mod_calib = module.get_calibration()
            if mod_calib is None:
                continue
            self.calibration.update(mod_calib)

        # disconnect and reconnect to hardware with calibration
        self.component_bus.disconnect()
        self.component_bus.connect(self.calibration)
        
        self.connected = True
        print("Robot connected")

    def initialize(self):
        if not self.is_connected():
            raise Exception("Robot not connected, cannot initialize, must call connect() first.")
        # initialize modules
        print("Initializing modules")
        for module_name, module in self.modules.items():
            module.initialize()
        print("Robot initialized")

    def on_start(self):
        """Hook method called after observation threads start but before control loop.
        
        Override this in subclasses to start additional threads or perform
        setup that should happen right before the main control loop begins.
        """
        pass

    def start(self):
        self.running = True
        for module_name, module in self.modules.items():
            module.start()

        # Hook for subclasses to start additional threads
        self.on_start()

        self.control_loop()
        
    def is_calibrated(self):
        for module_name, module in self.modules.items():
            if not module.is_calibrated():
                return False
        return True
        # return self.component_bus.is_calibrated()

    def is_connected(self):
        return self.connected


    def disconnect(self):
        """Clean shutdown, you can extend this method to do other cleanup"""
        self.running = False
        self.action_socket.close()
        self.state_socket.close()
        if self.component_bus is not None:
            self.component_bus.disconnect()
        self.context.term()
        self.connected = False

    def display(self):
        images = {}
        # add all images in observation
        for module_name, module_data in self.last_observation.data.items():
            for sensor_name, sensor_data in module_data.items():
                if sensor_data.get(ValueName.IMAGE) is not None:
                    images[sensor_name] = sensor_data

        # Initialize windows once if not already done
        if not self.windows_initialized:
            for name in images.keys():
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                
            self.windows_initialized = True

        COLOR = (255, 0, 0)

        for name, image in images.items():
            img = image[ValueName.IMAGE]
            
            # Check if the image is encoded (bytes) and decode it
            if isinstance(img, bytes):
                # Decode the JPEG image
                print("Decoding image for", name)
                img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"Failed to decode image for {name}")
                    continue
            elif not isinstance(img, np.ndarray):
                print(f"Unsupported image type for {name}: {type(img)}")
                continue                
            # Print debug info
            # print(f"Image type: {type(img)}, shape: {img.shape if hasattr(img, 'shape') else 'unknown'}")
            
            # Convert color only once before display
            display_img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Show frames
            cv2.imshow(name, display_img_bgr)
            
            # No need to call getWindowProperty every time since windows are created once
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
            
    
    def get_observation(self) -> Observation:
        """Create an observation with current robot state.
        Returns:
            Observation: Contains camera images and actuator readings organized by groups
        """
        # creates empty observation
        obs = get_class(settings.OBSERVATION_CLASS)()

        for module_name, obs_queue in self.observation_queues.items():
            if not obs_queue.empty():
                module_observation = obs_queue.get()
                obs[module_name] = module_observation
            else:
                obs[module_name] = None
        return obs
                    
    # @profile
    def process_action(self, action: Action):
        """Process received action
        Args:
            action (Action): Action to process
        """
        # put action into action queues
        for module_name in action.data.keys():
            self.action_queues[module_name].put(action[module_name])

    def update(self):
        """Update robot state"""
        pass

    def __str__(self):
        return f"{self.__class__.__name__}"

    def calibrate(self):
        for module_name, module in self.modules.items():
            module.calibrate()

    def write_calibration(self):
        for module_name, module in self.modules.items():
            module.write_calibration()
        