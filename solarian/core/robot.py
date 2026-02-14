import zmq
import time
import threading
import cv2
import numpy as np
from abc import ABC, abstractmethod
from multiprocessing.managers import SharedMemoryManager

from solarian.core.configs import ValueName, RobotConfig
# from .observation import Observation
from solarian.core.message import Message
from solarian.core.action import Action
from .module import Module
from .component_bus import ComponentBus
from solarian.core import Observation
from solarian.utils import profile, manual_profile, get_class, busy_wait, StatusBoard
from solarian.utils.shared_memory.observation_ring_buffer import ObservationRingBuffer
from solarian.utils.shared_memory.action_queue import ActionQueue

class Robot(ABC):

    _instance = None

    # Override in subclasses to use a custom config class (must extend ModuleConfig)
    config_class: type = RobotConfig
    

    def __init_subclass__(cls):
        super().__init_subclass__()

        # 1) enforce required fields + types
        req = getattr(Robot, "__annotations__", {})
        for field, expected_type in req.items():
            if not hasattr(cls, field):
                raise TypeError(f"{cls.__name__} must declare '{field}: {expected_type}'")
            value = getattr(cls, field)
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"{cls.__name__}.{field} must be {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

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

    @property
    def modules(self) -> dict[str, Module]:
        """Return all Module instances on this robot (both class-level and instance-level)."""
        result = {}
        # Check class-level attributes (descriptor-style modules)
        for name in self.__class__.__dict__:
            val = getattr(self, name, None)
            if isinstance(val, Module):
                result[name] = val
        # Check instance-level attributes (modules created from config)
        for name, val in self.__dict__.items():
            if isinstance(val, Module) and name not in result:
                result[name] = val
        return result


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

        # Create module instances from class annotations and separate YAML configs
        self._create_modules_from_annotations()

    def _create_modules_from_annotations(self):
        """
        Create module instances from the robot's class annotations and their
        corresponding YAML config files.
        
        For each annotation like `arm: UR5ArmModule` on the Robot subclass,
        loads `conf/arm.yaml`, parses it as a ModuleConfig, and creates a
        UR5ArmModule(cfg) instance.
        """
        from solarian.utils.configs import load_module_configs_for_robot
        
        module_configs = load_module_configs_for_robot(type(self))
        
        for name, (module_type, module_cfg) in module_configs.items():
            module = module_type(module_cfg)
            module.name = name  # Set name (normally done by __set_name__ for class-level descriptors)
            setattr(self, name, module)

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
    
    # def create_components(self):
    #     self.component_bus = ComponentBus()
    #     for module_name, module_cfg in self.cfg.modules.items():
    #         if module_name in settings.MODULE_CLASSES:
    #             module = get_class(settings.MODULE_CLASSES[module_name])(module_cfg, module_name, self.component_bus)
    #             self.modules[module_name] = module

    @profile
    def get_action(self):
        try:
            bytes = self.action_socket.recv(zmq.NOBLOCK)
            msg = Message.deserialize(bytes)

            if "action" not in msg.data:
                return
            action = Action.model_validate(msg.data["action"])
            self.current_action = action
            
            latency = (time.time_ns() - msg.controller_timestamp)
            manual_profile("process_action latency", latency)
            with self.observation_lock:
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
        msg = Message()
        msg.robot_timestamp = time.time_ns()

        # Encode images as JPEG
        for module_name in observation.data.keys():
            for sensor_name in observation.data[module_name].keys():
                if observation.data[module_name][sensor_name].get(ValueName.IMAGE) is not None:
                    img = observation.data[module_name][sensor_name][ValueName.IMAGE]

                    if not isinstance(img, np.ndarray):
                        # image was already encoded
                        continue
                    
                    start_time = time.time()
                    # encode_params = [cv2.IMWRITE_WEBP_QUALITY, self.cfg.img_compression_quality]
                    # encoded_img = cv2.imencode('.webp', img, encode_params)[1].tobytes()
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.cfg.img_compression_quality]
                    encoded_img = cv2.imencode('.jpeg', img, encode_params)[1].tobytes()
                    # encoded_img = img.tobytes()

                    # print image size
                    # print("-----------------------------------------------")
                    # print(f"Image size: {len(encoded_img)/1024:.2f} kb")
                    # print("Compression time: ", time.time() - start_time)
                    # print("-----------------------------------------------")

                    observation.data[module_name][sensor_name][ValueName.IMAGE] = encoded_img
                    # observation.data[module_name][sensor_name][ValueName.IMAGE] = img

        msg.data = {"observation": observation.model_dump()}
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

                if observation is not None:
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
        for module_name, module in self.modules.items():
            mod_calib = module.get_calibration()
            if mod_calib is None:
                continue
            self.calibration.update(mod_calib)

        if self.component_bus is not None:
            self.component_bus.connect(self.calibration)
        print("Robot connected")

    def initialize(self):
        from solarian.conf import settings

        self.component_bus = ComponentBus()

        example_observation = settings.PROJECT.observation_class()
        example_action = settings.PROJECT.action_class()

        # IMPORTANT: Store the manager so it doesn't get garbage collected
        # (which would clean up all shared memory before subprocess can access it)
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        self.observation_buffers = {}
        self.action_queues = {}
        # create one observation buffer for each module
        for module_name, module in self.modules.items():
            self.observation_buffers[module_name] = ObservationRingBuffer.create_from_example(
                shm_manager=self.shm_manager,
                example_observation=example_observation[module_name],
                get_time_budget=0.2,
            )
            self.action_queues[module_name] = ActionQueue.create_from_example(
                shm_manager=self.shm_manager,
                example_action=example_action[module_name],
                buffer_size=1024,
            )

        # initialize modules
        print("Initializing modules")
        for module_name, module in self.modules.items():
            module.initialize(self.component_bus, self.observation_buffers[module_name], self.action_queues[module_name])
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

    def disconnect(self):
        """Clean shutdown, you can extend this method to do other cleanup"""
        self.running = False
        self.action_socket.close()
        self.state_socket.close()
        if self.component_bus is not None:
            self.component_bus.disconnect()
        self.context.term()
        

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

    # def get_module_observation_loop(self, module_name: str):
    #     """Thread for capturing and sending module observation"""
    #     print(f"Getting observation for module: {module_name}")
    #     try:
    #         while self.running:
    #             point_0 = time.time_ns()
    #             with self.observation_lock:
    #                 try:
    #                     module_observation = self.modules[module_name].get_observation()
    #                     # print(f"Module observation for {module_name}:", module_observation)
    #                 except ConnectionError as e:
    #                     print(f"Failed to get observation for module {module_name}: {e}")
    #                     continue

    #                 point_1 = time.time_ns()
    #                 if self.last_observation is None:
    #                     self.last_observation = module_observation
    #                     # print("Observation1:", self.last_observation)
    #                 else:
    #                     data = self.last_observation.data.copy()
    #                     data.update(module_observation.data)
    #                     self.last_observation = Observation(data=data)
    #                     # print("Observation2:", self.last_observation)

    #                 # if type(self.last_observation.data['arm']['wrist']['image']) is np.ndarray:
    #                 #     print("Image is numpy array")
    #                 # else:
    #                 #     print("Image is not numpy array")
    #                 #     exit(1)

    #                 point_2 = time.time_ns()
    #                 # self.display()
    #                 # point_3 = time.time_ns()
    #             time.sleep(0.001)
    #             manual_profile(f"get_module_observation {module_name} 1", (point_1 - point_0))
    #             # manual_profile(f"get_module_observation {module_name} 2", (point_2 - point_1))
    #             # manual_profile(f"get_module_observation {module_name} 3", (point_3 - point_2))
    #     except KeyboardInterrupt:
    #         print("\nKeyboardInterrupt: Shutting down robot...")

    # def get_module_observation(self, module_name: str) -> Observation:
    #     # point_0 = time.time_ns()
    #     with self.observation_lock:
    #         module_observation = self.modules[module_name].get_observation()
    #         # print(f"Module observation for {module_name}:", module_observation)
    #         point_1 = time.time_ns()
    #         if self.last_observation is None:
    #             self.last_observation = module_observation
    #             # print("Observation1:", self.last_observation)
    #         else:
    #             data = self.last_observation.data.copy()
    #             data.update(module_observation.data)
    #             self.last_observation = Observation(data=data)
    #             # print("Observation2:", self.last_observation)

    #         # if type(self.last_observation.data['arm']['wrist']['image']) is np.ndarray:
    #         #     print("Image is numpy array")
    #         # else:
    #         #     print("Image is not numpy array")
    #         #     exit(1)

    #         point_2 = time.time_ns()
    #         # self.display()
    #         # point_3 = time.time_ns()
    #     time.sleep(0.001)
    #     manual_profile(f"get_module_observation {module_name} 1", (point_1 - point_0))
    #     # manual_profile(f"get_module_observation {module_name} 2", (point_2 - point_1))
    #     # manual_profile(f"get_module_observation {module_name} 3", (point_3 - point_2))
            
    
    def get_observation(self) -> Observation:
        """Create an observation with current robot state.
        Returns:
            Observation: Contains camera images and actuator readings organized by groups
        """
        from solarian.conf import settings

        # if self.last_observation is None:
        #     return None
        # # return a copy of the observation
        # # return self.last_observation.model_copy()       
        # return self.last_observation

        data = {}
        obs = settings.PROJECT.observation_class()
        for module_name, module in self.modules.items():
            module_observation = module.get_observation()
            data.update(module_observation.data)
        return Observation(data=data)
                    
    # @profile
    def process_action(self, action: Action):
        """Process received action
        Args:
            action (Action): Action to process
        """
        for module_name, module in self.modules.items():
            module.process_action(action)

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
        