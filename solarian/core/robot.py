import zmq
import time
import threading
import numpy as np
import os
import cv2
from abc import ABC, abstractmethod
from typing import Any
from multiprocessing.managers import SharedMemoryManager

from solarian.core.configs import RobotConfig
# from .observation import Observation
from solarian.core.message import Message
from solarian.core.action import Action
from .module import Module
from solarian.core import Observation
from solarian.utils import profile, manual_profile, get_class, busy_wait, StatusBoard, setup_process_logging
from solarian.core.data import ImageData
from solarian.utils.shared_memory.observation_queue import ObservationQueue
from solarian.utils.shared_memory.action_queue import ActionQueue
from solarian.conf import settings
from solarian.utils import get_class

import logging
logger = logging.getLogger("robot")


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

    
    @classmethod
    def module_names(cls):
        return [name for name in cls.__dict__ if isinstance(getattr(cls, name), Module)]
    
    @classmethod
    def validate_obs(cls, data: dict):
        for module_name, module_data in data.items():
            if module_name not in cls.module_names():
                raise ValueError(f"Invalid module name: '{module_name}'")
            module = getattr(cls, module_name)
            module.validate_obs(module_data)

    @classmethod
    def validate_act(cls, data: dict):
        for module_name, module_data in data.items():
            if module_name not in cls.module_names():
                raise ValueError(f"Invalid module name: '{module_name}'")
            module = getattr(cls, module_name)
            module.validate_act(module_data)

    @classmethod
    def __class_getitem__(cls, item):
        return getattr(cls, item)

    def __getitem__(self, item):
        return getattr(self, item)

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
        
        self.current_velocity = None
        self.current_arm_state = None
        
        self.running = False
        self.verbose = True
        self.captured_start_pos = False        
        self.connected = False

        self.modules = {}
        self.observation_queues = {}
        self.action_queues = {}
        self.latest_observation_by_module: dict[str, dict] = {}

        # IMPORTANT: Store the manager so it doesn't get garbage collected
        # (which would clean up all shared memory before subprocess can access it)
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        # Phase 1: Create modules from class-level attributes.
        # This must happen before Observation.zeros() / Action.zeros() so that
        # sensor descriptors (e.g. ImageData on cameras) are configured with
        # the correct shapes from their YAML configs.
        module_configs: dict[str, tuple[type, Any]] = {}
        for name in self.__class__.__dict__:
            val = getattr(self, name, None)
            if isinstance(val, Module):
                module_class = val.__class__
                module_cfg = self.load_module_config(name, module_class)
                module_configs[name] = (module_class, module_cfg)
                self.modules[name] = module_class(module_cfg, name)
                setattr(self, name, self.modules[name])

        # Phase 2: Build example observations/actions (now with correct shapes).
        example_observation = Observation.zeros()
        example_action = Action.zeros()

        # Phase 3: Create shared-memory queues and assign to modules.
        for name, (module_class, module_cfg) in module_configs.items():
            self.observation_queues[name] = ObservationQueue(
                shm_manager=self.shm_manager,
                example_obs=example_observation[name],
                buffer_size=1024,
            )
            self.action_queues[name] = ActionQueue(
                shm_manager=self.shm_manager,
                example_action=example_action[name],
                buffer_size=1024,
            )
            self.modules[name].observation_queue = self.observation_queues[name]
            self.modules[name].action_queue = self.action_queues[name]
        

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
        logger.info(f"Attempting to connect to tcp://{self.cfg.connection.ip}:{self.cfg.connection.state_receiver_port}...")
        test_socket.connect(f"tcp://{self.cfg.connection.ip}:{self.cfg.connection.state_receiver_port}")
        
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
            logger.info("Successfully connected to relay server")
            logger.info("Cleaning up test socket...")
            test_socket.close()

    @profile
    def get_action(self) -> Action | None:
        try:
            bytes = self.action_socket.recv(zmq.NOBLOCK)
            msg = Message.deserialize(bytes)

            if "action" not in msg.data:
                return None

            raw_action = msg.data["action"]
            action = Action(raw_action)
            # latency = (time.time_ns() - action.controller_timestamp)
            # manual_profile("process_action latency", latency)
            return action
        except zmq.Again:
            return None

    @profile
    def send_observation(self, observation: Observation):
        # if hasattr(observation, "wrist_camera") and hasattr(observation.wrist_camera, "image"):
        #     cv2.imshow("Observation", observation.wrist_camera.image.value)
        #     cv2.waitKey(1)
        if observation is None:
            return
        msg = Message()
        msg.robot_timestamp = time.time_ns()
        observation.compress()
        raw_obs = observation.to_dict()
                
        msg.data = {"observation": raw_obs}
        self.state_socket.send(msg.serialize())
    
    def _check_module_health(self) -> bool:
        """Check if any module process has crashed. Returns True if all healthy."""
        for module_name, module in self.modules.items():
            err = module.get_error()
            if err is not None:
                mod_name, tb_str = err
                StatusBoard.stop()
                from rich.console import Console
                console = Console(stderr=True)
                console.print()
                console.print(f"[bold red]Module '{mod_name}' crashed:[/bold red]\n{tb_str}")
                return False
        return True

    def control_loop(self):
        logger.info("Control loop started")
        try:
            while self.running:
                start_loop_t = time.time_ns()

                if not self._check_module_health():
                    break

                self.update()
                action = self.get_action()

                if action is not None:
                    StatusBoard.set("Action", action)
                    self.process_action(action)

                observation = self.get_observation()
                if not observation.is_empty():
                    self.send_observation(observation)
                    # Display a merged snapshot (latest per module), while
                    # transport keeps "new data only" semantics.
                    StatusBoard.set("Observation", self.get_display_observation())

                time.sleep(0.001)
        except KeyboardInterrupt:
            logger.info("\nKeyboardInterrupt: Shutting down robot...")
        except Exception:
            StatusBoard.print_exception(None)
        finally:
            self.running = False
            self.disconnect()

    def connect(self):
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

            logger.info(f"Calibration loaded for module {module_name}")

        self.connected = True
        logger.info("Robot connected")

    def initialize(self):
        # if not self.is_connected():
        #     raise Exception("Robot not connected, cannot initialize, must call connect() first.")
        # initialize modules
        logger.info("Initializing modules")
        for module_name, module in self.modules.items():
            module.initialize()
        logger.info("Robot initialized")

    def on_start(self):
        """Hook method called after observation threads start but before control loop.
        
        Override this in subclasses to start additional threads or perform
        setup that should happen right before the main control loop begins.
        """
        pass

    def start(self):
        self.running = True
        setup_process_logging(logger_name=None)
        StatusBoard()
        try:
            # # Pre-warm hardware drivers that fail after fork().  On macOS,
            # # AVFoundation's +[AVCaptureDevice initialize] cannot complete
            # # in a forked child process, causing cv2.VideoCapture to hang.
            # # Connecting (then disconnecting) here runs +initialize on the
            # # main thread so the forked child inherits the ready state.
            # for module in self.modules.values():
            #     if not module.is_connected:
            #         module.connect()
            #         module.disconnect()

            # Release parent-process hardware connections before forking so that
            # child processes can open their own clean handles (required for e.g.
            # macOS AVFoundation cameras which only allow one consumer).
            for module in self.modules.values():
                module.disconnect()

            for module_name, module in self.modules.items():
                module.start()

            # Hook for subclasses to start additional threads
            self.on_start()

            self.control_loop()
        except KeyboardInterrupt:
            StatusBoard.stop()
            logger.info("\nKeyboardInterrupt: Shutting down robot...")
        except Exception:
            StatusBoard.print_exception(None)
        finally:
            self.running = False
            self.disconnect()
        
    def is_calibrated(self):
        for module_name, module in self.modules.items():
            if not module.is_calibrated():
                return False
        return True

    def is_connected(self):
        return self.connected


    def disconnect(self):
        """Clean shutdown, you can extend this method to do other cleanup"""
        self.running = False
        self.action_socket.close()
        self.state_socket.close()
        for module in self.modules.values():
            module.disconnect()
        self.context.term()
        self.connected = False

    def display(self):
        pass
        # images = {}
        # # add all images in observation
        # for module_name, module_data in self.last_observation.data.items():
        #     for sensor_name, sensor_data in module_data.items():
        #         if sensor_data.get(ValueName.IMAGE) is not None:
        #             images[sensor_name] = sensor_data

        # # Initialize windows once if not already done
        # if not self.windows_initialized:
        #     for name in images.keys():
        #         cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                
        #     self.windows_initialized = True

        # COLOR = (255, 0, 0)

        # for name, image in images.items():
        #     img = image[ValueName.IMAGE]
            
        #     # Check if the image is encoded (bytes) and decode it
        #     if isinstance(img, bytes):
        #         # Decode the JPEG image
        #         print("Decoding image for", name)
        #         img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        #         if img is None:
        #             print(f"Failed to decode image for {name}")
        #             continue
        #     elif not isinstance(img, np.ndarray):
        #         print(f"Unsupported image type for {name}: {type(img)}")
        #         continue                
        #     # Print debug info
        #     # print(f"Image type: {type(img)}, shape: {img.shape if hasattr(img, 'shape') else 'unknown'}")
            
        #     # Convert color only once before display
        #     display_img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        #     # Show frames
        #     cv2.imshow(name, display_img_bgr)
            
        #     # No need to call getWindowProperty every time since windows are created once
        #     key = cv2.waitKey(1) & 0xFF
        #     if key == ord('q'):
        #         return False
            
    
    @profile
    def get_observation(self) -> Observation:
        """Create an observation with current robot state.

        Includes only modules that produced a new sample since the last call.

        Returns:
            Observation: Contains ModuleObs readings organized by module
        """
        obs_data = {}
        for module_name, obs_queue in self.observation_queues.items():
            if not obs_queue.empty():
                module_obs = obs_queue.get().to_dict()
                self.latest_observation_by_module[module_name] = module_obs
                obs_data[module_name] = module_obs
        return Observation(obs_data)

    def get_display_observation(self) -> Observation:
        """Create a display snapshot using the latest observation per module."""
        return Observation(dict(self.latest_observation_by_module))
                    
    @profile
    def process_action(self, action: Action):
        """Process received action
        Args:
            action (Action): Action to process
        """
        # put action into action queues
        for module_name, module_action in action.items():
            self.action_queues[module_name].put(module_action)
            StatusBoard.set(f"{module_name} Action Queue Size", self.action_queues[module_name].qsize())

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
        