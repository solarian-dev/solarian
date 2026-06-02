import zmq
import time
import threading
import os
from abc import ABC, abstractmethod
from typing import Any
from multiprocessing.managers import SharedMemoryManager

from solarian.core.configs import RobotConfig
from solarian.core.message import Message
from solarian.core.action import Action
from .module import Module
from .bus_server import BusServer, BusClient
from solarian.core import Observation
from solarian.utils import profile, manual_profile, get_class, busy_wait, StatusBoard, setup_process_logging
from solarian.utils.profiling import merge_stats as merge_profiling_stats
from solarian.core.data import ImageData
from solarian.utils.shared_memory.observation_ring_buffer import ObservationRingBuffer
from solarian.utils.shared_memory.action_queue import ModuleActionQueue
from solarian.utils.shared_memory.skill_status_buffer import SkillStatusBuffer
from solarian.conf import settings
from solarian.core.action import ActionType, ModuleAction

import logging
logger = logging.getLogger("robot")


class Robot(ABC):

    _instance = None

    # Override in subclasses to use a custom config class (must extend ModuleConfig)
    config_class: type = RobotConfig
    
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
        self.last_observation = Observation({})
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
        self._bus_servers: list[BusServer] = []
        self.observation_buffers = dict[str, ObservationRingBuffer]()
        self.action_queues = {}

        # IMPORTANT: Store the manager so it doesn't get garbage collected
        # (which would clean up all shared memory before subprocess can access it)
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        # Shared-memory buffer skills push status transitions into; the
        # observation-sender thread drains it and forwards records to the
        # controller inside the regular state message.
        self.skills_status_buffer = SkillStatusBuffer(
            shm_manager=self.shm_manager,
            get_max_k=64,
            get_time_budget=0.2,
            put_desired_frequency=60,
        )

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

        # Phase 3: Create shared-memory buffers and queues and assign to modules.
        for name, (module_class, module_cfg) in module_configs.items():
            module_obs = example_observation[name]
            module_act = example_action[name]
            assert module_obs is not None, f"No observation defined for module '{name}'"
            assert module_act is not None, f"No action defined for module '{name}'"

            self.observation_buffers[name] = ObservationRingBuffer.create_from_example(
                shm_manager=self.shm_manager,
                example_observation=module_obs,
                get_time_budget=0.2,
                put_desired_frequency=200
            )
            self.action_queues[name] = ModuleActionQueue(
                shm_manager=self.shm_manager,
                example_action=module_act,
                buffer_size=1024,
            )
            self.modules[name].observation_buffer = self.observation_buffers[name]
            self.modules[name].action_queue = self.action_queues[name]

        # Build action handler registry from @action_handler decorated methods
        self.action_handlers: dict[int, str] = {}
        for attr_name in dir(self):
            method = getattr(type(self), attr_name, None)
            if callable(method) and hasattr(method, '_handles_action'):
                self.action_handlers[getattr(method, '_handles_action')] = attr_name
        

    def load_module_config(self, module_name: str, module_class: type):
        """Load module config from YAML using the module's own config class"""
        from solarian.utils.configs import load_config_yaml
        conf_path = os.environ.get("SOLARIAN_CONF_PATH", os.path.join(os.getcwd(), "conf"))
        yaml_path = os.path.join(conf_path, f"{module_name}.yaml")
        module_cfg = load_config_yaml(yaml_path, module_class.config_class)
        return module_cfg

    def _detect_shared_ports(self) -> dict[str, list[str]]:
        """Return ports that are referenced by two or more modules.

        Returns:
            ``{port: [module_name, ...]}`` for every port used by >1 module.
        """
        port_to_modules: dict[str, list[str]] = {}
        for name, module in self.modules.items():
            for port in module.module_bus.components_by_port:
                port_to_modules.setdefault(port, []).append(name)
        return {p: names for p, names in port_to_modules.items() if len(names) > 1}

    def _start_shared_bus_servers(self):
        """For every port shared by multiple modules, start a ``BusServer``
        and inject ``BusClient`` proxies into the modules' ``ModuleBus``.
        """
        from multiprocessing.connection import Pipe

        shared_ports = self._detect_shared_ports()
        if not shared_ports:
            return

        for port, module_names in shared_ports.items():
            # Collect every component on this port across all modules
            all_components: dict = {}
            all_calibration: dict = {}
            bus_class = None

            for mod_name in module_names:
                module = self.modules[mod_name]
                components = module.module_bus.components_by_port[port]
                all_components.update(components)

                if bus_class is None:
                    bus_class = next(iter(components.values())).bus_class

                calibration = module.get_calibration() or {}
                if port in calibration:
                    all_calibration.update(calibration[port])

            assert bus_class is not None

            # One Pipe per module: (server_end, client_end)
            pipes: dict[str, tuple] = {}
            for mod_name in module_names:
                server_conn, client_conn = Pipe()
                pipes[mod_name] = (server_conn, client_conn)

            server = BusServer(
                port=port,
                bus_class=bus_class,
                components=all_components,
                calibration=all_calibration or None,
                server_conns={n: p[0] for n, p in pipes.items()},
            )
            server.start()
            server.ready_event.wait(timeout=30)

            if not server.is_alive():
                err = server.get_error()
                if err:
                    _name, tb_str = err
                    raise RuntimeError(f"BusServer on {port} died during startup:\n{tb_str}")
                raise RuntimeError(f"BusServer on {port} died during startup (check stderr)")

            self._bus_servers.append(server)

            # Hand each module a BusClient proxy for this port
            for mod_name in module_names:
                client = BusClient(
                    client_conn=pipes[mod_name][1],
                    port=port,
                    components=self.modules[mod_name].module_bus.components_by_port[port],
                )
                self.modules[mod_name].module_bus.override_port_bus(port, client)

            # logger.info(f"BusServer started on {port} serving modules: {module_names}")

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
            action = Action(action_type=ActionType(raw_action["action_type"]), data=raw_action["data"])
            # latency = (time.time_ns() - action.controller_timestamp)
            # manual_profile("process_action latency", latency)
            return action
        except zmq.Again:
            return None

    @profile
    def send_observation(self, observation: Observation):
        if observation is None:
            return
        msg = Message()
        msg.robot_timestamp = time.time_ns()
        observation.compress()
        raw_obs = observation.to_dict()
        msg.data = {"observation": raw_obs}

        # Forward any pending skill status transitions to the controller so
        # that the think process can react to them.  ``drain`` advances a
        # local cursor so records are only sent once even though this
        # method runs at ~200 Hz from ``_observation_sender``.
        if self.skills_status_buffer is not None:
            records = self.skills_status_buffer.drain()
            if records:
                msg.data["skill_status"] = [
                    {
                        "skill_id": r.skill_id,
                        "status": r.status,
                        "timestamp": r.timestamp,
                    }
                    for r in records
                ]

        self.state_socket.send(msg.serialize())
    
    def _check_module_health(self):
        """Raise if any module process or bus server has crashed."""
        for module_name, module in self.modules.items():
            err = module.get_error()
            if err is not None:
                mod_name, tb_str = err
                raise RuntimeError(f"Module '{mod_name}' crashed:\n{tb_str}")
        for srv in self._bus_servers:
            err = srv.get_error()
            if err is not None:
                name, tb_str = err
                raise RuntimeError(f"BusServer '{name}' crashed:\n{tb_str}")

    def _observation_sender(self):
        """Send observations to the server in a dedicated thread.

        Decoupled from the action loop so that image compression on slower
        hardware (e.g. Raspberry Pi) does not block action delivery.
        """
        while self.running:
            try:
                observation = self.get_observation()
                self.send_observation(observation)
                StatusBoard.set("Observation", observation)
            except Exception as e:
                logger.exception(f"Observation sender error: {e}")
            time.sleep(0.005)

    def control_loop(self):
        logger.info("Control loop started")

        # Drop any action that was already sitting in the relay's CONFLATE
        # buffer when we connected.  Without this, a stale actions from a
        # previous controller session would be
        # dispatched as the very first action of the new session, silently
        # launching skills the operator did not request.
        drained = 0
        while True:
            stale = self.get_action()
            if stale is None:
                break
            drained += 1
        if drained:
            logger.warning(
                "Discarded %d stale action(s) at control-loop start", drained
            )

        poller = zmq.Poller()
        poller.register(self.action_socket, zmq.POLLIN)

        while self.running:
            self._check_module_health()
            self.update()

            if poller.poll(timeout=1):
                action = self.get_action()
                if action is not None:
                    StatusBoard.set("Action", action)
                    self.process_action(action)
            # actions that are generated by a skill running on the robot are pushed to the action queue directly by the skill itself

    def connect(self):
        for module_name, module in self.modules.items():
            module.load_calibration()

            if module.cfg.calibration_file is None:
                logger.info(f"No calibration required for module {module_name}")
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
        self._bus_servers = []
        setup_process_logging(logger_name=None)

        StatusBoard()
        status_mp_q = StatusBoard.get_mp_queue()
        
        try:
            # Release parent-process hardware connections before forking so that
            # child processes can open their own clean handles (required for e.g.
            # macOS AVFoundation cameras which only allow one consumer).
            for module in self.modules.values():
                module._status_mp_q = status_mp_q
                module.disconnect()

            # For every port shared by 2+ modules, spin up a BusServer that
            # owns the real serial connection and hand each module a BusClient
            # proxy.  Modules keep their own processes; they just talk to the
            # server instead of opening the port themselves.
            self._start_shared_bus_servers()

            for module_name, module in self.modules.items():
                module.start()

            # Hook for subclasses to start additional threads
            self.on_start()

            threading.Thread(target=self._observation_sender, daemon=True).start()
            self.control_loop()
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
            if module.is_alive():
                module.terminate()
                module.join(timeout=5)
            try:
                child_stats = module._profiling_queue.get_nowait()
                merge_profiling_stats(child_stats)
            except Exception:
                pass
            module.disconnect()
        for srv in getattr(self, '_bus_servers', []):
            if srv.is_alive():
                srv.terminate()
                srv.join(timeout=5)
        self.context.term()
        self.connected = False

    def display(self):
        pass
        
            
    
    @profile
    def get_observation(self) -> Observation:
        """Create an observation with current robot state.

        Includes only modules that produced a new sample since the last call.

        Returns:
            Observation: Contains ModuleObs readings organized by module
        """
        obs_data = {}
        for module_name, obs_buffer in self.observation_buffers.items():
            if obs_buffer.count == 0:
                continue
            module_obs = obs_buffer.get().to_dict()
            obs_data[module_name] = module_obs
        return Observation(obs_data)

        # for module_name, obs_buffer in self.observation_buffers.items():
        #     if obs_buffer.count == 0:
        #         continue
        #     module_obs = obs_buffer.get()
        #     self.last_observation[module_name] = module_obs
        # return self.last_observation

    @profile
    def process_action(self, action: Action):
        """Process received action
        Args:
            action (Action): Action to process
        """

        if action.action_type == ActionType.MODULE_ACTION:
            # put module actions into action queues
            for module_name, module_action in action.items():
                self.action_queues[module_name].put(module_action)
                StatusBoard.set(f"{module_name} Action Queue Size", self.action_queues[module_name].qsize())
        else:
            self._dispatch_action(action, time.monotonic())

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
        

     # ========= action dispatch system ============
    def _dispatch_action(self, action: Action, t_now: float) -> bool:
        """
        Dispatch an action to its registered handler.
        
        Args:
            action: The Action object from the queue
            t_now: Current monotonic time
            
        Returns:
            bool: True to continue running, False to stop
        """
        handler_name = self.action_handlers.get(int(action.action_type))
        
        if handler_name:
            handler = getattr(self, handler_name)
            return handler(action, t_now)
        else:
            # Unknown action - log warning but continue
            if self.verbose:
                print(f"[Robot] Unknown action type {action.action_type}, ignoring")
            return True