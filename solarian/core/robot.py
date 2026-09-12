import zmq
import time
import threading
import os
import sys
from abc import ABC, abstractmethod
from typing import Any
from multiprocessing.managers import SharedMemoryManager

from solarian.core import curve
from solarian.core.configs import RobotConfig
from solarian.core.clock_sync import (
    sync_with_server,
    get_offset,
    synced_time_ns,
    latency_ms_from_stamp_ns,
    LatencyEma,
    format_latency_status,
)
from solarian.core.message import Message
from solarian.core.action import Action
from solarian.core.data import ImageData, set_image_jpeg_quality
from .module import Module
from .bus_server import BusServer, BusClient
from solarian.core import Observation
from solarian.utils import profile, manual_profile, get_class, busy_wait, StatusBoard, setup_process_logging
from solarian.utils.profiling import merge_stats as merge_profiling_stats
from solarian.utils.shared_memory.observation_buffer import ObservationBuffer
from solarian.utils.shared_memory.action_queue import ModuleActionQueue
from solarian.utils.shared_memory.skill_status_buffer import SkillStatusBuffer
from solarian.conf import settings
from solarian.core.action import ActionType, ModuleAction, action_handler
from solarian.core.event import EventHost, event_inbox_for
from solarian.core.execution import ExecutionMode

import logging
logger = logging.getLogger("robot")


class RobotSchemaError(Exception):
    """Raised when a robot module has no matching entry in the
    Observation/Action schema.

    Every ``Module`` declared on a ``Robot`` subclass needs a corresponding
    ``ModuleObs``/``ModuleAction`` entry (matched by attribute name) on the
    project's ``Observation``/``Action`` subclasses. When that entry is
    missing the buffers/queues can't be created, so we fail early here with a
    message that points the user at the exact file and change to make.
    """

    def __init__(self, module_name, module_class, schema_cls, schema_kind, base_name):
        try:
            import inspect
            schema_file = inspect.getsourcefile(schema_cls) or "<unknown>"
        except (TypeError, OSError):
            schema_file = "<unknown>"

        schema_name = getattr(schema_cls, "__name__", str(schema_cls))
        module_cls_name = getattr(module_class, "__name__", str(module_class))
        field_cls = f"{module_cls_name.replace('Module', '')}{schema_kind.capitalize()}"

        message = (
            f"Module '{module_name}' ({module_cls_name}) is declared on the robot "
            f"but has no {schema_kind} defined in the '{schema_name}' schema.\n"
            f"\n"
            f"Every module needs a matching '{base_name}' entry (by attribute name) "
            f"on your '{schema_name}' class.\n"
            f"\n"
            f"To fix, edit {schema_file} and add a '{base_name}' for '{module_name}', e.g.:\n"
            f"\n"
            f"    class {field_cls}({base_name}):\n"
            f"        ...  # actuator/sensor fields for '{module_name}'\n"
            f"\n"
            f"    class {schema_name}(...):\n"
            f"        ...\n"
            f"        {module_name} = {field_cls}()\n"
        )
        super().__init__(message)


class ModuleMap(dict):
    """Attribute-accessible mapping of module name → :class:`Module`."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _ModulesDescriptor:
    """``Robot.modules`` on the class (templates) and instance (live modules)."""

    def __get__(self, obj, cls=None):
        cls = cls if cls is not None else type(obj)
        if obj is not None:
            inst_map = obj.__dict__.get("_module_map")
            if inst_map is not None:
                return inst_map
        ns = ModuleMap()
        for name, val in cls.__dict__.items():
            if isinstance(val, Module):
                ns[name] = val
        return ns


class Robot(EventHost, ABC):

    _instance = None

    # Override in subclasses to use a custom config class (must extend ModuleConfig)
    config_class: type = RobotConfig

    #: Run this robot's modules in a MuJoCo simulation instead of on hardware.
    #: Defaults to this class attribute, but is overridable per-run by the
    #: ``sim`` config field (e.g. ``solarian run-robot sim=true``). When set, a
    #: ``scene`` must be declared and ``connect``/``start``/``disconnect``
    #: delegate to :mod:`solarian.sim.runtime` instead of opening hardware.
    is_sim: bool = False

    modules = _ModulesDescriptor()

    #: Simulation scene (:class:`~solarian.sim.configs.SceneConfig`). Required
    #: when ``is_sim`` is true; describes the environment and how each module's
    #: MJCF attaches. Ignored on hardware.
    scene = None

    #: Show a live overview window of the whole scene while running in sim.
    #: Overridable at runtime with the ``SOLARIAN_SIM_VIEW`` env var.
    view: bool = False

    #: Skill classes this robot can be asked to run, on top of the ones
    #: installed skill plugins publish. Declare a class here when the skill
    #: lives in the project rather than in a plugin.
    skills: tuple[type, ...] = ()

    #: Filled on first use by :meth:`_skill_classes`; entry-point discovery
    #: reads installed package metadata, so it is worth doing once.
    _skill_class_cache: dict[str, type] | None = None

    
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

        # Resolve the sim switch: the class attribute is the default, but the
        # ``sim`` config field (settable via yaml or ``run-robot sim=true``)
        # overrides it when present so the same robot class can run real or sim.
        cfg_sim = getattr(cfg, "sim", None)
        self.is_sim = type(self).is_sim if cfg_sim is None else bool(cfg_sim)
        if self.is_sim and self.scene is None:
            raise ValueError(
                f"Robot '{type(self).__name__}' has is_sim=True but no `scene` "
                f"is declared. Set a `scene` (SceneConfig) describing the "
                f"environment and how each module's MJCF attaches."
            )
        # Populated by solarian.sim.runtime when running in simulation.
        self._world_server = None
        self._viewer_proc = None

        # ZMQ is created in ``_init_sockets`` *after* module processes fork.
        # Creating a Context here would start ZMQ I/O threads and make child
        # modules prone to post-fork deadlocks (missing arm/gripper obs).
        self.context = None
        self.action_socket = None
        self.state_socket = None

        #self.control_lock = threading.Lock()
        self.observation_lock = threading.Lock()
        self.last_observation = Observation({})
        self.windows_initialized = False

        self.current_velocity = None
        self.current_arm_state = None
        
        self.running = False
        self.verbose = True
        self.captured_start_pos = False        
        self.connected = False
        self._action_latency = LatencyEma()
        self._obs_compress_latency = LatencyEma()
        #: Monotonic stamp of the last message from the controller, action or
        #: heartbeat. ``None`` until one arrives: a robot nothing has commanded
        #: yet has no link to have lost.
        self._last_command_ns: int | None = None
        self._link_lost = False
        set_image_jpeg_quality(getattr(cfg, "img_compression_quality", None))

        self._module_map = ModuleMap()
        self._bus_servers: list[BusServer] = []
        self.observation_buffers = dict[str, ObservationBuffer]()
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
                self._module_map[name] = module_class(module_cfg, name)
                setattr(self, name, self._module_map[name])

        # Phase 2: Build example observations/actions (now with correct shapes).
        example_observation = Observation.zeros()
        example_action = Action.zeros()

        # Phase 3: Create shared-memory buffers and queues and assign to modules.
        action_cls = Action._resolve_class()
        observation_cls = Observation._resolve_class()
        for name, (module_class, module_cfg) in module_configs.items():
            module_obs = example_observation[name]
            if module_obs is None:
                raise RobotSchemaError(
                    module_name=name,
                    module_class=module_class,
                    schema_cls=observation_cls,
                    schema_kind="observation",
                    base_name="ModuleObs",
                )
            try:
                module_act = example_action[name]
            except KeyError:
                raise RobotSchemaError(
                    module_name=name,
                    module_class=module_class,
                    schema_cls=action_cls,
                    schema_kind="action",
                    base_name="ModuleAction",
                ) from None

            self.observation_buffers[name] = ObservationBuffer.create_from_example(
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

        #: Robot-side skills currently running, keyed by skill name. Kept so
        #: each tick's events can reach them.
        self._local_skills: dict[str, Any] = {}
        #: Skills that have been asked to stop but have not exited yet, as
        #: ``name -> (skill, terminate_deadline)``. Reaped from the control
        #: loop rather than joined at the point of stopping.
        self._stopping_skills: dict[str, tuple[Any, float]] = {}
        self._init_events()


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
        from solarian.core.ik.actuator import CartesianIKActuator

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
                # Cartesian IK actuators are not hardware: they are driven by an
                # IK wrapper over the joint bus client-side and have no real
                # motor mode, so they must never reach the hardware BusServer.
                components = {
                    name: comp
                    for name, comp in module.module_bus.components_by_port[port].items()
                    if not isinstance(comp, CartesianIKActuator)
                }
                all_components.update(components)

                if bus_class is None and components:
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

            # Any message that arrives is evidence the link is up, including a
            # heartbeat carrying no action. This is the only place that can
            # tell "the controller has nothing to say" from "the controller is
            # gone", so it is where the watchdog's clock is reset.
            self._last_command_ns = time.monotonic_ns()

            if "action" not in msg.data:
                return None

            latency_ms = latency_ms_from_stamp_ns(msg.controller_timestamp)
            if latency_ms is not None:
                StatusBoard.set(
                    "Action latency (ms)",
                    format_latency_status(self._action_latency, latency_ms),
                )

            raw_action = msg.data["action"]
            action = Action(action_type=ActionType(raw_action["action_type"]), data=raw_action["data"])
            return action
        except zmq.Again:
            return None

    @profile
    def send_observation(self, observation: Observation):
        if observation is None:
            return
        msg = Message()
        msg.robot_timestamp = synced_time_ns()
        t0 = time.perf_counter()
        observation.compress()
        compress_ms = (time.perf_counter() - t0) * 1000.0
        StatusBoard.set(
            "Obs compress (ms)",
            self._obs_compress_latency.update(compress_ms).status_value(),
        )
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

        payload = msg.serialize()
        StatusBoard.set("Obs payload (KB)", round(len(payload) / 1024.0, 1))
        self.state_socket.send(payload)
    
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

    def _check_action_link(self) -> None:
        """Stop skills and hold safe when the controller has gone quiet.

        On a LAN a lost link is rare and brief. Once the command path crosses
        the internet it is neither, and the failure mode has to be chosen
        rather than inherited: with nothing here, the follower keeps driving
        toward the last waypoint it was given and nobody is left to change its
        mind. The robot is the only side still running when the link drops, so
        the decision has to live on this side of it.
        """
        timeout_s = self.cfg.connection.link_timeout_s
        if timeout_s is None or self._last_command_ns is None:
            return

        silence_s = (time.monotonic_ns() - self._last_command_ns) / 1e9
        if silence_s <= timeout_s:
            if self._link_lost:
                self._link_lost = False
                logger.info("Action link back after %.1f s", silence_s)
                StatusBoard.set("Action link", "up")
            return

        if self._link_lost:
            return
        self._link_lost = True
        logger.error(
            "No action or heartbeat for %.1f s (link_timeout_s=%s): stopping "
            "skills and holding modules safe",
            silence_s,
            timeout_s,
        )
        StatusBoard.set("Action link", f"LOST ({silence_s:.1f}s)")
        self.stop_all_skills(wait=False)
        # Modules mostly run in their own processes, where their bus lives, so
        # the safe state has to be asked for through the action queue like any
        # other command rather than called here. A camera has nothing to hold
        # and no handler for it.
        for name, queue in self.action_queues.items():
            if int(ActionType.HOLD_SAFE) in self.modules[name].action_handlers:
                queue.put(ModuleAction(action_type=ActionType.HOLD_SAFE))

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
        # A message from a session that is already over is not evidence of a
        # live link, so it must not arm the watchdog.
        self._last_command_ns = None

        poller = zmq.Poller()
        poller.register(self.action_socket, zmq.POLLIN)

        while self.running:
            self._check_module_health()
            self._check_action_link()
            self._reap_stopping_skills()
            self.update()

            # Same shape as the controller's tick: drain first so handlers run
            # on this thread, then advance the inline modules, whose loop this
            # is. A robot-side skill sees the batch through its inbox.
            events = self._drain_events()
            self._tick_local_modules()
            self._fan_out_events(events)

            if poller.poll(timeout=1):
                action = self.get_action()
                if action is not None:
                    StatusBoard.set("Action", action)
                    self.process_action(action)
            # actions that are generated by a skill running on the robot are pushed to the action queue directly by the skill itself

    def _tick_local_modules(self) -> None:
        """Advance every inline module once per control-loop iteration.

        An inline module is set up by :meth:`start` and then has no loop of
        its own; without this it would never publish an observation or emit
        an event after startup.
        """
        for module in self.modules.values():
            if module.execution_mode is ExecutionMode.INLINE:
                module.runner.tick()

    # ========= robot-side skills ============
    def start_skill(self, skill):
        """Wire a skill to this robot's shared memory and start it.

        Robot-side skills are launched from :meth:`start_named_skill` when the
        controller asks for one. Going through here rather than starting the
        skill directly is what connects it to the robot's buffers and gives
        it an event inbox, so it sees the same events the robot does.

        Wiring here rather than at construction is what keeps a skill the same
        thing on either side of the link: the controller's ``start_skill``
        takes a skill an operator's key handler built, which knows nothing
        about buffers, so it cannot pass them in — and a skill initialised one
        way on the robot and another way on the controller would be a worse
        trade than four assignments in the one place that owns the buffers.
        """
        skill.observations = self.observation_buffers
        skill.action_queues = self.action_queues
        skill.skills_status_buffer = self.skills_status_buffer
        skill.event_inbox = event_inbox_for(skill)
        self._local_skills[skill.skill_name] = skill
        skill.start()
        return skill

    def forget_skill(self, skill) -> None:
        """Drop a stopped skill from the registry so it stops receiving events."""
        self._local_skills.pop(skill.skill_name, None)

    @classmethod
    def _skill_classes(cls) -> dict[str, type]:
        """Skill classes this robot can be asked to run, by registry name.

        The ``solarian.skills`` entry points plus this class's own ``skills``,
        so a skill becomes startable on the robot by being installed or
        declared rather than by the robot growing a handler for it.
        """
        if cls.__dict__.get("_skill_class_cache") is None:
            from solarian.core.ai.system1.registry import discover_skill_classes

            found = {
                skill_cls.registry_name(): skill_cls
                for skill_cls in discover_skill_classes().values()
                if isinstance(skill_cls, type)
            }
            found.update(
                {skill_cls.registry_name(): skill_cls for skill_cls in cls.skills}
            )
            cls._skill_class_cache = found
        return cls._skill_class_cache

    @action_handler(ActionType.START_SKILL)
    def start_named_skill(self, action, t_now: float) -> bool:
        """Start the skill the controller named, with the params it sent.

        Generic on purpose: a project adds a robot-side skill by publishing it
        as a skill plugin, not by writing a per-skill action type and handler
        on both sides of the link.

        Every read here is a ``.get``: this payload was deserialized off the
        wire, and an unknown or missing skill name should log and carry on
        rather than take the control loop down with a ``KeyError``.
        """
        data = action.data
        name = str(data.get("skill") or "")
        skill_cls = self._skill_classes().get(name)
        if skill_cls is None:
            logger.warning(
                "Cannot start unknown skill '%s'; installed: %s",
                name,
                sorted(self._skill_classes()),
            )
            return True
        running = self._local_skills.get(name)
        if running is not None and running.is_alive():
            logger.info("Skill '%s' already running (pid=%s)", name, running.pid)
            return True
        if name in self._stopping_skills:
            # A second process would command the same modules as the one still
            # winding down, and both would own the same hardware.
            logger.info("Skill '%s' is still stopping; not restarting it", name)
            return True
        try:
            skill = self.start_skill(
                skill_cls(
                    params=data.get("params", {}),
                    status_mp_q=StatusBoard.get_mp_queue(),
                )
            )
        except Exception:
            logger.exception("Could not start skill '%s'", name)
            return True
        logger.info("Started skill '%s' (pid=%s)", name, skill.pid)
        return True

    @action_handler(ActionType.STOP_SKILL)
    def stop_named_skill(self, action, t_now: float) -> bool:
        self.stop_skill(str(action.data.get("skill") or ""))
        return True

    def stop_skill(self, name: str, timeout: float = 5.0) -> None:
        """Ask a robot-side skill to stop; the reaper finishes the job.

        Joining here would hold the control loop for as long as the skill's
        last tick takes — a policy skill mid-inference is hundreds of
        milliseconds — and no action is read meanwhile. On a CONFLATE link
        that is not a delay but a loss: every command the controller sends
        during the stall collapses into the last one.
        """
        skill = self._local_skills.get(name)
        if skill is None:
            return
        logger.info("Stopping skill '%s' (pid=%s)", name, skill.pid)
        skill.request_stop()
        self._stopping_skills[name] = (skill, time.monotonic() + float(timeout))
        self.forget_skill(skill)

    def _reap_stopping_skills(self, force: bool = False) -> None:
        """Retire skills that have finished stopping, and cancel their chunks.

        The waypoints a skill scheduled outlive it, so cancelling them is part
        of stopping it — otherwise the next skill to command those modules
        spends seconds fighting a trajectory whose author is gone. Only once
        the skill is dead: its final tick dispatches one more chunk after the
        stop is requested.
        """
        now = time.monotonic()
        for name, (skill, deadline) in list(self._stopping_skills.items()):
            if skill.is_alive():
                if not (force or now >= deadline):
                    continue
                logger.warning("Skill '%s' did not stop; terminating", name)
                skill.terminate()
                skill.join(timeout=1.0)
            self._stopping_skills.pop(name, None)
            for module_name in skill.commanded_module_names():
                module = self.modules.get(module_name)
                if module is None:
                    continue
                if int(ActionType.CANCEL_TRAJECTORY) in module.action_handlers:
                    self.action_queues[module_name].put(
                        ModuleAction(action_type=ActionType.CANCEL_TRAJECTORY)
                    )
            logger.info("Skill '%s' stopped", name)

    def stop_all_skills(self, timeout: float = 5.0, wait: bool = True) -> None:
        """Stop every robot-side skill, waiting for them out unless told not to.

        Shutdown waits, because the shared memory a skill is still reading is
        about to be released. The watchdog does not: a skill finishing an
        inference must not delay the safe hold that follows it, and whatever
        its last tick dispatched is undone by the reaper's cancel.
        """
        for name in list(self._local_skills):
            self.stop_skill(name, timeout=timeout)
        if not wait:
            return
        deadline = time.monotonic() + float(timeout)
        while self._stopping_skills and time.monotonic() < deadline:
            self._reap_stopping_skills()
            time.sleep(0.01)
        self._reap_stopping_skills(force=True)

    def connect(self, skip_calibrate: bool = False):
        # Align wall-clock stamps with the relay before accepting actions.
        self.sync_clock()

        if self.is_sim:
            from solarian.sim.runtime import sim_connect
            sim_connect(self)
            return

        for module_name, module in self.modules.items():
            module.load_calibration()

            if not module.needs_motor_calibration():
                logger.info(f"No calibration required for module {module_name}")
                continue

            module.connect()

            if module.calibration is None and not skip_calibrate:
                module.calibrate()

            calibration = module.require_calibration()
            if not module.is_calibrated():
                module.write_calibration()

            module.disconnect()
            module.connect(calibration)

            logger.info(f"Calibration loaded for module {module_name}")

        self.connected = True
        logger.info("Robot connected")

    def sync_clock(self) -> None:
        """Estimate wall-clock offset vs the relay server (soft-fail)."""
        try:
            # Use a temporary ZMQ context and tear it down — do not touch the
            # robot's long-lived context (created after module fork).
            result = sync_with_server(
                self.cfg.connection.ip,
                self.cfg.connection.clock_sync_port,
                connection=self.cfg.connection,
            )
            StatusBoard.set("Clock offset (ms)", result.offset_s * 1000.0)
            StatusBoard.set("Clock sync RTT (ms)", result.rtt_s * 1000.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Clock sync failed (%s); using local wall clock (offset=%.1f ms)",
                exc,
                get_offset() * 1000.0,
            )

    def _init_sockets(self) -> None:
        """Create ZMQ sockets after module processes have been forked."""
        if self.context is not None:
            return
        self.context = zmq.Context()

        self.action_socket = self.context.socket(zmq.PULL)
        self.action_socket.setsockopt(zmq.CONFLATE, 1)
        curve.apply_client(self.action_socket, self.cfg.connection, "robot actions")
        self.action_socket.connect(
            f"tcp://{self.cfg.connection.ip}:{self.cfg.connection.action_sender_port}"
        )

        self.state_socket = self.context.socket(zmq.PUB)
        self.state_socket.setsockopt(zmq.CONFLATE, 1)
        curve.apply_client(self.state_socket, self.cfg.connection, "robot state")
        self.state_socket.connect(
            f"tcp://{self.cfg.connection.ip}:{self.cfg.connection.state_receiver_port}"
        )
        if curve.is_enabled(self.cfg.connection):
            logger.info(
                "Robot link is CURVE-encrypted to relay key %s",
                self.cfg.connection.curve_server_key,
            )

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

        # Queues only — do not start the UI thread until after module forks.
        StatusBoard(start_ui=False)
        status_mp_q = StatusBoard.get_mp_queue()
        
        try:
            # Release parent-process hardware connections before forking so that
            # child processes can open their own clean handles (required for e.g.
            # macOS AVFoundation cameras which only allow one consumer). In sim
            # this clears the parent-side buses while keeping each module's
            # BusClient override so the forked children re-attach to the world.
            for module in self.modules.values():
                module._status_mp_q = status_mp_q
                module.disconnect()

            if self.is_sim:
                # The single MuJoCo world server (started in connect()) is the
                # sole "bus server"; no hardware ports to share.
                self._bus_servers = [self._world_server] if self._world_server is not None else []
            else:
                # For every port shared by 2+ modules, spin up a BusServer that
                # owns the real serial connection and hand each module a
                # BusClient proxy.  Modules keep their own processes; they just
                # talk to the server instead of opening the port themselves.
                self._start_shared_bus_servers()

            for module_name, module in self.modules.items():
                # The whole connection, not just host and port: with CURVE on,
                # the clock port refuses a plaintext peer like every other relay
                # socket, so a forked module needs the keys too.
                module._clock_sync_connection = self.cfg.connection
                module.start()

            # Safe to become multithreaded now that children have forked.
            self._init_sockets()
            StatusBoard.start_ui()

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
        if self.is_sim:
            # Tear down the live viewer; the world server is terminated below
            # because it is listed in self._bus_servers.
            from solarian.sim.runtime import sim_teardown
            sim_teardown(self)

        self.running = False
        if self.action_socket is not None:
            self.action_socket.close()
            self.action_socket = None
        if self.state_socket is not None:
            self.state_socket.close()
            self.state_socket = None
        for module in self.modules.values():
            if module.is_alive():
                module.terminate()
                module.join(timeout=5)
            child_stats = module.get_profiling_stats()
            if child_stats is not None:
                merge_profiling_stats(child_stats)
            module.disconnect()
        for srv in self._bus_servers:
            if srv.is_alive():
                srv.terminate()
                srv.join(timeout=5)
        if self.context is not None:
            self.context.term()
            self.context = None
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