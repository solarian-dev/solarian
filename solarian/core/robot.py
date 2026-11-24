import zmq
import time
import threading
import os
import sys
from abc import ABC, abstractmethod
from typing import Any
from multiprocessing.managers import SharedMemoryManager

from solarian.core import curve
from solarian.core.conf_paths import project_path
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
from solarian.utils.configs import ConfigError, config_error_in
from solarian.utils.profiling import merge_stats as merge_profiling_stats
from solarian.utils.shared_memory.observation_buffer import ObservationBuffer
from solarian.utils.shared_memory.action_queue import ModuleActionQueue
from solarian.utils.shared_memory.skill_status_buffer import SkillStatusBuffer
from solarian.conf import settings
from solarian.core.action import ActionType, ModuleAction, action_handler
from solarian.core.event import EventHost, event_inbox_for
from solarian.core.execution import ExecutionMode
from solarian.core.skill import SkillStatus, failure_reason, refuse_disabled_modules
from solarian.core.skill_registry import load_skill_class

import logging
logger = logging.getLogger("robot")

# A command older than this was meant for a session that is already over.
# The relay holds the in-flight send until a robot connects, and the drain
# at control-loop start races that handshake, so the check is on every recv.
_MAX_COMMAND_AGE_S = 1.0

# Messages the state socket queues before it starts dropping, when the relay
# falls behind. A few frames of slack; more would only deliver stale ones.
_STATE_SNDHWM = 16

# How often the control loop asks whether a module died or the link dropped.
# Both are judged in seconds, so checking at 20 Hz is already generous.
_SUPERVISION_PERIOD_S = 0.05

# How long the control loop waits for an action before it drains module events
# and ticks inline modules anyway. Each turn reads every module's event queue,
# so 1 ms turns cost about 14% of a Pi 5 core; an action still wakes it at once.
_IDLE_POLL_MS = 5

# A camera child blocked in one capture (1.5s on a Pi CSI camera) still has to
# notice the stop and finish its driver's close, which gives up after 2s.
_MODULE_STOP_JOIN_S = 5.0

# terminate() is a SIGTERM, and the handler's close is the same 2s bound.
_MODULE_TERM_JOIN_S = 3.0

_MODULE_KILL_JOIN_S = 1.0


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


def _is_image(value) -> bool:
    return getattr(value, "ndim", None) == 3


def _has_image(observation: Observation) -> bool:
    for _, module_obs in observation.items():
        for _, field in module_obs.items():
            if _is_image(field.value):
                return True
    return False


def _without_images(observation: Observation) -> Observation:
    """Drop every module carrying an image, whole.

    Its other fields (a bbox) describe that frame, and receivers replace a
    module wholesale: a camera module without its image would blank the
    controller's view and zero-fill the image in a server skill's history.
    """
    return Observation(
        {
            name: module_obs.to_dict()
            for name, module_obs in observation.items()
            if not any(_is_image(field.value) for _, field in module_obs.items())
        }
    )


class Robot(EventHost, ABC):

    _instance = None

    # Override in subclasses to use a custom config class (must extend ModuleConfig)
    config_class: type = RobotConfig

    modules = _ModulesDescriptor()

    #: Simulation scene (:class:`~solarian.sim.configs.SceneConfig`). Required
    #: to run with ``solarian run-robot --sim``; describes the environment and
    #: how each module's MJCF attaches. Ignored on hardware.
    scene = None

    
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

        # When true, connect/start/disconnect delegate to solarian.sim.runtime
        # instead of opening hardware. Calibration commands and the shell pass a
        # plain DictConfig that may lack the key.
        self.is_sim = bool(getattr(cfg, "sim", False))
        if self.is_sim and self.scene is None:
            raise ValueError(
                f"Robot '{type(self).__name__}' was asked to run in sim (--sim) "
                f"but declares no `scene`. Set a `scene` (SceneConfig) describing "
                f"the environment and how each module's MJCF attaches."
            )
        #: Draw the scene in a window. Per run, like ``is_sim``; the calibration
        #: commands set it on hardware for their live mirror.
        self.view = bool(getattr(cfg, "view", False))
        if self.view and not self.is_sim:
            raise ValueError(
                "--view draws the simulation, so it needs --sim: "
                "`solarian run-robot --sim --view`."
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
        #: Built by :meth:`_open_recorder` just before the modules fork.
        #: ``None`` when recording is disabled.
        self.recorder = None
        #: The episode currently being sealed, if any. Sealing is slow and
        #: runs off the control loop; see :meth:`_seal_off_the_control_loop`.
        self._seal_thread: threading.Thread | None = None
        #: Id of the last episode boundary acted on, echoed back with the
        #: observation. The controller's action socket is CONFLATE, so a
        #: RECORD_START it sent may never arrive; this is how it learns the
        #: difference and asks again.
        self._record_request = 0
        self._action_latency = LatencyEma()
        self._obs_compress_latency = LatencyEma()
        #: Monotonic stamp of the last message from the controller, action or
        #: heartbeat. ``None`` until one arrives: a robot nothing has commanded
        #: yet has no link to have lost.
        self._last_command_ns: int | None = None
        self._link_lost = False
        set_image_jpeg_quality(getattr(cfg, "img_compression_quality", None))

        #: Cores this process may use before any reservation narrows it, so
        #: the widening in :meth:`_reserve_encoder_core` has a full set to go
        #: back to. Empty on platforms with no affinity API (macOS).
        self._all_cores: list[int] = (
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        )

        self._module_map = ModuleMap()
        self._bus_servers: list[BusServer] = []
        self.observation_buffers = dict[str, ObservationBuffer]()
        #: Last buffer count included in an observation. ``get_observation``
        #: skips a module whose count has not moved, so a 30 Hz camera frame
        #: is not JPEG-encoded on every 5 ms wake of the sender.
        self._last_obs_count: dict[str, int] = {}
        #: Recording tuple last successfully sent. ``None`` until the first
        #: message, so the controller hears the initial state.
        self._last_recording_sent: tuple[bool, bool, int, int] | None = None
        #: Skill-status records included in the last successful send.
        self._skill_status_sent = 0
        #: Last time a camera image was put on the wire. Monotonic.
        self._last_preview_image_s = 0.0
        self.action_queues = {}
        #: Modules an action arrived for with no queue here, warned about once.
        self._unrouted_action_modules: set[str] = set()

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
        declared = {
            name: val
            for name in self.__class__.__dict__
            if isinstance(val := getattr(self, name, None), Module)
        }
        loaded = {name: self.load_module_config(name, type(val)) for name, val in declared.items()}
        #: Modules left out of this run: ``enabled: false`` in their yaml, or,
        #: in sim, mounted on one that is. Reported to the controller with
        #: every observation, so its own copy of the yamls does not decide.
        self.disabled_modules: set[str] = {n for n, cfg in loaded.items() if not cfg.enabled}
        if self.is_sim and self.disabled_modules:
            self._prune_scene(loaded)
        if self.disabled_modules:
            logger.info("Disabled modules: %s", ", ".join(sorted(self.disabled_modules)))
        module_configs: dict[str, tuple[type, Any]] = {}
        for name, val in declared.items():
            if name in self.disabled_modules:
                continue
            module_class = val.__class__
            module_cfg = loaded[name]
            module_configs[name] = (module_class, module_cfg)
            self._module_map[name] = val.bind(module_cfg, name)
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
                get_max_k=module_cfg.get_max_k,
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
        #: ``START_SKILL`` payloads for skills that arrived while a skill of the
        #: same name was still stopping, launched once the reaper retires it.
        self._deferred_starts: dict[str, dict] = {}
        #: Why a skill never started, by run id, for its FAILED status row.
        self._launch_failures: dict[int, str] = {}
        self._init_events()


    def load_module_config(self, module_name: str, module_class: type):
        """Load module config from YAML using the module's own config class"""
        from solarian.core.conf_paths import module_config_path
        from solarian.utils.configs import load_config_yaml
        yaml_path = str(module_config_path(module_name))
        module_cfg = load_config_yaml(yaml_path, module_class.config_class)
        return module_cfg

    def _prune_scene(self, loaded: dict[str, Any]) -> None:
        """Take the disabled modules out of the sim scene, and the modules that lose their mount.

        A gripper on a disabled arm's flange, or a camera rendering from a
        mount on it, has nothing left to drive or render, so it is disabled
        too, with a warning since its own yaml said otherwise.
        """
        from solarian.sim.scene import scene_without

        full = type(self).scene
        self.scene = scene_without(full, self.disabled_modules)
        kept_attachments = {a.module for a in self.scene.attachments}
        dropped_cameras = {c.name for c in full.cameras} - {c.name for c in self.scene.cameras}
        orphans = {
            a.module for a in full.attachments if a.module not in kept_attachments
        } - self.disabled_modules
        for name, cfg in loaded.items():
            if any(s.sim is not None and s.sim.camera in dropped_cameras for s in cfg.sensors.values()):
                orphans.add(name)
        orphans -= self.disabled_modules
        if orphans:
            logger.warning(
                "Disabling %s in sim: mounted on a disabled module", ", ".join(sorted(orphans))
            )
            self.disabled_modules |= orphans

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
                    bus_class=bus_class,
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
    def get_actions(self) -> list[Action]:
        """Every action waiting on the link, oldest first, without blocking.

        The link is lossless, so a chunk the relay sent as several messages
        arrives as several messages and has to be read as all of them. A
        message carries the controller's ``action`` and/or a batch of
        ``actions`` from skills the relay hosts.
        """
        actions: list[Action] = []
        while True:
            try:
                raw = self.action_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                return actions
            msg = Message.deserialize(raw)

            latency_ms = latency_ms_from_stamp_ns(msg.controller_timestamp)
            if latency_ms is not None:
                StatusBoard.set(
                    "Action latency (ms)",
                    format_latency_status(self._action_latency, latency_ms),
                )
            if latency_ms is not None and latency_ms > _MAX_COMMAND_AGE_S * 1000.0:
                logger.warning(
                    "Dropping command %.1f s old (limit %.1f s)",
                    latency_ms / 1000.0,
                    _MAX_COMMAND_AGE_S,
                )
                continue

            # Any fresh message from the controller is evidence its link is
            # up, including a heartbeat carrying no action. A batch the relay
            # composed on its own is not: a relay-hosted skill keeps sending
            # after the operator is gone. A command left over from a previous
            # session is not that evidence either.
            if "action" in msg.data or "heartbeat" in msg.data:
                self._last_command_ns = time.monotonic_ns()

            raw_actions = list(msg.data.get("actions", ()))
            if "action" in msg.data:
                raw_actions.append(msg.data["action"])
            for raw_action in raw_actions:
                actions.append(
                    Action(
                        action_type=ActionType(raw_action["action_type"]),
                        data=raw_action["data"],
                    )
                )

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

        # The controller displays recording state but does not own it, so it
        # gets told rather than asked. A RECORD_START it sent is not evidence
        # the robot received it; this is. Sent even with recording disabled,
        # because "the robot heard you and is not recording" is the answer a
        # controller retrying a boundary needs.
        enabled, active, episode, request, session = self._recording_snapshot()
        msg.data["recording"] = {
            "enabled": enabled,
            "active": active,
            "episode": episode,
            "request": request,
            # The session directory's name, so a controller logging beside the
            # recording files its notes under the same episode.
            "session": session,
        }
        sent_recording = (enabled, active, episode, request, session)
        # Every message rather than once: a controller that connects or
        # reconnects later learns at once which of its modules are not here.
        msg.data["disabled_modules"] = sorted(self.disabled_modules)

        # Forward any pending skill status transitions to the controller so
        # that the think process can react to them.  ``drain`` advances a
        # local cursor so records are only sent once. The count is taken
        # before the drain: a record that lands during the send stays
        # pending for the next one.
        skill_seen = self._skill_status_sent
        if self.skills_status_buffer is not None:
            skill_seen = self.skills_status_buffer.count
            records = self.skills_status_buffer.drain()
            if records:
                msg.data["skill_status"] = [
                    {
                        "skill_id": r.skill_id,
                        "status": r.status,
                        "timestamp": r.timestamp,
                        "reason": self._skill_failure_reason(r)
                        if r.status == int(SkillStatus.FAILED)
                        else "",
                    }
                    for r in records
                ]

        payload = msg.serialize()
        StatusBoard.set("Obs payload (KB)", round(len(payload) / 1024.0, 1))
        self.state_socket.send(payload)
        self._last_recording_sent = sent_recording
        self._skill_status_sent = skill_seen
    
    def _check_module_health(self):
        """Raise if any module process or bus server has crashed."""
        for module_name, module in self.modules.items():
            err = module.get_error()
            if err is not None:
                mod_name, tb_str = err
                config_error = config_error_in(tb_str)
                if config_error is not None:
                    raise ConfigError(message=f"Module '{mod_name}': {config_error}")
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
                if self.recorder is not None:
                    # Before the row, so a command and the sample that
                    # followed it are stored in that order. Drained even
                    # between episodes, or the tap fills and the producers
                    # start warning.
                    for module_name, queue in self.action_queues.items():
                        for module_action in queue.drain_tap():
                            self.recorder.note_module_action(
                                module_name, module_action
                            )
                observation = self.get_observation()
                # Before the send, which compresses the observation in place.
                # This is also the event the episode table is keyed on:
                # get_observation returns exactly the modules that advanced.
                # The row keeps every capture; the preview below may drop
                # the image so the Pi is not JPEG-encoding at camera rate.
                if self.recorder is not None:
                    self.recorder.write_row(observation)
                preview = self._preview_observation(observation)
                if self._should_send_observation(preview):
                    self.send_observation(preview)
                    if not preview.is_empty():
                        StatusBoard.set("Observation", preview)
            except Exception as e:
                logger.exception(f"Observation sender error: {e}")
            time.sleep(0.005)

    def _recording_snapshot(self) -> tuple[bool, bool, int, int, str | None]:
        recorder = self.recorder
        return (
            recorder is not None,
            recorder is not None and recorder.is_recording,
            recorder.episode_count if recorder else 0,
            self._record_request,
            recorder.session_dir_name if recorder else None,
        )

    def _skill_failure_reason(self, record) -> str:
        """Why the robot-side skill behind a FAILED ``record`` failed, from its crash report."""
        skills = [*self._local_skills.values(), *(s for s, _ in self._stopping_skills.values())]
        for skill in skills:
            if skill.skill_id != record.skill_id:
                continue
            error = skill.get_error()
            if error is not None:
                return failure_reason(error[1])
        return self._launch_failures.pop(record.skill_id, "")

    def _should_send_observation(self, observation: Observation) -> bool:
        """False when this wake would repeat a message the controller has.

        Joint state still goes out, because those modules advance at their
        own rate. An empty observation is sent only when recording state or
        skill status changed: the controller merges by module name, so a
        message with no camera keeps the previous frame.
        """
        if not observation.is_empty():
            return True
        if self._recording_snapshot() != self._last_recording_sent:
            return True
        skill_count = (
            self.skills_status_buffer.count
            if self.skills_status_buffer is not None
            else self._skill_status_sent
        )
        return skill_count != self._skill_status_sent

    def _preview_observation(self, observation: Observation) -> Observation:
        """Drop camera images when the preview already has a recent one.

        Joint fields are unchanged. An image-only wake then sends nothing,
        and the controller keeps the frame it already decoded.
        """
        if observation.is_empty() or not _has_image(observation):
            return observation
        now = time.monotonic()
        if now - self._last_preview_image_s >= 1.0 / self.cfg.preview_image_hz:
            self._last_preview_image_s = now
            return observation
        return _without_images(observation)

    def control_loop(self):
        logger.info("Control loop started")

        # Drop any action that was already queued on the link when we
        # connected.  Without this, a stale action from a previous controller
        # session would be dispatched as the very first action of the new
        # session, silently launching skills the operator did not request.
        drained = len(self.get_actions())
        if drained:
            logger.warning(
                "Discarded %d stale action(s) at control-loop start", drained
            )
        # A message from a session that is already over is not evidence of a
        # live link, so it must not arm the watchdog.
        self._last_command_ns = None

        poller = zmq.Poller()
        poller.register(self.action_socket, zmq.POLLIN)

        next_supervision = 0.0
        while self.running:
            # An action wakes the poll below the moment it lands; otherwise
            # the loop turns over every _IDLE_POLL_MS. Health and link checks
            # answer questions measured in seconds, and each one polls a pipe
            # per module, so running them every turn is thousands of syscalls
            # a second buying nothing.
            now = time.monotonic()
            if now >= next_supervision:
                next_supervision = now + _SUPERVISION_PERIOD_S
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

            if poller.poll(timeout=_IDLE_POLL_MS):
                for action in self.get_actions():
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
        refuse_disabled_modules(skill, self.disabled_modules)
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

    @action_handler(ActionType.START_SKILL)
    def start_named_skill(self, action, t_now: float) -> bool:
        """Start the skill class at the import path the controller sent.

        Generic on purpose: any skill the robot's environment can import can
        run here, with no per-skill action type or handler on either side of
        the link, and nothing to declare on the robot.

        Every read here is a ``.get``: this payload was deserialized off the
        wire, and an unknown or missing skill should log and carry on rather
        than take the control loop down with a ``KeyError``.
        """
        data = action.data
        name = str(data.get("skill") or "")
        path = str(data.get("path") or "")
        try:
            load_skill_class(path)
        except Exception as exc:
            logger.warning("Cannot start skill '%s' from %r: %s", name, path, exc)
            return True
        run_id = int(data.get("run_id") or 0)
        running = self._local_skills.get(name)
        if running is not None and running.is_alive():
            if running.skill_id == run_id:
                logger.info("Skill '%s' already running (pid=%s)", name, running.pid)
                return True
            # A newer run whose STOP for the old one was lost on the conflated
            # socket: it replaces the old run rather than being dropped.
            self.stop_skill(name)
        if name in self._stopping_skills:
            # A second process would command the same modules as the one still
            # winding down, and both would own the same hardware.
            logger.info("Skill '%s' is still stopping; starting it once it has", name)
            self._deferred_starts[name] = data
            return True
        self._launch_named_skill(name, data)
        return True

    def _launch_named_skill(self, name: str, data: dict) -> None:
        try:
            skill = self.start_skill(
                load_skill_class(str(data.get("path") or ""))(
                    params=data.get("params", {}),
                    skill_id=int(data.get("run_id") or 0),
                    status_mp_q=StatusBoard.get_mp_queue(),
                )
            )
        except ConfigError as exc:
            logger.error("Could not start skill '%s': %s", name, exc)
            # Without an ending row the controller would believe it runs.
            run_id = int(data.get("run_id") or 0)
            self._launch_failures[run_id] = str(exc)
            self.skills_status_buffer.put(
                skill_id=run_id,
                status=int(SkillStatus.FAILED),
                timestamp=time.time(),
            )
            return
        except Exception:
            logger.exception("Could not start skill '%s'", name)
            return
        logger.info("Started skill '%s' (pid=%s)", name, skill.pid)
        self._reserve_encoder_core()

    @action_handler(ActionType.STOP_SKILL)
    def stop_named_skill(self, action, t_now: float) -> bool:
        """Stop the run the controller named; a retried stop for a run already gone is ignored.

        Without the run id check, a stop the controller retries for a run
        that has since been replaced would stop its replacement.
        """
        data = action.data
        name = str(data.get("skill") or "")
        run_id = data.get("run_id")
        if run_id is None:
            self.stop_skill(name)
            return True
        deferred = self._deferred_starts.get(name)
        if deferred is not None and int(deferred.get("run_id") or 0) == int(run_id):
            self._deferred_starts.pop(name)
        running = self._local_skills.get(name)
        if running is not None and running.skill_id == int(run_id):
            self.stop_skill(name)
        return True

    @action_handler(ActionType.START_SERVICE)
    def start_service(self, action, t_now: float) -> bool:
        return self._route_service_action(action)

    @action_handler(ActionType.STOP_SERVICE)
    def stop_service(self, action, t_now: float) -> bool:
        return self._route_service_action(action)

    def _route_service_action(self, action) -> bool:
        """Hand a service start/stop to the module whose processor computes it.

        Robot-side processors live in their module's process, so the request
        travels on that module's action queue like any other command.
        """
        module_name = str(action.data.get("module") or "")
        queue = self.action_queues.get(module_name)
        if queue is None:
            logger.warning("Service action for unknown module '%s'", module_name)
            return True
        queue.put(
            ModuleAction(
                action_type=action.action_type,
                params={
                    "field": action.data.get("field"),
                    "params": action.data.get("params") or {},
                },
            )
        )
        return True

    def stop_skill(self, name: str, timeout: float = 5.0) -> None:
        """Ask a robot-side skill to stop; the reaper finishes the job.

        Joining here would hold the control loop for as long as the skill's
        last tick takes — a policy skill mid-inference is hundreds of
        milliseconds — and no action is read meanwhile. The relay queues
        only a bounded backlog toward the robot, so a long stall turns into
        dropped commands rather than late ones.
        """
        self._deferred_starts.pop(name, None)
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
            deferred = self._deferred_starts.pop(name, None)
            if deferred is not None:
                self._launch_named_skill(name, deferred)

    def stop_all_skills(self, timeout: float = 5.0, wait: bool = True) -> None:
        """Stop every robot-side skill, waiting for them out unless told not to.

        Shutdown waits, because the shared memory a skill is still reading is
        about to be released. The watchdog does not: a skill finishing an
        inference must not delay the safe hold that follows it, and whatever
        its last tick dispatched is undone by the reaper's cancel.
        """
        self._deferred_starts.clear()
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

            if module.calibration is None:
                if skip_calibrate:
                    module.disconnect()
                    continue
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
        curve.apply_client(self.action_socket, self.cfg.connection, "robot actions")
        self.action_socket.connect(
            f"tcp://{self.cfg.connection.ip}:{self.cfg.connection.action_sender_port}"
        )

        self.state_socket = self.context.socket(zmq.PUB)
        # Not CONFLATE: each message carries only the modules that advanced,
        # so conflating would let a joint-only message replace one carrying
        # a camera frame. PUB drops at the high-water mark instead of
        # blocking, which bounds the backlog when the relay stalls.
        self.state_socket.setsockopt(zmq.SNDHWM, _STATE_SNDHWM)
        curve.apply_client(self.state_socket, self.cfg.connection, "robot state")
        self.state_socket.connect(
            f"tcp://{self.cfg.connection.ip}:{self.cfg.connection.state_receiver_port}"
        )
        if curve.is_enabled(self.cfg.connection):
            logger.info(
                "Robot link is CURVE-encrypted to relay key %s",
                self.cfg.connection.curve_server_key,
            )

    # ========= recording ============
    def _open_recorder(self) -> None:
        """Start one encoder per camera and the session recorder.

        Runs before the modules fork, because a camera bus reaches its encoder
        the same way it reaches its observation buffer: by having been handed
        the shared memory while it was still in this process.
        """
        cfg = self.cfg.recording
        if not cfg.enabled:
            return

        from solarian.core.recording import (
            CameraMeta,
            SessionRecorder,
            VideoWriter,
            build_schema,
            build_structure,
        )

        schema = build_schema()
        fps = float(cfg.fps if cfg.fps is not None else self.cfg.fps)
        writers: dict[str, VideoWriter] = {}
        cameras: dict[str, CameraMeta] = {}

        for camera in schema.cameras:
            module = self.modules.get(camera.module)
            if module is None:
                continue
            sensor_cfg = module.sensors[camera.sensor].cfg
            height, width, _ = self._archive_shape(sensor_cfg)
            writer = VideoWriter(
                name=camera.name,
                fps=fps,
                shape=(height, width, 3),
                codec=cfg.codec,
                crf=cfg.crf,
                keyint=cfg.keyint,
                preset=cfg.preset,
                threads=cfg.threads,
                x264_options=cfg.x264_options,
                input_pix_fmt="bgr24" if sensor_cfg.color_mode == "bgr" else "rgb24",
                buffer_size=cfg.queue_size,
            )
            module.archive_sinks[camera.sensor] = writer.open(self.shm_manager)
            writers[camera.name] = writer
            cameras[camera.name] = CameraMeta(
                width=width,
                height=height,
                color_mode=sensor_cfg.color_mode,
                source_fps=float(sensor_cfg.fps),
                sensor_mode=f"{sensor_cfg.width}x{sensor_cfg.height}",
                # Stored rather than applied: the archive is pre-remap, so
                # export needs to know which calibration to warp it with.
                optical_remap=sensor_cfg.optical_remap,
                observation_resize=(
                    (sensor_cfg.resize_width, sensor_cfg.resize_height)
                    if sensor_cfg.resize_width is not None
                    else None
                ),
                remap_size=(
                    (sensor_cfg.remap_width, sensor_cfg.remap_height)
                    if sensor_cfg.remap_width is not None
                    else None
                ),
            )

        self.recorder = SessionRecorder(
            root=project_path(cfg.root),
            # Until a controller names one. `solarian record name=...` does;
            # an ordinary teleop session has no name to give.
            session=self.__class__.__name__,
            fps=fps,
            writers=writers,
            robot=build_structure(self),
            cameras=cameras,
            video_options={
                "codec": cfg.codec,
                "crf": cfg.crf,
                "keyint": cfg.keyint,
                "preset": cfg.preset,
                "pix_fmt": "yuv420p",
            },
            max_session_bytes=(
                int(cfg.max_session_gb * 1e9) if cfg.max_session_gb else None
            ),
            schema=schema,
        )

        # Every command a module receives passes through its action queue,
        # whichever process put it there, so that is where they are read back
        # for the table. Opened here, before anything forks, because a copy of
        # the queue is how a robot-side skill reaches the module at all.
        for queue in self.action_queues.values():
            queue.open_tap(self.shm_manager)

    def _archive_shape(self, sensor_cfg) -> tuple[int, int, int]:
        """``(height, width, 3)`` of the frames this camera will archive."""
        if self.is_sim:
            # MuJoCo renders straight to the observation size, so there is no
            # higher-fidelity source to keep. See sim.runtime._camera_spec.
            if sensor_cfg.resize_width is not None:
                height, width = sensor_cfg.resize_height, sensor_cfg.resize_width
            else:
                height, width = sensor_cfg.height, sensor_cfg.width
        elif sensor_cfg.rotation in (90, -90):
            height, width = sensor_cfg.width, sensor_cfg.height
        else:
            height, width = sensor_cfg.height, sensor_cfg.width

        if sensor_cfg.archive_width is not None and sensor_cfg.archive_height is not None:
            height, width = sensor_cfg.archive_height, sensor_cfg.archive_width
        return int(height), int(width), 3

    def _reserve_encoder_core(self) -> None:
        """Keep the control path off the core the encoder pins itself to.

        ``VideoWriter.run`` confines itself to the last core, which buys
        nothing while the scheduler is still free to put the camera module or
        a policy skill on that same core. Only in force while an episode is
        open: an idle encoder sits blocked on its queue, and a quarter of the
        board is too much to pay for a process that is not running.

        Re-applied whenever a skill starts, because a skill launched after
        the episode opened was forked with the reservation already in place
        on the parent but is not in ``_local_skills`` at that point.
        """
        # Below three cores, taking one away costs the control path more than
        # the encoder's migrations do. Matches VideoWriter's own threshold.
        if len(self._all_cores) < 3:
            return
        _, recording, *_ = self._recording_snapshot()
        cores = set(self._all_cores[:-1] if recording else self._all_cores)
        pids = [
            os.getpid(),
            *(module.pid for module in self.modules.values()),
            *(skill.pid for skill in self._local_skills.values()),
            *(srv.pid for srv in self._bus_servers),
        ]
        # Affinity is a thread property, and passing a process id moves only
        # that process's main thread. That is the one that matters in each
        # case here — the control loop, a module's worker body, a bus
        # server's read loop — and threads started afterwards inherit it.
        for pid in pids:
            if pid is None:
                continue
            try:
                os.sched_setaffinity(pid, cores)
            except OSError:
                # Exited between building the list and getting here.
                continue

    @action_handler(ActionType.RECORD_START)
    def start_recording(self, action, t_now: float) -> bool:
        self._record_request = int(action.data.get("request", 0))
        if self.recorder is None:
            logger.warning("Recording is disabled on this robot; ignoring RECORD_START")
            return True
        # An encoder holds one episode at a time, so a take cannot open until
        # the previous one is closed. Usually already true by the time the
        # operator presses r again.
        self._await_seal()
        session = action.data.get("session")
        if session:
            self.recorder.open_session(session)
        self.recorder.start_episode(
            task=action.data.get("task"), skill=action.data.get("skill")
        )
        self._reserve_encoder_core()
        return True

    @action_handler(ActionType.RECORD_STOP)
    def stop_recording(self, action, t_now: float) -> bool:
        self._record_request = int(action.data.get("request", 0))
        if self.recorder is not None:
            self._seal_off_the_control_loop(self.recorder.end_episode)
        return True

    @action_handler(ActionType.RECORD_DROP)
    def drop_recording(self, action, t_now: float) -> bool:
        self._record_request = int(action.data.get("request", 0))
        if self.recorder is not None:
            self._seal_off_the_control_loop(self.recorder.drop_episode)
        return True

    def _seal_off_the_control_loop(self, seal) -> None:
        """Finish an episode on a thread of its own.

        Sealing blocks until every encoder has drained its backlog and closed
        its mux. The backlog is deliberately deep — that is what lets the
        encoder fall behind during a burst of motion instead of dropping
        captures — so on the control loop it would be seconds of frozen arm,
        and the deeper the queue the longer the freeze. Recording itself stops
        immediately either way: the recorder clears its flag before any of the
        blocking work.
        """
        self._await_seal()

        def seal_then_widen():
            # The encoder keeps its reserved core until its backlog is on
            # disk. Lifting the reservation when recording stops rather than
            # when the encoder goes idle would hand the control path a core
            # the encoder is still working flat out on.
            seal()
            self._reserve_encoder_core()

        self._seal_thread = threading.Thread(
            target=seal_then_widen, name="EpisodeSeal", daemon=True
        )
        self._seal_thread.start()

    def _await_seal(self) -> None:
        """Block until the episode being sealed, if any, is on disk."""
        if self._seal_thread is not None:
            self._seal_thread.join()
            self._seal_thread = None

    def initialize(self):
        # if not self.is_connected():
        #     raise Exception("Robot not connected, cannot initialize, must call connect() first.")
        # initialize modules
        logger.info("Initializing modules")
        for module_name, module in self.modules.items():
            # A process module reconnects in its own process, and what
            # initialize() set up here would not survive that: the bus state
            # lives in this process, and a disconnect may disable its motors.
            # worker_setup() initializes it there instead.
            if module.execution_mode.crosses_process_boundary:
                continue
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

            self._open_recorder()

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
        # run-robot's SIGTERM handler raises out of the main thread. Once
        # this has started, a second signal has to be ignored or it aborts
        # the camera close below.
        self._shutdown_started = True
        if self.is_sim:
            # Tear down the live viewer; the world server is terminated below
            # because it is listed in self._bus_servers.
            from solarian.sim.runtime import sim_teardown
            sim_teardown(self)

        self.running = False
        # Before the modules go: an episode still open has to be sealed while
        # its encoders are alive, or its mp4 is left without a footer. That
        # includes one a background seal has not finished with yet.
        self._await_seal()
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None
        if self.action_socket is not None:
            self.action_socket.close()
            self.action_socket = None
        if self.state_socket is not None:
            self.state_socket.close()
            self.state_socket = None
        for module in self.modules.values():
            # Stop first so the child closes its camera in teardown. SIGTERM
            # still runs that teardown; SIGKILL is only if the process is
            # stuck and would otherwise keep the pipeline locked.
            if module.is_alive():
                module.stop(wait=False)
                module.join(timeout=_MODULE_STOP_JOIN_S)
            if module.is_alive():
                module.terminate()
                module.join(timeout=_MODULE_TERM_JOIN_S)
            if module.is_alive():
                module.kill()
                module.join(timeout=_MODULE_KILL_JOIN_S)
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
            # Count first. A sample that lands after this read is sent on
            # the next call; storing the count from after ``get`` would
            # skip it.
            count = obs_buffer.count
            if count == 0 or count == self._last_obs_count.get(module_name):
                continue
            obs_data[module_name] = obs_buffer.get().to_dict()
            self._last_obs_count[module_name] = count
        return Observation(obs_data)

    @profile
    def process_action(self, action: Action):
        """Process received action
        Args:
            action (Action): Action to process
        """

        if action.action_type == ActionType.MODULE_ACTION:
            # Not recorded here. The queue below is where these meet the
            # commands a robot-side skill writes directly, and recording at
            # the one place both arrive is what keeps the table complete.
            for module_name, module_action in action.items():
                queue = self.action_queues.get(module_name)
                if queue is None:
                    # A controller that has not heard which modules are
                    # disabled yet, or runs an older version, still sends.
                    if module_name not in self._unrouted_action_modules:
                        self._unrouted_action_modules.add(module_name)
                        logger.warning("Dropping actions for module '%s': not running on this robot", module_name)
                    continue
                queue.put(module_action)
                StatusBoard.set(f"{module_name} Action Queue Size", queue.qsize())
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