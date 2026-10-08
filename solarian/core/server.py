import logging
import zmq
import threading
import time

from solarian.core import curve
from solarian.core.action import Action, ActionType
from solarian.core.message import Message
from solarian.core.configs import ServerConfig
from solarian.core.clock_sync import serve_clock_sync_reply
from solarian.core.data import set_image_jpeg_quality
from solarian.core.execution import ExecutionPlace
from solarian.core import Observation
from solarian.utils import StatusBoard

logger = logging.getLogger(__name__)

# Messages a robot-facing socket queues before it drops. The action link is
# lossless so a chunk sent as several messages arrives whole, but a robot
# that is gone must not leave a backlog it replays when it comes back; the
# robot's command-age check discards whatever slips through.
_ROBOT_HWM = 64

# Observation messages queued per controller. Small: a controller that falls
# behind should skip ahead, not watch a growing delay.
_CONTROLLER_HWM = 8

# TO DO: rename state to observation, rename cmd to action

class RelayServer:
    def __init__(self, cfg: ServerConfig):
        self.context = zmq.Context()
        self.running = False
        self.cfg = cfg
        self._logged_absent_robot = False
        set_image_jpeg_quality(cfg.img_compression_quality)

        # ZAP first: the authenticator has to be running before any CURVE server
        # socket completes a handshake, or early connections are refused.
        self._authenticator = curve.start_authenticator(
            self.context, cfg.curve_allowed_client_keys
        )

        # Setup action relay (Client → Relay → Robot)
        self.client_cmd_receiver = self.context.socket(zmq.PULL)
        self.client_cmd_receiver.setsockopt(zmq.LINGER, 0)
        curve.apply_server(self.client_cmd_receiver, cfg.connection, "action receiver")
        self.client_cmd_receiver.bind(f"tcp://*:{self.cfg.connection.action_receiver_port}")  # Clients send here
        
        self.robot_cmd_sender = self.context.socket(zmq.PUSH)
        self.robot_cmd_sender.setsockopt(zmq.SNDHWM, _ROBOT_HWM)
        self.robot_cmd_sender.setsockopt(zmq.LINGER, 0)
        # IMMEDIATE=1 makes PUSH only enqueue to completed connections. The
        # relay send is non-blocking, so a controller talking while no robot
        # is attached drops the command instead of blocking the relay thread
        # on it. A blocking send holds that one message and delivers it to
        # the next robot that connects.
        self.robot_cmd_sender.setsockopt(zmq.IMMEDIATE, 1)
        curve.apply_server(self.robot_cmd_sender, cfg.connection, "action sender")
        self.robot_cmd_sender.bind(f"tcp://*:{self.cfg.connection.action_sender_port}")  # Robot connects here

        # Setup image relay (Robot → Relay → Client). Not CONFLATE: each robot
        # message carries only the modules that advanced, so keeping just the
        # newest would drop camera frames behind joint-only messages.
        self.robot_state_receiver = self.context.socket(zmq.SUB)
        self.robot_state_receiver.setsockopt(zmq.RCVHWM, _ROBOT_HWM)
        self.robot_state_receiver.setsockopt(zmq.LINGER, 0)
        curve.apply_server(self.robot_state_receiver, cfg.connection, "state receiver")
        self.robot_state_receiver.bind(f"tcp://*:{self.cfg.connection.state_receiver_port}")  # Robot publishes here
        self.robot_state_receiver.setsockopt(zmq.SUBSCRIBE, b"")
        
        # Not CONFLATE either, for the same reason; a slow controller loses
        # the oldest messages at the high-water mark instead.
        self.client_state_sender = self.context.socket(zmq.PUB)
        self.client_state_sender.setsockopt(zmq.SNDHWM, _CONTROLLER_HWM)
        self.client_state_sender.setsockopt(zmq.LINGER, 0)
        curve.apply_server(self.client_state_sender, cfg.connection, "state sender")
        self.client_state_sender.bind(f"tcp://*:{self.cfg.connection.state_sender_port}")  # Clients subscribe here

        # Application clock authority for controller/robot SCHEDULE_* stamps.
        self.clock_sync_socket = self.context.socket(zmq.REP)
        self.clock_sync_socket.setsockopt(zmq.LINGER, 0)
        curve.apply_server(self.clock_sync_socket, cfg.connection, "clock sync")
        if curve.is_enabled(cfg.connection):
            logger.info(
                "Relay sockets are CURVE servers; public key %s",
                cfg.connection.curve_public_key,
            )
        self.clock_sync_socket.bind(f"tcp://*:{self.cfg.connection.clock_sync_port}")

        #: ``{(module_name, field_name): Processor}`` for the fields computed here.
        self._processors = Observation.create_processors(ExecutionPlace.SERVER)
        self.skill_host = None
        if cfg.host_skills:
            from solarian.core.server_skill_host import ServerSkillHost

            try:
                self.skill_host = ServerSkillHost()
            except Exception:
                logger.exception("Cannot host skills on this relay; server skills are disabled")

    # -- actions (controller -> robot) ---------------------------------------

    def _handle_locally(self, raw_action: dict) -> bool:
        """Apply a controller action addressed to this relay; True if it was.

        Service starts/stops for fields computed here and skill lifecycle for
        skills placed here stop at the relay. Everything else goes on to the
        robot, including service actions for fields the robot computes.
        """
        action_type = int(raw_action.get("action_type", -1))
        data = raw_action.get("data") or {}
        if action_type in (ActionType.START_SERVICE, ActionType.STOP_SERVICE):
            key = (str(data.get("module") or ""), str(data.get("field") or ""))
            proc = self._processors.get(key)
            if proc is None:
                return False
            if action_type == ActionType.START_SERVICE:
                proc.start(dict(data.get("params") or {}))
                logger.info("Started service %s/%s", *key)
            else:
                proc.stop()
                logger.info("Stopped service %s/%s", *key)
            return True
        if self.skill_host is None:
            return False
        if action_type == ActionType.START_SKILL:
            if data.get("execution_place") != ExecutionPlace.SERVER.value:
                return False
            self.skill_host.start_named_skill(data)
            return True
        if action_type == ActionType.STOP_SKILL:
            name = str(data.get("skill") or "")
            if not self.skill_host.runs(name):
                return False
            self.skill_host.stop_skill(name)
            return True
        return False

    def _send_to_robot(self, msg: Message) -> None:
        try:
            self.robot_cmd_sender.send(msg.serialize(), zmq.NOBLOCK)
        except zmq.Again:
            if not self._logged_absent_robot:
                logger.warning(
                    "No robot connected; dropping actions until one connects"
                )
                self._logged_absent_robot = True
        else:
            self._logged_absent_robot = False

    def _relay_controller_message(self, raw: bytes) -> None:
        msg = Message.deserialize(raw)
        raw_action = msg.data.get("action")
        if raw_action is not None and self._handle_locally(raw_action):
            # Still evidence the controller is alive, which the robot's
            # watchdog needs to hear.
            msg.data = {"heartbeat": True}
        self._send_to_robot(msg)

    def _send_skill_actions(self) -> None:
        """Send what server-hosted skills queued since the last tick, in order."""
        self.skill_host.reap()
        batch = self.skill_host.drain_actions()
        if not batch:
            return
        msg = Message()
        # The relay is the clock authority, so its own clock is synced time.
        msg.controller_timestamp = time.time_ns()
        msg.server_timestamp = msg.controller_timestamp
        msg.data = {"actions": [action.to_dict() for action in batch]}
        self._send_to_robot(msg)

    def _action_relay(self):
        """Forward controller actions to the robot, with server skills' actions.

        One thread owns the robot-facing socket: ZMQ sockets are not
        thread-safe, so a skill's chunk is composed and sent from here too.
        """
        poller = zmq.Poller()
        poller.register(self.client_cmd_receiver, zmq.POLLIN)
        while self.running:
            try:
                if poller.poll(timeout=1):
                    while True:
                        try:
                            raw = self.client_cmd_receiver.recv(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        try:
                            self._relay_controller_message(raw)
                        except Exception:
                            logger.exception("Error processing controller message")
                if self.skill_host is not None:
                    self._send_skill_actions()
            except Exception:
                logger.exception("Error in action relay")
                time.sleep(0.001)

    # -- observations (robot -> controller) ----------------------------------

    def _relay_robot_message(self, raw: bytes) -> None:
        if not self._processors and self.skill_host is None:
            self.client_state_sender.send(raw)
            return

        msg = Message.deserialize(raw)
        obs_data = msg.data.get("observation")
        if obs_data:
            start_time = time.perf_counter()
            observation = Observation(obs_data)
            for (module_name, field_name), proc in self._processors.items():
                module_obs = observation[module_name]
                if module_obs is None:
                    continue
                result = proc(module_obs)
                module_obs[field_name] = result
                # Only the computed fields are re-encoded; the robot's JPEGs
                # are forwarded as they arrived.
                encoded = result.model_copy()
                encoded.compress()
                obs_data[module_name][field_name] = encoded.to_dict()
            if self._processors:
                StatusBoard.set(
                    "Server observation enrichment time (ms)",
                    (time.perf_counter() - start_time) * 1000.0,
                )
        if self.skill_host is not None:
            records = self.skill_host.drain_status_records()
            if records:
                msg.data.setdefault("skill_status", []).extend(records)
        self.client_state_sender.send(msg.serialize())

        if obs_data and self.skill_host is not None:
            self.skill_host.put_observation(observation)

    def _state_relay(self):
        """Forward observations from robot to clients, computing server-side fields."""
        poller = zmq.Poller()
        poller.register(self.robot_state_receiver, zmq.POLLIN)
        while self.running:
            try:
                if not poller.poll(timeout=1):
                    continue
                while True:
                    try:
                        raw = self.robot_state_receiver.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    self._relay_robot_message(raw)
            except Exception:
                logger.exception("Error in state relay")
                time.sleep(0.001)

    def _clock_sync(self):
        """Answer Cristian/NTP-style clock sync requests from controller/robot."""
        poller = zmq.Poller()
        poller.register(self.clock_sync_socket, zmq.POLLIN)
        while self.running:
            try:
                if not poller.poll(timeout=100):
                    continue
                serve_clock_sync_reply(self.clock_sync_socket)
            except Exception as e:
                logger.warning("Clock sync reply failed: %s", e)
                time.sleep(0.01)

    def start(self):
        self.running = True
        # StatusBoard.set() defers the UI thread; start it for live metrics.
        StatusBoard.start_ui()
        # TO DO: action relay and state relay should be implemented as processes instead of threads
        threading.Thread(target=self._action_relay, daemon=True).start()
        threading.Thread(target=self._state_relay, daemon=True).start()
        threading.Thread(target=self._clock_sync, daemon=True).start()
        print("Relay server started")
        logger.info(
            "Clock sync listening on tcp://*:%s",
            self.cfg.connection.clock_sync_port,
        )

        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.running = False
        for proc in self._processors.values():
            proc.stop()
        if self.skill_host is not None:
            self.skill_host.shutdown()
        self.context.destroy()
