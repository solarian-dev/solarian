import logging
import zmq
import threading
import time

from solarian.core import curve
from solarian.core.message import Message
from solarian.core.configs import ServerConfig
from solarian.core.clock_sync import serve_clock_sync_reply
from solarian.core.data import DataComputer
from solarian.core import Observation
from solarian.utils import StatusBoard

logger = logging.getLogger(__name__)

# TO DO: rename state to observation, rename cmd to action

class RelayServer:
    def __init__(self, cfg: ServerConfig):
        self.context = zmq.Context()
        self.running = False
        self.cfg = cfg
        self.captured_click = None

        # ZAP first: the authenticator has to be running before any CURVE server
        # socket completes a handshake, or early connections are refused.
        self._authenticator = curve.start_authenticator(
            self.context, cfg.curve_allowed_client_keys
        )

        # Setup action relay (Client → Relay → Robot)
        self.client_cmd_receiver = self.context.socket(zmq.PULL)
        self.client_cmd_receiver.setsockopt(zmq.CONFLATE, 1)
        self.client_cmd_receiver.setsockopt(zmq.LINGER, 0)
        curve.apply_server(self.client_cmd_receiver, cfg.connection, "action receiver")
        self.client_cmd_receiver.bind(f"tcp://*:{self.cfg.connection.action_receiver_port}")  # Clients send here
        
        self.robot_cmd_sender = self.context.socket(zmq.PUSH)
        self.robot_cmd_sender.setsockopt(zmq.CONFLATE, 1)
        self.robot_cmd_sender.setsockopt(zmq.LINGER, 0)
        # IMMEDIATE=1 makes PUSH only enqueue to completed connections, so a
        # controller talking to the relay while no robot is attached won't
        # accumulate a stale message that hits the next robot to connect
        # (which previously caused the robot to auto-launch the last
        # controller-issued skill on startup).
        self.robot_cmd_sender.setsockopt(zmq.IMMEDIATE, 1)
        curve.apply_server(self.robot_cmd_sender, cfg.connection, "action sender")
        self.robot_cmd_sender.bind(f"tcp://*:{self.cfg.connection.action_sender_port}")  # Robot connects here

        # Setup image relay (Robot → Relay → Client)
        self.robot_state_receiver = self.context.socket(zmq.SUB)
        self.robot_state_receiver.setsockopt(zmq.CONFLATE, 1)
        self.robot_state_receiver.setsockopt(zmq.LINGER, 0)
        curve.apply_server(self.robot_state_receiver, cfg.connection, "state receiver")
        self.robot_state_receiver.bind(f"tcp://*:{self.cfg.connection.state_receiver_port}")  # Robot publishes here
        self.robot_state_receiver.setsockopt(zmq.SUBSCRIBE, b"")
        
        self.client_state_sender = self.context.socket(zmq.PUB)
        self.client_state_sender.setsockopt(zmq.CONFLATE, 1)
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

        self._processors = Observation.create_processors(DataComputer.SERVER)

    def _action_relay(self):
        """Forward actions from clients to robot, dispatching events to server processors."""
        poller = zmq.Poller()
        poller.register(self.client_cmd_receiver, zmq.POLLIN)
        while self.running:
            try:
                if not poller.poll(timeout=1):
                    continue
                raw = self.client_cmd_receiver.recv()
                try:
                    msg = Message.deserialize(raw)

                    if "stop_tracking" in msg.data:
                        for proc in self._processors.values():
                            proc.on_event("stop_tracking")
                    elif "click" in msg.data:
                        for proc in self._processors.values():
                            proc.on_event("click", msg.data["click"]["point"])
                        self.captured_click = msg.data["click"]["id"]
                    else:
                        self.captured_click = None

                    self.robot_cmd_sender.send(msg.serialize())

                except Exception as e:
                    print(f"Error processing message: {e}")

            except Exception as e:
                print(f"Error in action relay: {e}")
                time.sleep(0.001)

    def _state_relay(self):
        """Forward observations from robot to clients, computing server-side fields."""
        poller = zmq.Poller()
        poller.register(self.robot_state_receiver, zmq.POLLIN)
        while self.running:
            try:
                if not poller.poll(timeout=1):
                    continue
                raw = self.robot_state_receiver.recv()

                msg = Message.deserialize(raw)
                if self.captured_click is not None:
                    msg.data["click_ack"] = self.captured_click
                
                if self._processors:
                    obs_data = msg.data.get("observation")
                    if obs_data is None:    
                        start_time = time.time_ns()
                        observation = Observation(obs_data)
                        observation = self.enrich_observation(observation)
                        msg.data["observation"] = observation.to_dict()
                        elapsed_ms = (time.time_ns() - start_time) / 1e6
                        StatusBoard.set("Server observation enrichment time (ms)", elapsed_ms)
                self.client_state_sender.send(msg.serialize())
            except zmq.Again:
                pass

    def enrich_observation(self, observation: Observation) -> Observation:
        """Enrich the observation by running server-side processors."""
        for (module_name, field_name), proc in self._processors.items():
            observation[module_name][field_name] = proc(observation[module_name])
        return observation

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
        self.context.destroy()

