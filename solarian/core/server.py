import zmq
import threading
import time

from solarian.core.message import Message
from solarian.core.configs import ServerConfig
from solarian.core.data import DataComputer
from solarian.core import Observation
from solarian.utils import StatusBoard

# TO DO: rename state to observation, rename cmd to action

class RelayServer:
    def __init__(self, cfg: ServerConfig):
        self.context = zmq.Context()
        self.running = False
        self.cfg = cfg
        self.captured_click = None
        
        # Setup action relay (Client → Relay → Robot)
        self.client_cmd_receiver = self.context.socket(zmq.PULL)
        self.client_cmd_receiver.setsockopt(zmq.CONFLATE, 1)
        self.client_cmd_receiver.setsockopt(zmq.LINGER, 0)
        self.client_cmd_receiver.bind(f"tcp://*:{self.cfg.connection.action_receiver_port}")  # Clients send here
        
        self.robot_cmd_sender = self.context.socket(zmq.PUSH)
        self.robot_cmd_sender.setsockopt(zmq.CONFLATE, 1)
        self.robot_cmd_sender.setsockopt(zmq.LINGER, 0)
        self.robot_cmd_sender.bind(f"tcp://*:{self.cfg.connection.action_sender_port}")  # Robot connects here

        # Setup image relay (Robot → Relay → Client)
        self.robot_state_receiver = self.context.socket(zmq.SUB)
        self.robot_state_receiver.setsockopt(zmq.CONFLATE, 1)
        self.robot_state_receiver.setsockopt(zmq.LINGER, 0)
        self.robot_state_receiver.bind(f"tcp://*:{self.cfg.connection.state_receiver_port}")  # Robot publishes here
        self.robot_state_receiver.setsockopt(zmq.SUBSCRIBE, b"")
        
        self.client_state_sender = self.context.socket(zmq.PUB)
        self.client_state_sender.setsockopt(zmq.CONFLATE, 1)
        self.client_state_sender.setsockopt(zmq.LINGER, 0)
        self.client_state_sender.bind(f"tcp://*:{self.cfg.connection.state_sender_port}")  # Clients subscribe here

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

    def start(self):
        self.running = True
        # TO DO: action relay and state relay should be implemented as processes instead of threads
        threading.Thread(target=self._action_relay, daemon=True).start()
        threading.Thread(target=self._state_relay, daemon=True).start()
        print("Relay server started")

        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.running = False
        self.context.destroy()

