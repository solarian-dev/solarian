import zmq
import threading
import time
import cv2
import numpy as np

from proto.perception.vision.image_processor import ImageProcessor
from proto.control import Message
from proto.common.configs import ValueName, ServerConfig
from proto.robots.observation import Observation
from proto.common.utils import StatusBoard


class RelayServer:
    def __init__(self, cfg: ServerConfig):
        self.context = zmq.Context()
        self.running = False
        self.cfg = cfg
        self.captured_click = None

        self.image_processor_lock = threading.Lock()
        
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

        if self.cfg.image_processing and self.cfg.tracking_model:
            self.image_processor = ImageProcessor(self.cfg.tracking_model, self.cfg.depth_model, self.cfg.ref_x, self.cfg.ref_y)

    def _action_relay(self):
        """Forward actions from clients to robot"""
        while self.running:
            try:
                bytes = self.client_cmd_receiver.recv(zmq.NOBLOCK)
                try:
                    msg = Message.deserialize(bytes)
                    # print("Message received (relay): ", msg)
                    
                    # TODO use enumeration instead of string for message
                    if self.cfg.image_processing and "stop_tracking" in msg.data:
                        with self.image_processor_lock:
                            self.image_processor.stop_tracking()
                    elif self.cfg.image_processing and "click" in msg.data:
                        # print(f"Tracking point (relay): {msg.data['click']['point']}")
                        with self.image_processor_lock:
                            self.image_processor.track_object(msg.data["click"]["point"])
                        self.captured_click = msg.data["click"]["id"]
                    else:
                        self.captured_click = None
                    
                    # Reserialize before sending
                    self.robot_cmd_sender.send(msg.serialize())
                    
                except Exception as e:
                    print(f"Error processing message: {e}")
                    
            except zmq.Again:
                time.sleep(0.001)
            except Exception as e:
                print(f"Error in action relay: {e}")
                time.sleep(0.001)

    def _state_relay(self):
        """Forward images from robot to clients"""
        while self.running:
            try:
                bytes = self.robot_state_receiver.recv(zmq.NOBLOCK)
                msg = Message.deserialize(bytes)
                # print("Message state received (relay): ", msg)
                if self.cfg.image_processing:
                    # print("Processing image")
                    start_time = time.time_ns()
                    msg = self.process_image(msg, self.cfg.image_processing)
                    end_time = time.time_ns()
                    # print(f"Image processing time: {(end_time - start_time) / 1e6} ms")
                    StatusBoard.set("Image processing time (ms)", (end_time - start_time) / 1e6)

                self.client_state_sender.send(msg.serialize())
            except zmq.Again:
                time.sleep(0.001)
            # except Exception as e:
            #     print(f"Error in image relay: {e}")
            #     time.sleep(0.001)

    def process_image(self, msg: Message, cams_to_process):
        obs_data = msg.data["observation"]
        observation = Observation.model_validate(obs_data)

        for module_name in observation.data.keys():
            for sensor_name in observation.data[module_name].keys():
                if observation.data[module_name][sensor_name].get(ValueName.IMAGE) is not None and sensor_name in cams_to_process:
                    frame = cv2.imdecode(np.frombuffer(observation.data[module_name][sensor_name][ValueName.IMAGE], dtype=np.uint8),
                        cv2.IMREAD_COLOR)
                    with self.image_processor_lock:
                        tracked_box = self.image_processor.process_image(frame)
                    observation.data[module_name][sensor_name][ValueName.BOUNDING_BOX] = tracked_box

        if self.captured_click is not None:
            msg.data["click_ack"] = self.captured_click

        msg.data["observation"] = observation
        return msg

    def start(self):
        self.running = True
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

