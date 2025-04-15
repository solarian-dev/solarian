import zmq
import threading
import time
import msgpack
from image_processor import ImageProcessor
from config import (
    LeKiwiWithKochRobotConfig,
    DummyRobotConfig
)

class RelayServer:
    def __init__(self, robot_config, image_processor_model=None):
        self.context = zmq.Context()
        self.running = False
        self.robot_config = robot_config
        self.captured_click = None
        
        # Setup command relay (Client → Relay → Robot)
        self.client_cmd_receiver = self.context.socket(zmq.PULL)
        self.client_cmd_receiver.setsockopt(zmq.CONFLATE, 1)
        self.client_cmd_receiver.setsockopt(zmq.LINGER, 0)
        self.client_cmd_receiver.bind("tcp://*:5560")  # Clients send here
        
        self.robot_cmd_sender = self.context.socket(zmq.PUSH)
        self.robot_cmd_sender.setsockopt(zmq.CONFLATE, 1)
        self.robot_cmd_sender.setsockopt(zmq.LINGER, 0)
        self.robot_cmd_sender.bind("tcp://*:5555")  # Robot connects here

        # Setup image relay (Robot → Relay → Client)
        self.robot_image_receiver = self.context.socket(zmq.SUB)
        self.robot_image_receiver.setsockopt(zmq.CONFLATE, 1)
        self.robot_image_receiver.setsockopt(zmq.LINGER, 0)
        self.robot_image_receiver.bind("tcp://*:5557")  # Robot publishes here
        self.robot_image_receiver.setsockopt(zmq.SUBSCRIBE, b"")
        
        self.client_image_sender = self.context.socket(zmq.PUB)
        self.client_image_sender.setsockopt(zmq.CONFLATE, 1)
        self.client_image_sender.setsockopt(zmq.LINGER, 0)
        self.client_image_sender.bind("tcp://*:5562")  # Clients subscribe here

        if self.robot_config.image_processing and image_processor_model:
            self.image_processor = ImageProcessor(image_processor_model)

    def _command_relay(self):
        """Forward commands from clients to robot"""
        while self.running:
            try:
                msg = self.client_cmd_receiver.recv(zmq.NOBLOCK)

                if self.image_processor:
                    try:
                        # Try to unpack the message
                        cmd = msgpack.unpackb(msg, raw=False)
                        # print(f"Received command: {cmd}")
                    except msgpack.exceptions.ExtraData:
                        # If there's extra data, only unpack the first object
                        cmd = msgpack.unpackb(msg, raw=False, strict_map_key=False)
                    except Exception as e:
                        print(f"Error unpacking message: {e}")
                        cmd = None

                    if cmd and isinstance(cmd, dict) and "click" in cmd:
                        print(f"Tracking point (relay): {cmd['click']['point']}")
                        self.image_processor.track_object(cmd["click"]["point"])
                        self.captured_click = cmd["click"]["id"]
                    else:
                        self.captured_click = None
                
                self.robot_cmd_sender.send(msg)
            except zmq.Again:
                time.sleep(0.001)
            except Exception as e:
                print(f"Error in command relay: {e}")
                time.sleep(0.001)

    def _image_relay(self):
        """Forward images from robot to clients"""
        while self.running:
            try:
                msg = self.robot_image_receiver.recv(zmq.NOBLOCK)
                if self.robot_config.image_processing:
                    # print("Processing image")
                    start_time = time.time_ns()
                    msg = self.process_image(msg, self.robot_config.image_processing)
                    end_time = time.time_ns()
                    print(f"Image processing time: {(end_time - start_time) / 1e6} ms")
                self.client_image_sender.send(msg)
            except zmq.Again:
                time.sleep(0.001)
            except Exception as e:
                print(f"Error in image relay: {e}")
                time.sleep(0.001)

    def process_image(self, msg, cams_to_process):
        observation = msgpack.unpackb(msg)
        images_dict = observation.get("images", {})
        observation["objects"] = {}
        frames = {}
        for cam_name, image in images_dict.items():
            if image:
                if cam_name in cams_to_process:
                    # print(f"Processing image for {cam_name}")
                    image, tracked_box = self.image_processor.process_image(image)
                    observation["objects"][cam_name] = tracked_box
            # frames[cam_name] = image
        # observation["images"] = frames
        
        if self.captured_click is not None:
            observation["click_ack"] = self.captured_click
        return msgpack.packb(observation)

    def start(self):
        self.running = True
        threading.Thread(target=self._command_relay, daemon=True).start()
        # threading.Thread(target=self._state_relay, daemon=True).start()
        threading.Thread(target=self._image_relay, daemon=True).start()

    def stop(self):
        self.running = False
        self.context.destroy()

if __name__ == "__main__":
    # robot_config = LeKiwiWithKochRobotConfig()
    robot_config = DummyRobotConfig()
    relay = RelayServer(robot_config=robot_config, image_processor_model="./models/FastSAM-s.pt")
    # relay = RelayServer(robot_config=robot_config, image_processor_model="./models/yoloe-11s-seg-pf.pt")
    # relay = RelayServer(robot_config=robot_config, image_processor_model="./models/yoloe-11s-seg.pt")
    # relay = RelayServer(robot_config=robot_config, image_processor_model="./models/yolov8s-world.pt")
    relay.start()
    print("Relay server started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        relay.stop()