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
        
        # Setup command relay (Client → Relay → Robot)
        self.client_cmd_receiver = self.context.socket(zmq.PULL)
        self.client_cmd_receiver.setsockopt(zmq.CONFLATE, 1)
        self.client_cmd_receiver.setsockopt(zmq.LINGER, 0)
        self.client_cmd_receiver.bind("tcp://*:5560")  # Clients send here
        
        self.robot_cmd_sender = self.context.socket(zmq.PUSH)
        self.robot_cmd_sender.setsockopt(zmq.CONFLATE, 1)
        self.robot_cmd_sender.setsockopt(zmq.LINGER, 0)
        self.robot_cmd_sender.bind("tcp://*:5555")  # Robot connects here

        # Setup state relay (Robot → Relay → Client)
        # self.robot_state_receiver = self.context.socket(zmq.SUB)
        # self.robot_state_receiver.bind("tcp://*:5556")  # Robot publishes here
        # self.robot_state_receiver.setsockopt(zmq.SUBSCRIBE, b"")
        
        # self.client_state_sender = self.context.socket(zmq.PUB)
        # self.client_state_sender.bind("tcp://*:5561")  # Clients subscribe here

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
                cmd = self.client_cmd_receiver.recv(zmq.NOBLOCK)
                # print(f"Received command: {cmd}")  # Debug print
                self.robot_cmd_sender.send(cmd)
            except zmq.Again:
                time.sleep(0.001)

    # def _state_relay(self):
    #     """Forward state updates from robot to clients"""
    #     while self.running:
    #         try:
    #             state = self.robot_state_receiver.recv(zmq.NOBLOCK)
    #             self.client_state_sender.send(state)
    #         except zmq.Again:
    #             time.sleep(0.001)

    def _image_relay(self):
        """Forward images from robot to clients"""
        while self.running:
            try:
                msg = self.robot_image_receiver.recv(zmq.NOBLOCK)
                if self.robot_config.image_processing:
                    msg = self.process_image(msg, self.robot_config.image_processing)
                self.client_image_sender.send(msg)
            except zmq.Again:
                time.sleep(0.001)
            except Exception as e:
                print(f"Error in image relay: {e}")
                time.sleep(0.001)


    def process_image(self, msg, cams_to_process):
        observation = msgpack.unpackb(msg)
        images_dict = observation.get("images", {})
        frames = {}
        for cam_name, image in images_dict.items():
            if image:
                if cam_name in cams_to_process:
                    image = self.image_processor.process_image(image)
            frames[cam_name] = image
        observation["images"] = frames
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
    relay = RelayServer(robot_config=robot_config, image_processor_model="yoloe-11s-seg.pt")
    # relay = RelayServer(robot_config=robot_config, image_processor_model="yolov8s-world.pt")
    relay.start()
    print("Relay server started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        relay.stop()