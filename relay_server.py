# relay_server.py
import zmq
import threading
import time
import msgpack
import cv2
import numpy as np

import ultralytics
from ultralytics import YOLO

class RelayServer:
    def __init__(self, do_image_processing=True):
        self.context = zmq.Context()
        self.running = False
        self.do_image_processing = do_image_processing

        # Setup command relay (Client → Relay → Robot)
        self.client_cmd_receiver = self.context.socket(zmq.PULL)
        self.client_cmd_receiver.bind("tcp://*:5560")  # Clients send here
        
        self.robot_cmd_sender = self.context.socket(zmq.PUSH)
        self.robot_cmd_sender.bind("tcp://*:5555")  # Robot connects here

        # Setup state relay (Robot → Relay → Client)
        # self.robot_state_receiver = self.context.socket(zmq.SUB)
        # self.robot_state_receiver.bind("tcp://*:5556")  # Robot publishes here
        # self.robot_state_receiver.setsockopt(zmq.SUBSCRIBE, b"")
        
        # self.client_state_sender = self.context.socket(zmq.PUB)
        # self.client_state_sender.bind("tcp://*:5561")  # Clients subscribe here

        # Setup image relay (Robot → Relay → Client)
        self.robot_image_receiver = self.context.socket(zmq.SUB)
        self.robot_image_receiver.bind("tcp://*:5557")  # Robot publishes here
        self.robot_image_receiver.setsockopt(zmq.SUBSCRIBE, b"")
        
        self.client_image_sender = self.context.socket(zmq.PUB)
        self.client_image_sender.bind("tcp://*:5562")  # Clients subscribe here

        if self.do_image_processing:
            self.object_det_model = YOLO("yolov8x-worldv2.pt")
            classes = ["toy lego block"]
            self.object_det_model.set_classes(classes)

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
                if self.do_image_processing:
                    msg = self.process_image(msg, ["front"])
                self.client_image_sender.send(msg)
            except zmq.Again:
                time.sleep(0.001)


    def process_image(self, msg, cams_to_process):
        unpacked = msgpack.unpackb(msg)

        observation = unpacked
        # print("Speed: ", observation.get("present_speed", {}))
        # print("Arm state: ", observation.get("follower_arm_state", None))
        images_dict = observation.get("images", {})
        # new_speed = observation.get("present_speed", {})
        # new_arm_state = observation.get("follower_arm_state", None)
        timestamp = observation.get("timestamp", None)

        frames = {}
        for cam_name, image in images_dict.items():
            if image:
                if cam_name in cams_to_process:
                    # Decode JPEG images
                    image = cv2.imdecode(
                        np.frombuffer(image, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    results = self.object_det_model.track(image, device='mps', conf=0.01, max_det=1, persist=True)
                    # Visualize the results on the frame
                    image = results[0].plot(labels=False)
                    # Compress image (adjust quality as needed)
                    _, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    image = jpeg.tobytes()

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
    relay = RelayServer()  
    relay.start()
    print("Relay server started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        relay.stop()