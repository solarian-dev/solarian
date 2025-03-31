# relay_server.py
import zmq
import threading
import time

class RelayServer:
    def __init__(self):
        self.context = zmq.Context()
        self.running = False

        # Setup command relay (Client → Relay → Robot)
        self.client_cmd_receiver = self.context.socket(zmq.PULL)
        self.client_cmd_receiver.bind("tcp://*:5560")  # Clients send here
        
        self.robot_cmd_sender = self.context.socket(zmq.PUSH)
        self.robot_cmd_sender.bind("tcp://*:5555")  # Robot connects here

        # Setup state relay (Robot → Relay → Client)
        self.robot_state_receiver = self.context.socket(zmq.SUB)
        self.robot_state_receiver.bind("tcp://*:5556")  # Robot publishes here
        self.robot_state_receiver.setsockopt(zmq.SUBSCRIBE, b"")
        
        self.client_state_sender = self.context.socket(zmq.PUB)
        self.client_state_sender.bind("tcp://*:5561")  # Clients subscribe here

        # Setup image relay (Robot → Relay → Client)
        self.robot_image_receiver = self.context.socket(zmq.SUB)
        self.robot_image_receiver.bind("tcp://*:5557")  # Robot publishes here
        self.robot_image_receiver.setsockopt(zmq.SUBSCRIBE, b"")
        
        self.client_image_sender = self.context.socket(zmq.PUB)
        self.client_image_sender.bind("tcp://*:5562")  # Clients subscribe here

    def _command_relay(self):
        """Forward commands from clients to robot"""
        while self.running:
            try:
                cmd = self.client_cmd_receiver.recv(zmq.NOBLOCK)
                self.robot_cmd_sender.send(cmd)
            except zmq.Again:
                time.sleep(0.001)

    def _state_relay(self):
        """Forward state updates from robot to clients"""
        while self.running:
            try:
                state = self.robot_state_receiver.recv(zmq.NOBLOCK)
                self.client_state_sender.send(state)
            except zmq.Again:
                time.sleep(0.001)

    def _image_relay(self):
        """Forward images from robot to clients"""
        while self.running:
            try:
                img = self.robot_image_receiver.recv(zmq.NOBLOCK)
                self.client_image_sender.send(img)
            except zmq.Again:
                time.sleep(0.001)

    def start(self):
        self.running = True
        threading.Thread(target=self._command_relay, daemon=True).start()
        threading.Thread(target=self._state_relay, daemon=True).start()
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