import cv2
import zmq
import msgpack
import time
import threading
import json
import argparse
from pathlib import Path

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus, TorqueMode

from config import (
    LeKiwiWithKochRobotConfig,
    DummyRobotConfig
)


def calibrate_follower_arm(motors_bus, calib_dir_str):
    """
    Calibrates the follower arm. Attempts to load an existing calibration file;
    if not found, runs manual calibration and saves the result.
    """
    calib_dir = Path(calib_dir_str)
    calib_dir.mkdir(parents=True, exist_ok=True)
    calib_file = calib_dir / "main_follower.json"
    try:
        from lerobot.common.robot_devices.robots.feetech_calibration import run_arm_manual_calibration
    except ImportError:
        print("[WARNING] Calibration function not available. Skipping calibration.")
        return

    if calib_file.exists():
        with open(calib_file) as f:
            calibration = json.load(f)
        print(f"[INFO] Loaded calibration from {calib_file}")
    else:
        print("[INFO] Calibration file not found. Running manual calibration...")
        calibration = run_arm_manual_calibration(motors_bus, "lekiwi", "follower_arm", "follower")
        print(f"[INFO] Calibration complete. Saving to {calib_file}")
        with open(calib_file, "w") as f:
            json.dump(calibration, f)
    try:
        motors_bus.set_calibration(calibration)
        print("[INFO] Applied calibration for follower arm.")
    except Exception as e:
        print(f"[WARNING] Could not apply calibration: {e}")



class RobotController:
    def __init__(self, robot_config, relay_server_ip):
        self.context = zmq.Context()
        self.robot_config = robot_config
        
        # Command receiver (PULL)
        self.command_socket = self.context.socket(zmq.PULL)
        self.command_socket.setsockopt(zmq.CONFLATE, 1)
        self.command_socket.connect(f"tcp://{relay_server_ip}:5555")
        
        # # State publisher (PUB)
        # self.state_socket = self.context.socket(zmq.PUB)
        # self.state_socket.setsockopt(zmq.SNDHWM, 1)
        # self.state_socket.connect(f"tcp://{relay_server_ip}:5556")
        
        # Image publisher (PUB)
        self.image_socket = self.context.socket(zmq.PUB)
        self.image_socket.setsockopt(zmq.CONFLATE, 1)
        self.image_socket.connect(f"tcp://{relay_server_ip}:5557")
        
        self.current_command = None
        self.current_velocity = None
        self.current_arm_state = None
        
        self.running = False
        self.verbose = True
        
        # Cameras setup
        self.cameras = self.setup_cameras(robot_config.cameras)
        print("Cameras: ", self.cameras)


        if not isinstance(robot_config, DummyRobotConfig):
            # Motors setup
            motors_bus_config = robot_config.follower_arms.get("main")
            self.motors_bus = self.setup_motors_bus(motors_bus_config)


            # # Disable torque for each arm motor.
            for motor in self.motors_bus.motor_names:
                self.motors_bus.write("Torque_Enable", TorqueMode.DISABLED.value, motor)
            
            # # Calibrate the follower arm.
            calibrate_follower_arm(self.motors_bus, robot_config.calibration_dir)

            # Initialize motors in velocity mode.
            self.motors_bus.write("Lock", 0)
            self.motors_bus.write("Mode", [1, 1, 1], robot_config.base_motor_ids)
            self.motors_bus.write("Lock", 1)

        
    def setup_motors_bus(self, motor_config):
        # Initialize the motors bus using the follower arm configuration.
        
        if motor_config is None:
            print("[ERROR] Follower arm 'main' configuration not found.")
            return
        motors_bus = FeetechMotorsBus(motor_config)
        motors_bus.connect()
        return motors_bus
    
    # def setup_cameras(self, cam_ids):
    #     """Initialize multiple webcams with optimal settings"""
    #     cameras = []
    #     for cam_id in cam_ids:
    #         cap = cv2.VideoCapture(cam_id)
    #         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    #         cap.set(cv2.CAP_PROP_FPS, 30)
    #         cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    #         if cap.isOpened():
    #             cameras.append(cap)
    #             if self.verbose:
    #                 print(f"Camera {cam_id} initialized")
    #         else:
    #             print(f"Warning: Camera {cam_id} failed to initialize!")
        
    def setup_cameras(self, cameras_config):
        # Initialize cameras from the robot configuration.
        cameras = make_cameras_from_configs(cameras_config)
        for cam in cameras.values():
            cam.connect()    
        return cameras

    def process_arm_command(self, data):
        """Process arm-related commands"""

        # if isinstance(self.robot_config, DummyRobotConfig):
        #     self.current_arm_state = []
        #     return
        
        if "arm_positions" in data:
            arm_positions = data["arm_positions"]
            if not isinstance(arm_positions, list):
                print(f"[ERROR] Invalid arm_positions: {arm_positions}")
            elif len(arm_positions) != len(self.robot_config.arm_motor_ids):
                print(
                    f"[WARNING] Received {len(arm_positions)} arm positions, expected {len(self.robot_config.arm_motor_ids)}"
                )
            else:
                for motor, pos in zip(self.robot_config.arm_motor_ids, arm_positions, strict=False):
                    self.motors_bus.write("Goal_Position", pos, motor)
        self.current_arm_state = self.read_follower_arm_state()

    def process_base_command(self, data):
        """Process base-related commands"""

        # if isinstance(self.robot_config, DummyRobotConfig):
        #     return
        
        if "raw_velocity" in data:
            raw_command = data["raw_velocity"]
            # Expect keys: "left_wheel", "back_wheel", "right_wheel".
            # command_speeds = [
            #     int(raw_command.get("left_wheel", 0)),
            #     int(raw_command.get("back_wheel", 0)),
            #     int(raw_command.get("right_wheel", 0)),
            # ]
            command_speeds = [int(raw_command.get(motor, 0)) for motor in self.robot_config.base_motor_ids]
            self.set_velocity(command_speeds)
            self.current_velocity = self.read_velocity()
            
            
    
    def _command_handler(self):
        """Thread for receiving commands"""
        while self.running:
            try:
                msg = self.command_socket.recv(zmq.NOBLOCK)
                command = msgpack.unpackb(msg)
                self.current_command = command
                if self.verbose:
                    latency = (time.time_ns() - command["timestamp"])/1e6
                    # print(f"Received command: {command}")
                    # print(f"Command latency: {latency:.2f}ms")
                self.process_arm_command(command)
                self.process_base_command(command)
            except zmq.Again:
                time.sleep(0.001)

    # def _state_publisher(self):
    #     """Thread for publishing robot state"""
    #     while self.running:
    #         # Read actual joint positions from sensors (simulated here)
    #         state = {
    #             "timestamp": time.time_ns(),
                
    #             # Add other sensor data here
    #         }
    #         self.state_socket.send(msgpack.packb(state))
    #         time.sleep(0.01)  # 100Hz update rate

    def read_velocity(self):
        """
        Reads the raw speeds for all wheels. Returns a dictionary with motor names:
        """
        if isinstance(self.robot_config, DummyRobotConfig):
            return {motor: 0 for motor in self.robot_config.base_motor_ids}
        
        raw_speeds = self.motors_bus.read("Present_Speed", self.robot_config.base_motor_ids)
        # return {
        #     "left_wheel": int(raw_speeds[0]),
        #     "back_wheel": int(raw_speeds[1]),
        #     "right_wheel": int(raw_speeds[2]),
        # }
        return {motor: int(speed) for motor, speed in zip(self.robot_config.base_motor_ids, raw_speeds)}

    def set_velocity(self, command_speeds):
        """
        Sends raw velocity commands (16-bit encoded values) directly to the motor bus.
        The order of speeds must correspond to self.motor_ids.
        """

        if isinstance(self.robot_config, DummyRobotConfig):
            return
        self.motors_bus.write("Goal_Speed", command_speeds, self.robot_config.base_motor_ids)
        
    
    def read_follower_arm_state(self):
        """Read state of the follower arm"""

        # Read the follower arm state from the motors bus.
        follower_arm_state = []

        for motor in self.robot_config.arm_motor_ids:
            try:
                pos = self.motors_bus.read("Present_Position", motor)
                # Convert the position to a float (or use as is if already numeric).
                follower_arm_state.append(float(pos) if not isinstance(pos, (int, float)) else pos)
            except Exception as e:
                print(f"[ERROR] Reading motor {motor} failed: {e}")

        return follower_arm_state


    def _image_publisher(self):
        """Thread for capturing and sending images"""
        while self.running:
            frames = {}
            for name, cam in self.cameras.items():
                frame = cam.read()    
                # Compress image (adjust quality as needed)
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frames[name] = jpeg.tobytes()
            observation = {
                "images": frames,
                "present_speed": self.current_velocity,
                "follower_arm_state": self.current_arm_state,
                "timestamp": time.time_ns()
            }
                
            self.image_socket.send(msgpack.packb(observation))
            # print("Sent observation", observation)
            
            time.sleep(0.033)  # ~30 FPS


    def _control_loop(self):
        """Main robot control logic (simulated)"""
        while self.running:
            if self.current_command:
                # Implement actual motor control here
                
                # Add wheel control logic
                time.sleep(0.005)

    def start(self):
        """Start all threads"""
        self.running = True
        threading.Thread(target=self._command_handler, daemon=True).start()
        # threading.Thread(target=self._state_publisher, daemon=True).start()
        threading.Thread(target=self._image_publisher, daemon=True).start()
        # threading.Thread(target=self._control_loop, daemon=True).start()

    def stop(self):
        """Clean shutdown"""
        self.running = False

        # Release cameras
        for cam in self.cameras.values():
            cam.release()

        # Stops the robot by setting all motor speeds to zero.
        self.motor_bus.write("Goal_Speed", [0, 0, 0], self.robot_config.base_motor_ids)
        print("Motors stopped.")

        self.command_socket.close()
        # self.state_socket.close()
        self.image_socket.close()
        self.context.term()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot controller that connects to a relay server')
    parser.add_argument('--server-ip', type=str, default="127.0.0.1",
                      help='IP address of the relay server (default: 127.0.0.1)')
    args = parser.parse_args()

    # robot_config = LeKiwiWithKochRobotConfig()
    robot_config = DummyRobotConfig() 
    robot = RobotController(robot_config, relay_server_ip=args.server_ip)
    robot.start()
    try:
        while True:  # Keep main thread alive
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down robot controller...")
        robot.stop()