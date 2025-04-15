import cv2
import numpy as np
import torch
import zmq
import msgpack
import time
import threading
from pynput import keyboard
import json
import argparse
import random
import string
from pathlib import Path
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus, TorqueMode
from config import (
    LeKiwiWithKochRobotConfig,
    DummyRobotConfig
)

def calibrate_arm(motors_bus, calib_dir_str):
    """
    Calibrates the arm. Attempts to load an existing calibration file;
    if not found, runs manual calibration and saves the result.
    """
    calib_dir = Path(calib_dir_str)
    calib_dir.mkdir(parents=True, exist_ok=True)
    #RG TODO use name and arm type from config
    calib_file = calib_dir / "main_leader.json"
    try:
        from lerobot.common.robot_devices.robots.dynamixel_calibration import run_arm_calibration
    except ImportError:
        print("[WARNING] Calibration function not available. Skipping calibration.")
        return

    if calib_file.exists():
        with open(calib_file) as f:
            calibration = json.load(f)
        print(f"[INFO] Loaded calibration from {calib_file}")
    else:
        print("[INFO] Calibration file not found. Running manual calibration...")
        calibration = run_arm_manual_calibration(motors_bus, "lekiwi", "leader_arm", "leader")
        print(f"[INFO] Calibration complete. Saving to {calib_file}")
        with open(calib_file, "w") as f:
            json.dump(calibration, f)
    try:
        motors_bus.set_calibration(calibration)
        print("[INFO] Applied calibration for leader arm.")
    except Exception as e:
        print(f"[WARNING] Could not apply calibration: {e}")
    

class MobileManipulator:
    def __init__(self, relay_server_ip, robot_config):
        self.relay_server_ip = relay_server_ip
        self.context = zmq.Context()
        self.robot_config = robot_config

        self.clicked_point = None
        self.clicked_id = None
        
        # Command socket
        self.command_socket = self.context.socket(zmq.PUSH)
        self.command_socket.setsockopt(zmq.CONFLATE, 1)
        self.command_socket.setsockopt(zmq.LINGER, 0)
        self.command_socket.connect(f"tcp://{self.relay_server_ip}:5560")
        
        # Image subscription
        self.image_socket = self.context.socket(zmq.SUB)
        self.image_socket.setsockopt(zmq.CONFLATE, 1)
        self.image_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.image_socket.setsockopt(zmq.LINGER, 0)
        self.image_socket.connect(f"tcp://{self.relay_server_ip}:5562")

        # Verify connection
        self._verify_connection()
        
        self.latest_state = None
        self.latest_images = None
        self.image_lock = threading.Lock()  # Thread-safe image access
        self.running = False

        self.last_frames = {}
        self.last_objects = {}
        self.last_present_speed = {}
        self.last_remote_arm_state = torch.zeros(6, dtype=torch.float32)
        self.last_timestamp = None

        self.pressed_keys = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "rotate_left": False,
            "rotate_right": False,
        }

        self.teleop_keys = {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # quit teleop
            "quit": "q",
        }

        # Define three speed levels and a current index
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},  # slow
            {"xy": 0.2, "theta": 60},  # medium
            {"xy": 0.3, "theta": 90},  # fast
        ]
        self.speed_index = 0  # Start at slow

        
        if not isinstance(robot_config, DummyRobotConfig):
            # Motors setup
            motors_bus_config = robot_config.leader_arms.get("main")
            self.motors_bus = self.setup_motors_bus(motors_bus_config)

            # # Disable torque for each arm motor.
            for motor in self.motors_bus.motor_names:
                self.motors_bus.write("Torque_Enable", TorqueMode.DISABLED.value, motor)

            # Calibrate the leader arm.
            calibrate_arm(self.motors_bus, robot_config.calibration_dir)

            # Apply Koch preset to leader arms
            if robot_config.gripper_open_degree is not None:
                print(f"Applying Koch preset to leader arm.")
                self.motors_bus.write("Torque_Enable", 1, "gripper")
                self.motors_bus.write("Goal_Position", robot_config.gripper_open_degree, "gripper")

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )

        self.listener.start()

    # Mouse callback function
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at (x={x}, y={y})")
            self.clicked_point = (x, y)
            # generate random alphanumeric id
            self.clicked_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    def _verify_connection(self, timeout=5000):
        """Verify that we can connect to the relay server"""
        print(f"Verifying connection to relay server at {self.relay_server_ip}...")
        
        # Create a temporary PUSH socket to test connection
        test_socket = self.context.socket(zmq.PUSH)
        test_socket.setsockopt(zmq.LINGER, 0)
        print(f"Attempting to connect to tcp://{self.relay_server_ip}:5560...")
        test_socket.connect(f"tcp://{self.relay_server_ip}:5560")
        
        try:
            # Try to send a message
            print("Attempting to send test message...")
            test_socket.send_string("test", zmq.NOBLOCK)
            print("Test message sent successfully")
            time.sleep(1)  # Give it a moment to ensure the message was sent
            return True
        except zmq.ZMQError as e:
            print(f"ZMQ error occurred: {e}")
            raise ConnectionError(f"ZMQ error while connecting to relay server: {e}")
        finally:
            print("Cleaning up test socket...")
            test_socket.close()
        
        print("Successfully connected to relay server")


    def setup_motors_bus(self, motor_config):
        # Initialize the motors bus using the leader arm configuration.
        
        if motor_config is None:
            print("[ERROR] Leader arm 'main' configuration not found.")
            return
        motors_bus = DynamixelMotorsBus(motor_config)
        motors_bus.connect()
        return motors_bus

    def on_press(self, key):
        try:
            # Movement
            if key.char == self.teleop_keys["forward"]:
                self.pressed_keys["forward"] = True
            elif key.char == self.teleop_keys["backward"]:
                self.pressed_keys["backward"] = True
            elif key.char == self.teleop_keys["left"]:
                self.pressed_keys["left"] = True
            elif key.char == self.teleop_keys["right"]:
                self.pressed_keys["right"] = True
            elif key.char == self.teleop_keys["rotate_left"]:
                self.pressed_keys["rotate_left"] = True
            elif key.char == self.teleop_keys["rotate_right"]:
                self.pressed_keys["rotate_right"] = True

            # Quit teleoperation
            elif key.char == self.teleop_keys["quit"]:
                self.running = False
                return False

            # Speed control
            elif key.char == self.teleop_keys["speed_up"]:
                self.speed_index = min(self.speed_index + 1, 2)
                print(f"Speed index increased to {self.speed_index}")
            elif key.char == self.teleop_keys["speed_down"]:
                self.speed_index = max(self.speed_index - 1, 0)
                print(f"Speed index decreased to {self.speed_index}")

        except AttributeError:
            # e.g., if key is special like Key.esc
            if key == keyboard.Key.esc:
                self.running = False
                return False

    def on_release(self, key):
        try:
            if hasattr(key, "char"):
                if key.char == self.teleop_keys["forward"]:
                    self.pressed_keys["forward"] = False
                elif key.char == self.teleop_keys["backward"]:
                    self.pressed_keys["backward"] = False
                elif key.char == self.teleop_keys["left"]:
                    self.pressed_keys["left"] = False
                elif key.char == self.teleop_keys["right"]:
                    self.pressed_keys["right"] = False
                elif key.char == self.teleop_keys["rotate_left"]:
                    self.pressed_keys["rotate_left"] = False
                elif key.char == self.teleop_keys["rotate_right"]:
                    self.pressed_keys["rotate_right"] = False
        except AttributeError:
            pass

    def _image_receiver(self):
        """Receive and decode image streams"""

        frames = {}
        present_speed = {}
        remote_arm_state_tensor = torch.zeros(6, dtype=torch.float32)

        while self.running:
            try:
                msg = self.image_socket.recv(zmq.NOBLOCK)
                unpacked = msgpack.unpackb(msg)

                # print("Received image message")
                if unpacked.get("click_ack", None) and unpacked["click_ack"] == self.clicked_id:
                    print("Click acknowledged", self.clicked_id)
                    self.clicked_id = None
                    self.clicked_point = None
                    
                
                # Thread-safe update of images
                with self.image_lock:
                    observation = unpacked
                    # print("Observation: ", observation)
                    # print("Speed: ", observation.get("present_speed", {}))
                    # print("Arm state: ", observation.get("follower_arm_state", None))
                    images_dict = observation.get("images", {})
                    objects_dict = observation.get("objects", {})
                    new_speed = observation.get("present_speed", {})
                    new_arm_state = observation.get("follower_arm_state", None)
                    timestamp = observation.get("timestamp", None)

                    for cam_name, image in images_dict.items():
                        if image:
                            frames[cam_name] = image

                    # If remote_arm_state is None and frames is None there is no message then use the previous message
                    if new_arm_state is not None and frames is not None:
                        self.last_frames = frames
                        self.last_objects = objects_dict

                        remote_arm_state_tensor = torch.tensor(new_arm_state, dtype=torch.float32)
                        self.last_remote_arm_state = remote_arm_state_tensor

                        present_speed = new_speed
                        self.last_present_speed = new_speed

                        self.last_timestamp = timestamp
                    


            except zmq.Again:
                time.sleep(0.001)

            except Exception as e:
                print(f"[ERROR] Image receiver failed: {e}")

    def send_command(self, cmd):
        """Send control commands to robot"""
        cmd["timestamp"] = time.time_ns()
        self.command_socket.send(msgpack.packb(cmd))

    def get_images(self):
        """Get latest images with thread-safe access"""
        with self.image_lock:
            # return self.latest_images
            return self.last_frames

    def get_robot_state(self):
        """Get latest state"""
        with self.image_lock:
            return self.last_frames, self.last_objects, self.last_present_speed, self.last_remote_arm_state, self.last_timestamp

    def start(self):
        """Start receiver threads"""
        self.running = True
        # threading.Thread(target=self._state_receiver, daemon=True).start()
        threading.Thread(target=self._image_receiver, daemon=True).start()

    def stop(self):
        """Clean shutdown"""
        self.running = False
        self.command_socket.close()
        # self.state_socket.close()
        self.image_socket.close()
        self.context.term()

    @staticmethod
    def degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = abs(degps) * steps_per_deg
        speed_int = int(round(speed_in_steps))
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF
        if degps < 0:
            return speed_int | 0x8000
        else:
            return speed_int & 0x7FFF

    @staticmethod
    def raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed & 0x7FFF
        degps = magnitude / steps_per_deg
        if raw_speed & 0x8000:
            degps = -degps
        return degps

    def body_to_wheel_raw(
        self,
        x_cmd: float,
        y_cmd: float,
        theta_cmd: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> dict:
        """
        Convert desired body-frame velocities into wheel raw commands.

        Parameters:
          x_cmd      : Linear velocity in x (m/s).
          y_cmd      : Linear velocity in y (m/s).
          theta_cmd  : Rotational velocity (deg/s).
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the center of rotation to each wheel (meters).
          max_raw    : Maximum allowed raw command (ticks) per wheel.

        Returns:
          A dictionary with wheel raw commands:
             {"left_wheel": value, "back_wheel": value, "right_wheel": value}.

        Notes:
          - Internally, the method converts theta_cmd to rad/s for the kinematics.
          - The raw command is computed from the wheels angular speed in deg/s
            using degps_to_raw(). If any command exceeds max_raw, all commands
            are scaled down proportionally.
        """
        # Convert rotational velocity from deg/s to rad/s.
        theta_rad = theta_cmd * (np.pi / 180.0)
        # Create the body velocity vector [x, y, theta_rad].
        velocity_vector = np.array([x_cmd, y_cmd, theta_rad])

        # Define the wheel mounting angles (defined from y axis cw)
        angles = np.radians(np.array([300, 180, 60]))
        # Build the kinematic matrix: each row maps body velocities to a wheel’s linear speed.
        # The third column (base_radius) accounts for the effect of rotation.
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Compute each wheel’s linear speed (m/s) and then its angular speed (rad/s).
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # Convert wheel angular speeds from rad/s to deg/s.
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        # Scaling
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # Convert each wheel’s angular speed (deg/s) to a raw integer.
        wheel_raw = [MobileManipulator.degps_to_raw(deg) for deg in wheel_degps]

        return {"left_wheel": wheel_raw[0], "back_wheel": wheel_raw[1], "right_wheel": wheel_raw[2]}

    def wheel_raw_to_body(
        self, wheel_raw: dict, wheel_radius: float = 0.05, base_radius: float = 0.125
    ) -> tuple:
        """
        Convert wheel raw command feedback back into body-frame velocities.

        Parameters:
          wheel_raw   : Dictionary with raw wheel commands (keys: "left_wheel", "back_wheel", "right_wheel").
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the robot center to each wheel (meters).

        Returns:
          A tuple (x_cmd, y_cmd, theta_cmd) where:
             x_cmd      : Linear velocity in x (m/s).
             y_cmd      : Linear velocity in y (m/s).
             theta_cmd  : Rotational velocity in deg/s.
        """
        # Extract the raw values in order.
        raw_list = [
            int(wheel_raw.get("left_wheel", 0)),
            int(wheel_raw.get("back_wheel", 0)),
            int(wheel_raw.get("right_wheel", 0)),
        ]

        # Convert each raw command back to an angular speed in deg/s.
        wheel_degps = np.array([MobileManipulator.raw_to_degps(r) for r in raw_list])
        # Convert from deg/s to rad/s.
        wheel_radps = wheel_degps * (np.pi / 180.0)
        # Compute each wheel’s linear speed (m/s) from its angular speed.
        wheel_linear_speeds = wheel_radps * wheel_radius

        # Define the wheel mounting angles (defined from y axis cw)
        angles = np.radians(np.array([300, 180, 60]))
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Solve the inverse kinematics: body_velocity = M⁻¹ · wheel_linear_speeds.
        m_inv = np.linalg.inv(m)
        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x_cmd, y_cmd, theta_rad = velocity_vector
        theta_cmd = theta_rad * (180.0 / np.pi)
        return (x_cmd, y_cmd, theta_cmd)

    def teleop_step(self, record_data: bool = False) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        
        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]  # e.g. 0.1, 0.25, or 0.4
        theta_speed = speed_setting["theta"]  # e.g. 30, 60, or 90

        # Prepare to assign the position of the leader to the follower
        arm_positions = []
        # for name in self.leader_arms:

        if not isinstance(self.robot_config, DummyRobotConfig):
            pos = self.motors_bus.read("Present_Position")
            pos_tensor = torch.from_numpy(pos).float()
            arm_positions.extend(pos_tensor.tolist())
        else:
            arm_positions.extend([0.0] * len(self.robot_config.arm_motor_ids))

        y_cmd = 0.0  # m/s forward/backward
        x_cmd = 0.0  # m/s lateral
        theta_cmd = 0.0  # deg/s rotation
        if self.pressed_keys["forward"]:
            y_cmd -= xy_speed
        if self.pressed_keys["backward"]:
            y_cmd += xy_speed
        if self.pressed_keys["left"]:
            x_cmd -= xy_speed
        if self.pressed_keys["right"]:
            x_cmd += xy_speed
        if self.pressed_keys["rotate_left"]:
            theta_cmd += theta_speed
        if self.pressed_keys["rotate_right"]:
            theta_cmd -= theta_speed

        wheel_commands = self.body_to_wheel_raw(x_cmd, y_cmd, theta_cmd)

        message = {
            "raw_velocity": wheel_commands, 
            "arm_positions": arm_positions,
        }

        if self.clicked_point is not None:
            message["click"] = {
                "point": self.clicked_point,
                "id": self.clicked_id
            }
        
        self.send_command(message)

        if not record_data:
            return

        # obs_dict = self.capture_observation()

        arm_state_tensor = torch.tensor(arm_positions, dtype=torch.float32)

        wheel_velocity_tuple = self.wheel_raw_to_body(wheel_commands)
        wheel_velocity_mm = (
            wheel_velocity_tuple[0] * 1000.0,
            wheel_velocity_tuple[1] * 1000.0,
            wheel_velocity_tuple[2],
        )
        wheel_tensor = torch.tensor(wheel_velocity_mm, dtype=torch.float32)
        action_tensor = torch.cat([arm_state_tensor, wheel_tensor])
        action_dict = {"action": action_tensor}

        # return obs_dict, action_dict

# Usage with Video Display #####################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Teleop client that connects to a relay server')
    parser.add_argument('--server-ip', type=str, default="127.0.0.1",
                      help='IP address of the relay server (default: 127.0.0.1)')
    args = parser.parse_args()

    # robot_config = LeKiwiWithKochRobotConfig()
    robot_config = DummyRobotConfig()
    client = MobileManipulator(relay_server_ip=args.server_ip, robot_config=robot_config)
    
    client.start()
    
    try:
        while True:
            client.teleop_step()
            
            # Get robot state
            images, objects, speed, arm_state, timestamp = client.get_robot_state()
            # print("-------------------")
            # print("Timestamp: ", timestamp)
            # print(f"Images: {images}")
            # print(f"Speed: {speed}")
            # print(f"Arm state: {arm_state}")

            # Calculate latency
            if timestamp:
                latency = (time.time_ns() - timestamp)/1e6

            for name, image in images.items():
                # print(f"Image received for {name}")
                if not image:
                    print(f"No image received for {name}. Skipping...")
                    continue

                # Decode JPEG images
                frame1 = cv2.imdecode(
                    np.frombuffer(image, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )

                # Resize if needed
                # frame1 = cv2.resize(frame1, (640, 480))
                
                cv2.putText(frame1, f"Latency: {latency:.1f}ms", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                if name in objects and objects[name] is not None:
                    x1, y1, x2, y2 = objects[name]
                    cv2.rectangle(frame1, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    (0, 255, 0), 2)
                    
                # cv2.putText(frame1, f"Tracking ID: {self.tracked_id}", 
                #         (int(x1), int(y1) - 10), 
                #         cv2.FONT_HERSHEY_SIMPLEX, 
                #         0.9, (0, 255, 0), 2)

                # Show frames
                cv2.imshow(name, cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR))

                # Add this line to ensure window is ready
                cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
        
                cv2.setMouseCallback(name, client.mouse_callback, {'frame': frame1})
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Control loop rate
            time.sleep(0.001)
            # time.sleep(0.066)
            
    except KeyboardInterrupt:
        print("\nShutting down teleop client...")
    finally:
        client.stop()
        cv2.destroyAllWindows()