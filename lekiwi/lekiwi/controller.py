from pynput import keyboard
import cv2
import random
import string

from proto.control import Controller
from proto.robots.action import Action
from proto.robots.observation import Observation
from proto.common.utils import body_to_wheel_raw

from proto.conf import settings

from proto.common.configs import ControllerConfig, ValueName
from proto.common.utils import profile


class LeKiwiController(Controller):
    def __init__(self, cfg: ControllerConfig):
        super().__init__(cfg)

        self.last_action = None

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

        self.clicked_point = None
        self.clicked_id = None
        self.start_pos = False
        
        # Initialize OpenCV windows once
        self.windows_initialized = False
        self.last_display_time = 0
        self.display_interval = 1/45  # Limit to 30 FPS

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )

        self.listener.start()

    @profile
    def display(self):
        

        # if self.last_action:
        #     print("Last action: ", self.last_action)

        if self.last_observation is None:
            return

        # print("Last observation: ", self.last_observation)

        images = {}
        # add all images in observation
        for module_name, module_data in self.last_observation.data.items():
            for sensor_name, sensor_data in module_data.items():
                if sensor_data.get(ValueName.IMAGE) is not None:
                    images[sensor_name] = sensor_data

        # Initialize windows once if not already done
        if not self.windows_initialized:
            for name in images.keys():
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(name, self.mouse_callback)
            self.windows_initialized = True

        COLOR = (255, 0, 0)

        for name, image in images.items():
            img = image[ValueName.IMAGE]

            # print("Image: ", image)
            
            # Only modify the image if bounding box exists
            if image.get(ValueName.BOUNDING_BOX, None) is not None:
                # Create a copy only when drawing on the image
                display_img = img.copy()
                print("bounding box", image[ValueName.BOUNDING_BOX])
                x1, y1, x2, y2, depth = image[ValueName.BOUNDING_BOX]
                cv2.rectangle(display_img,
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            COLOR, 2)
                cv2.putText(display_img, f"Depth: {depth:.3f}", 
                            (int(x1), int(y1) + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0, 255, 0), 2)
            else:
                display_img = img
                
            # Convert color only once before display
            display_img_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
            
            # Show frames
            cv2.imshow(name, display_img_bgr)
            
            # No need to call getWindowProperty every time since windows are created once
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False

    @profile
    def control_step(self, observation: Observation) -> Action:
        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]  # e.g. 0.1, 0.25, or 0.4
        theta_speed = speed_setting["theta"]  # e.g. 30, 60, or 90

        action_data = {
            settings.ModuleName.ARM: {}, 
            settings.ModuleName.BASE: {}
        }

        sensor_names = settings.ACTION[settings.ModuleName.ARM].keys()

        if settings.ModuleName.LEADER_ARM in self.modules:
            action_data[settings.ModuleName.ARM] = self.modules[settings.ModuleName.LEADER_ARM].read_sensors(ValueName.POS, sensor_names, normalize=True)
        
        y_cmd = 0.0  # m/s forward/backward
        x_cmd = 0.0  # m/s lateral
        theta_cmd = 0.0  # deg/s rotation
        if self.pressed_keys["forward"]:
            x_cmd += xy_speed
        if self.pressed_keys["backward"]:
            x_cmd -= xy_speed
        if self.pressed_keys["left"]:
            y_cmd += xy_speed
        if self.pressed_keys["right"]:
            y_cmd -= xy_speed
        if self.pressed_keys["rotate_left"]:
            theta_cmd += theta_speed
        if self.pressed_keys["rotate_right"]:
            theta_cmd -= theta_speed

        wheel_actions = body_to_wheel_raw(x_cmd, y_cmd, theta_cmd)
        
        # TODO: this is not the best way to do it
        for actuator, speed in wheel_actions.items():
            action_data[settings.ModuleName.BASE][actuator] = {ValueName.SPEED: speed}
    

        # if self.clicked_point is not None:
        #     message["click"] = {
        #         "point": self.clicked_point,
        #         "id": self.clicked_id
        #     }
        
        self.last_action = Action(data=action_data)
        return self.last_action

    # Mouse callback function
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at (x={x}, y={y})")
            self.clicked_point = (x, y)
            # generate random alphanumeric id
            self.clicked_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    def on_press(self, key):
        try:
            # Handle recording-specific keys
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                self.events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                self.events["rerecord_episode"] = True
                self.events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                self.events["stop_recording"] = True
                self.events["exit_early"] = True
                
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

