from pynput import keyboard
import cv2
import random
import string

from proto.control import Controller
from proto.robots.action import Action
from proto.robots.observation import Observation
from proto.common.utils import body_to_wheel_raw
from proto.common.configs import ValueName

from .settings import ActuatorName, ModuleName, SensorName

from proto.common.configs import ProjectConfig
from proto.common.utils import profile

from proto.conf import settings


from lerobot.common.policies.act.modeling_act import ACTPolicy


class PolicyController(Controller):
    def __init__(self, cfg: ProjectConfig):
        super().__init__(cfg)

        # self.actuators = self._create_actuator_bus()

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


        # Create a directory to store the video of the evaluation
        output_directory = Path("outputs/eval/example_pusht_diffusion")
        output_directory.mkdir(parents=True, exist_ok=True)

        # Select your device
        device = "mps"

        # Provide the [hugging face repo id](https://huggingface.co/lerobot/diffusion_pusht):
        pretrained_policy_path = "output/act_lekiwi_pick_and_place_dice2"
        # OR a path to a local outputs/train folder.
        # pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")

        self.policy = ACTPolicy.from_pretrained(pretrained_policy_path)

    def setup_actuators(self):
        # Disable torque for all actuators
        self.modules[ModuleName.ARM].disable_actuators_torque()
        
        # Calibrate the leader arm
        self.modules[ModuleName.ARM].calibrate()

        # Enable torque for the gripper and set position
        print(f"Setting gripper open angle...")
        self.modules[ModuleName.ARM].enable_actuators_torque([ActuatorName.GRIPPER])
        self.modules[ModuleName.ARM].write_actuators_pos({ActuatorName.GRIPPER: self.cfg.gripper_open_degree})

    def setup_sensors(self):
        pass

    @profile
    def display_state(self):
        if self.last_observation is None:
            return

        # Initialize windows once if not already done
        if not self.windows_initialized:
            for name in [SensorName.FRONT, SensorName.WRIST]:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(name, self.mouse_callback)
            self.windows_initialized = True

        images = {}
        # add all images in observation
        for module_name, module_data in self.last_observation.data.items():
            for sensor_name, sensor_data in module_data.items():
                if sensor_data.get(ValueName.IMAGE) is not None:
                    images[sensor_name] = sensor_data

        COLOR = (255, 0, 0)

        for name, image in images.items():
            img = image[ValueName.IMAGE]
            
            # Only modify the image if bounding box exists
            if image.get(ValueName.BOUNDING_BOX, None) is not None:
                # Create a copy only when drawing on the image
                display_img = img.copy()
                x1, y1, x2, y2 = image[ValueName.BOUNDING_BOX]
                cv2.rectangle(display_img,
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            COLOR, 2)
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

        # self.display_state()
        
        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]  # e.g. 0.1, 0.25, or 0.4
        theta_speed = speed_setting["theta"]  # e.g. 30, 60, or 90

        action_data = {
            ModuleName.ARM: {}, 
            ModuleName.BASE: {}
        }

        sensor_names = settings.ACTION[ModuleName.ARM].keys()
        action_data[ModuleName.ARM] = self.modules[ModuleName.ARM].read_sensors(ValueName.POS, sensor_names)
        
        y_cmd = 0.0  # m/s forward/backward
        x_cmd = 0.0  # m/s lateral
        theta_cmd = 0.0  # deg/s rotation
        if self.pressed_keys["forward"]:
            y_cmd += xy_speed
        if self.pressed_keys["backward"]:
            y_cmd -= xy_speed
        if self.pressed_keys["left"]:
            x_cmd += xy_speed
        if self.pressed_keys["right"]:
            x_cmd -= xy_speed
        if self.pressed_keys["rotate_left"]:
            theta_cmd += theta_speed
        if self.pressed_keys["rotate_right"]:
            theta_cmd -= theta_speed

        wheel_actions = body_to_wheel_raw(x_cmd, y_cmd, theta_cmd)
        
        # TODO: this is not the best way to do it
        for actuator, speed in wheel_actions.items():
            action_data[ModuleName.BASE][actuator] = {ValueName.SPEED: speed}
    

        # if self.clicked_point is not None:
        #     message["click"] = {
        #         "point": self.clicked_point,
        #         "id": self.clicked_id
        #     }
        
        
        # # RG review, probably redundance here
        # obs_dict = self.capture_observation()

        # arm_state_tensor = torch.tensor(arm_positions, dtype=torch.float32)

        # if self.cfg.base_motor_ids:
        #     wheel_velocity_tuple = wheel_raw_to_body(wheel_actions)
        #     wheel_velocity_mm = (
        #         wheel_velocity_tuple[0] * 1000.0,
        #         wheel_velocity_tuple[1] * 1000.0,
        #         wheel_velocity_tuple[2],
        #     )
        #     wheel_tensor = torch.tensor(wheel_velocity_mm, dtype=torch.float32)
        #     action_tensor = torch.cat([arm_state_tensor, wheel_tensor])
        # else:
        #     action_tensor = arm_state_tensor
        
        action = Action(data=action_data)
        return action

    # Mouse callback function
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at (x={x}, y={y})")
            self.clicked_point = (x, y)
            # generate random alphanumeric id
            self.clicked_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

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

