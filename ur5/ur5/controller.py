from pynput import keyboard
import cv2
import math


from proto.control import Controller
from proto.robots.action import Action
from proto.robots.observation import Observation
from proto.common.utils import body_to_wheel_raw

from proto.conf import settings

from proto.common.configs import ControllerConfig, ValueName
from proto.common.utils import profile, StatusBoard


class UR5Controller(Controller):
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

        # self.start_pos = False

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )

        self.listener.start()

    

    @profile
    def control_step(self, observation: Observation) -> Action:
        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]  # e.g. 0.1, 0.25, or 0.4
        theta_speed = speed_setting["theta"]  # e.g. 30, 60, or 90

        action_data = {
            
        }

        # sensor_names = settings.ACTION[settings.ModuleName.ARM].keys()

        # if settings.ModuleName.LEADER_ARM in self.modules:
        #     action_data[settings.ModuleName.ARM] = self.modules[settings.ModuleName.LEADER_ARM].read_sensors(ValueName.POS, sensor_names, normalize=True)
        
        sensor_names = [
            settings.SensorName.SHOULDER_PAN, 
            settings.SensorName.SHOULDER_LIFT, 
            settings.SensorName.ELBOW_FLEX, 
            settings.SensorName.WRIST_FLEX, 
            settings.SensorName.WRIST_ROLL,
            settings.SensorName.GRIPPER
        ]
        leader_data = self.modules[settings.ModuleName.LEADER_ARM].read_sensors(ValueName.POS, sensor_names, normalize=True)
        # StatusBoard().set("Leader Arm", leader_data)

        action_data[settings.ModuleName.GRIPPER] = {settings.SensorName.GRIPPER: leader_data[settings.SensorName.GRIPPER]}


        # StatusBoard().set("Leader Arm", action_data)

        DELTA = 0.05

        # if self.pressed_keys["forward"]:
        #     # elbow_flex = observation.data[settings.ModuleName.ARM][settings.ActuatorName.ELBOW_FLEX][ValueName.POS]
        #     # # add 2 degrees in radians
        #     # elbow_flex += math.radians(2)
        #     # action_data[settings.ModuleName.ARM][settings.ActuatorName.ELBOW_FLEX] = {ValueName.POS: elbow_flex}
            
        #     value = observation.data[settings.ModuleName.ARM][settings.ActuatorName.X][ValueName.POS]
        #     # add 0.1 meters to X
        #     value += DELTA
        #     action_data[settings.ModuleName.ARM][settings.ActuatorName.X] = {ValueName.POS: value}


        # elif self.pressed_keys["backward"]:
        #     # elbow_flex = observation.data[settings.ModuleName.ARM][settings.ActuatorName.ELBOW_FLEX][ValueName.POS]
        #     # # add 2 degrees in radians
        #     # elbow_flex -= math.radians(2)
        #     # action_data[settings.ModuleName.ARM][settings.ActuatorName.ELBOW_FLEX] = {ValueName.POS: elbow_flex}

        #     value = observation.data[settings.ModuleName.ARM][settings.ActuatorName.X][ValueName.POS]
        #     # add 0.1 meters
        #     value -= DELTA
        #     action_data[settings.ModuleName.ARM][settings.ActuatorName.X] = {ValueName.POS: value}

        # elif self.pressed_keys["left"]:
        #     value = observation.data[settings.ModuleName.ARM][settings.ActuatorName.Y][ValueName.POS]
        #     # add 0.1 meters to Y
        #     value +=  DELTA
        #     action_data[settings.ModuleName.ARM][settings.ActuatorName.Y] = {ValueName.POS: value}

        # elif self.pressed_keys["right"]:
        #     value = observation.data[settings.ModuleName.ARM][settings.ActuatorName.Y][ValueName.POS]
        #     # add 0.1 meters
        #     value -= DELTA
        #     action_data[settings.ModuleName.ARM][settings.ActuatorName.Y] = {ValueName.POS: value}

        # elif self.pressed_keys["rotate_left"]:
        #     value = observation.data[settings.ModuleName.ARM][settings.ActuatorName.Z][ValueName.POS]
        #     # add 0.1 meters to Z
        #     value -= DELTA
        #     action_data[settings.ModuleName.ARM][settings.ActuatorName.Z] = {ValueName.POS: value}

        # elif self.pressed_keys["rotate_right"]:
        #     value = observation.data[settings.ModuleName.ARM][settings.ActuatorName.Z][ValueName.POS]
        #     # add 0.1 meters
        #     value += DELTA
        #     action_data[settings.ModuleName.ARM][settings.ActuatorName.Z] = {ValueName.POS: value}

        # else:
        #     # if no cartessian commands are sent, then send leader arm commands
        #     action_data[settings.ModuleName.ARM] = leader_data
        #     # pass
        
        
        self.last_action = Action(data=action_data)
        return self.last_action

    

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

