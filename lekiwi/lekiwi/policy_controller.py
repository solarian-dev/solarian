import torch
import math
from pynput import keyboard
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.policies import PreTrainedConfig

from proto.common.configs import ValueName
from proto.common.configs import ControllerConfig
from proto.common.utils import body_to_wheel_raw
from proto.common.utils import profile
from proto.conf import settings
from proto.control import Controller
from proto.robots.action import Action
from proto.robots.observation import Observation
from proto.conf import settings


class PolicyController(Controller):
    def __init__(self, cfg: ControllerConfig):
        super().__init__(cfg)

        self.last_action = None

        config = PreTrainedConfig.from_pretrained(
            pretrained_name_or_path=cfg.policy_path
        )
        config.n_action_steps = 50
        config.n_action_steps = 1
        config.temporal_ensemble_coeff = 0.01

        self.homing = True
        
        self.policy = ACTPolicy.from_pretrained(cfg.policy_path, config=config)
        
        print(self.policy.config.input_features)
        self.policy.to(cfg.device)
        self.policy.reset()

        self.pressed_keys = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "rotate_left": False,
            "rotate_right": False,
            "run_policy": False,
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
            "run_policy": "p",
        }

        # Define three speed levels and a current index
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},  # slow
            {"xy": 0.2, "theta": 60},  # medium
            {"xy": 0.3, "theta": 90},  # fast
        ]

        self.speed_index = 0  # Start at slow

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
        # print("Observation: ", observation)
        

        # print("PolicyController.control_step")

        # {
        #     'observation.state': PolicyFeature(type=<FeatureType.STATE: 'STATE'>, shape=(6,)), 
        #     'observation.images.front': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 540, 960)), 
        #     'observation.images.wrist': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 540, 960))
        # }

        action_data = {
            settings.ModuleName.ARM: {}, 
            settings.ModuleName.BASE: {}
        }

        if observation:
            
            if self.pressed_keys["run_policy"]:
                # self.pressed_keys["run_policy"] = False
                action_data = self.call_model(observation)
            elif self.homing:
                print("Homing")
                action_data = self.move_home(observation)
            else:
                
                # # use observation as action
                # for i, name in enumerate(settings.ACTION[settings.ModuleName.ARM].keys()):
                #     action_data[settings.ModuleName.ARM][name] = observation.data[settings.ModuleName.ARM][name]

                if settings.ModuleName.LEADER_ARM in self.modules:
                    sensor_names = settings.ACTION[settings.ModuleName.ARM].keys()
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
            

            # sensor_names = settings.ACTION[ModuleName.ARM].keys()
            # action_data[ModuleName.ARM] = self.modules[ModuleName.LEADER_ARM].read_sensors(ValueName.POS, sensor_names, normalize=True)
            # print("Action data: ", action_data)
        
        self.last_action = Action(data=action_data)
        return self.last_action

    @profile
    def call_model(self, observation):

        # print("Observation: ", observation)
        lerobot_obs = observation.to_lerobot_format()
        # print("LeroBot obs: ", lerobot_obs)
        policy_obs = {}

        # Prepare observation for the policy running in Pytorch
        for key in lerobot_obs.keys():
            if "state" in key:
                state = torch.from_numpy(lerobot_obs[key])
                # Convert to float32
                state = state.to(torch.float32)
                # Send data tensors from CPU to GPU
                state = state.to(self.cfg.device, non_blocking=True)
                # Add extra (empty) batch dimension, required to forward the policy
                state = state.unsqueeze(0)
                policy_obs[key] = state
            # elif "image" in key:
            #     image = torch.from_numpy(lerobot_obs[key])
            #     image = image.to(torch.float32) / 255
            #     image = image.permute(2, 0, 1)
            #     image = image.to(self.cfg.device, non_blocking=True)
            #     image = image.unsqueeze(0)
            #     policy_obs[key] = image

        # print("Policy obs: ", policy_obs)
        
        action_data = {
            settings.ModuleName.ARM: {}, 
            settings.ModuleName.BASE: {}
        }
        
        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            action = self.policy.select_action(policy_obs)
            # print("Predicted action: ", action)

            numpy_action = action.squeeze(0).to("cpu").numpy().tolist()
            # print_list("Obs: ", lerobot_obs["observation.state"].tolist())
            print_list("Act: ", numpy_action)
            for i, name in enumerate(settings.ACTION[settings.ModuleName.ARM].keys()):
                action_data[settings.ModuleName.ARM][name] = {ValueName.POS: numpy_action[i]}
        # print("Action data: ", action_data)
        return action_data


    def move_home(self, observation):
        action_data = {
            settings.ModuleName.ARM: {}, 
            settings.ModuleName.BASE: {}
        }
        
        self.homing = False
        if settings.ModuleName.LEADER_ARM in self.modules:
            sensor_names = settings.ACTION[settings.ModuleName.ARM].keys()

            leader_arm_data = self.modules[settings.ModuleName.LEADER_ARM].read_sensors(ValueName.POS, sensor_names, normalize=True)

            MAX_POS_STEP = 10

            # # First move shoulder pan
            # sensor_names = [settings.SensorName.SHOULDER_LIFT]#, settings.SensorName.SHOULDER_LIFT]
            # first_stage = False
            # for name in sensor_names:
            #     action_data[settings.ModuleName.ARM][name] = {}
            #     if abs(leader_arm_data[name][ValueName.POS] - observation.data[settings.ModuleName.ARM][name][ValueName.POS]) > MAX_POS_STEP:
            #         action_data[settings.ModuleName.ARM][name][ValueName.POS] = observation.data[settings.ModuleName.ARM][name][ValueName.POS] + MAX_POS_STEP*math.copysign(1, leader_arm_data[name][ValueName.POS] - observation.data[settings.ModuleName.ARM][name][ValueName.POS])
            #         self.homing = True
            #         first_stage = True
            #     else:
            #         action_data[settings.ModuleName.ARM][name][ValueName.POS] = leader_arm_data[name][ValueName.POS]
                    

            # if first_stage:
            #     return action_data


            # Then move the rest of the joints
            sensor_names = settings.ACTION[settings.ModuleName.ARM].keys()
            for i, name in enumerate(sensor_names):
                action_data[settings.ModuleName.ARM][name] = {}

                if abs(leader_arm_data[name][ValueName.POS] - observation.data[settings.ModuleName.ARM][name][ValueName.POS]) > MAX_POS_STEP:
                    action_data[settings.ModuleName.ARM][name][ValueName.POS] = observation.data[settings.ModuleName.ARM][name][ValueName.POS] + MAX_POS_STEP*math.copysign(1, leader_arm_data[name][ValueName.POS] - observation.data[settings.ModuleName.ARM][name][ValueName.POS])
                    self.homing = True
                else:
                    action_data[settings.ModuleName.ARM][name][ValueName.POS] = leader_arm_data[name][ValueName.POS]
                    
        
        return action_data

    def on_press(self, key):
        try:
            # Handle recording-specific keys
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                self.events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                self.events["rerecord_episode"] = True
                self.events["exit_ea150rly"] = True
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
            elif key.char == self.teleop_keys["run_policy"]:
                print("Run policy key pressed")
                self.pressed_keys["run_policy"] = not self.pressed_keys["run_policy"]
                if self.pressed_keys["run_policy"]:
                    self.policy.reset()
                else:
                    self.homing = True

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
                # elif key.char == self.teleop_keys["run_policy"]:
                #     self.pressed_keys["run_policy"] = False
        except AttributeError:
            pass


def print_list(name, l):
    # print a list of 6 floats with 3 decimal places in the same line
    print(name, end="")
    for i in range(6):
        print("{:.3f}".format(l[i]), end=" ")
    print("]")
    