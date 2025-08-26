import torch
import time
from pynput import keyboard
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from proto.common.configs import ControllerConfig, ValueName, ActionValue
from proto.common.utils import body_to_wheel_raw, profile, manual_profile
from proto.conf import settings
from proto.control import Controller
from proto.robots.action import Action
from proto.robots.observation import Observation
from torch.utils.data import DataLoader


class ReplayController(Controller):
    def __init__(self, cfg: ControllerConfig):
        super().__init__(cfg)

        self.dataloader = None
        self.data_index = 0
        self.dataloader_iter = None
        self.batch = None
        self.last_action = None

        config = PreTrainedConfig.from_pretrained(
            pretrained_name_or_path=cfg.policy_path
        )
        config.n_action_steps = 50
        config.n_action_steps = 1
        config.temporal_ensemble_coeff = 0.01
        
        self.policy = ACTPolicy.from_pretrained(cfg.policy_path, config=config)
        
        print(self.policy.config.input_features)
        self.policy.to(self.device)
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


    def load_dataset_episode(self, episode_number):
        # Load the dataset to replay
        dataset = LeRobotDataset(self.cfg.repo_id, episodes=[episode_number])

        # And see how many frames you have:
        print(f"Selected episodes: {dataset.episodes}")
        print(f"Number of episodes selected: {dataset.num_episodes}")
        print(f"Number of frames selected: {dataset.num_frames}")

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=0,
            batch_size=self.cfg.batch_size,
            # shuffle=True,
        )

        self.data_index = 0
        self.dataloader_iter = iter(self.dataloader)
        self.batch = next(self.dataloader_iter)

        print("Dataset loaded")

    
    @profile
    def control_step(self, observation: Observation) -> Action:
        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]  # e.g. 0.1, 0.25, or 0.4
        theta_speed = speed_setting["theta"]  # e.g. 30, 60, or 90
        # print("Observation: ", observation)
        action_data = {
            settings.ModuleName.ARM: {}, 
            settings.ModuleName.BASE: {}
        }

        # print("PolicyController.control_step")

        # {
        #     'observation.state': PolicyFeature(type=<FeatureType.STATE: 'STATE'>, shape=(6,)), 
        #     'observation.images.front': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 540, 960)), 
        #     'observation.images.wrist': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 540, 960))
        # }

        # Inference

        data_inference = [[] for _ in range(6)]
        data_action = [[] for _ in range(6)]
        data_obs_state = [[] for _ in range(6)]
        data_obs_env_state = [[] for _ in range(5)]

        if self.data_index >= self.batch['observation.state'].shape[0]:
            # batch is empty, get next batch
            self.data_index = 0
            try:
                self.batch = next(self.dataloader_iter)
            except StopIteration:
                return ActionValue.STOP
        
        # print(f"{batch[camera_key].shape=}")  # (batch_size, 4, c, h, w)
        # print(f"{batch['observation.state'].shape=}")  # (batch_size, 6)
        # print(f"{batch['observation.environment_state'].shape=}")  # (batch_size, 5)
        # print(f"{batch['action'].shape=}")  # (batch_size, 6)

        i = self.data_index
        start_loop_t = time.time_ns()

        # Read the follower state and access the frames from the cameras
        observation = {}
        observation['observation.state'] = self.batch['observation.state'][i]
        observation['observation.environment_state'] = self.batch['observation.environment_state'][i]

        # print("Observation: ", observation)
        # break
        

        # Convert to pytorch format: channel first and float32 in [0,1]
        # with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(self.device)

        observation['task'] = "Grasp a lego block."
        

        # Compute the next action with the policy
        # based on the current observation
        action = self.policy.select_action(observation)
        # Remove batch dimension
        action = action.squeeze(0)
        # Move to cpu, if not already the case
        action = action.to("cpu")
        
        print("Action: ", action)
        for j in range(6):
            data_inference[j].append(action[j].item())


        for j in range(6):
            data_action[j].append(self.batch['action'][i, j].item())

        for j in range(6):
            data_obs_state[j].append(self.batch['observation.state'][i, j].item())

        for j in range(5):
            data_obs_env_state[j].append(self.batch['observation.environment_state'][i, j].item())


        action_data = {
            settings.ModuleName.ARM: {}, 
            settings.ModuleName.BASE: {}
        }
        
        for j, name in enumerate(settings.ACTION[settings.ModuleName.ARM].keys()):
            # action_data[settings.ModuleName.ARM][name] = {ValueName.POS: self.batch['action'][i, j].item()}
            action_data[settings.ModuleName.ARM][name] = {ValueName.POS: action[j].item()}
        
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

        # dt_s = time.perf_counter() - start_time
        # print("Inference time: ", dt_s)
        # busy_wait(1 / fps - dt_s)

        manual_profile("inference latency", (time.time_ns() - start_loop_t))

        self.data_index += 1
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
                state = state.to(self.device, non_blocking=True)
                # Add extra (empty) batch dimension, required to forward the policy
                state = state.unsqueeze(0)
                policy_obs[key] = state
            # elif "image" in key:
            #     image = torch.from_numpy(lerobot_obs[key])
            #     image = image.to(torch.float32) / 255
            #     image = image.permute(2, 0, 1)
            #     image = image.to(self.device, non_blocking=True)
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
                self.policy.reset()

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
    