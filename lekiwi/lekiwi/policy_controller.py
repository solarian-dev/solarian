import random
import string

import cv2
from lerobot.common.policies.act.modeling_act import ACTPolicy
from pynput import keyboard
import torch

from proto.common.configs import ValueName
from proto.common.configs import ControllerConfig
from proto.common.utils import body_to_wheel_raw
from proto.common.utils import profile
from proto.conf import settings
from proto.control import Controller
from proto.robots.action import Action
from proto.robots.observation import Observation

from .settings import ActuatorName, ModuleName, SensorName


class PolicyController(Controller):
    def __init__(self, cfg: ControllerConfig):
        super().__init__(cfg)

        self.last_action = None

        
        
        # Initialize OpenCV windows once
        self.windows_initialized = False
        self.last_display_time = 0
        self.display_interval = 1/45  # Limit to 30 FPS


        self.policy = ACTPolicy.from_pretrained(cfg.policy_path)
        self.policy.config.n_action_steps = 100
        print(self.policy.config.input_features)
        self.policy.to(cfg.device)

        self.policy.reset()

        

    @profile
    def display_state(self):
        

        # if self.last_action:
        #     print("Last action: ", self.last_action)

        if self.last_observation is None:
            return

        # print("Last observation: ", self.last_observation)

        # Initialize windows once if not already done
        if not self.windows_initialized:
            for name in [SensorName.FRONT, SensorName.WRIST]:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
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
            try:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return False
            except KeyboardInterrupt:
                return False

    @profile
    def control_step(self, observation: Observation) -> Action:

        
        
        action_data = {
            ModuleName.ARM: {}, 
            ModuleName.BASE: {}
        }

        # print("PolicyController.control_step")

        # {
        #     'observation.state': PolicyFeature(type=<FeatureType.STATE: 'STATE'>, shape=(6,)), 
        #     'observation.images.front': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 540, 960)), 
        #     'observation.images.wrist': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 540, 960))
        # }

        if observation:

            action_data = self.call_model(observation)

            # sensor_names = settings.ACTION[ModuleName.ARM].keys()
            # action_data[ModuleName.ARM] = self.modules[ModuleName.LEADER_ARM].read_sensors(ValueName.POS, sensor_names, normalize=True)
            # print("Action data: ", action_data)
        
        self.last_action = Action(data=action_data)
        return self.last_action

    @profile
    def call_model(self, observation):

        lerobot_obs = observation.to_lerobot_format()
        policy_obs = {}

        # Prepare observation for the policy running in Pytorch
        state = torch.from_numpy(lerobot_obs["observation.state"])
        # Convert to float32
        state = state.to(torch.float32)
        # Send data tensors from CPU to GPU
        state = state.to(self.cfg.device, non_blocking=True)
        # Add extra (empty) batch dimension, required to forward the policy
        state = state.unsqueeze(0)
        policy_obs["observation.state"] = state


        # Prepare images
        for key in lerobot_obs.keys():
            if "image" in key:
                image = torch.from_numpy(lerobot_obs[key])
                image = image.to(torch.float32) / 255
                image = image.permute(2, 0, 1)
                image = image.to(self.cfg.device, non_blocking=True)
                image = image.unsqueeze(0)
                policy_obs[key] = image
        
        action_data = {
            ModuleName.ARM: {}, 
            ModuleName.BASE: {}
        }
        
        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            action = self.policy.select_action(policy_obs)
            # print("Predicted action: ", action)

            numpy_action = action.squeeze(0).to("cpu").numpy().tolist()
            print_list("Obs: ", lerobot_obs["observation.state"].tolist())
            print_list("Act: ", numpy_action)
            for i, name in enumerate(settings.ACTION[ModuleName.ARM].keys()):
                action_data[ModuleName.ARM][name] = {ValueName.POS: numpy_action[i]}
        # print("Action data: ", action_data)
        return action_data



def print_list(name, l):
    # print a list of 6 floats with 3 decimal places in the same line
    print(name, end="")
    for i in range(6):
        print("{:.3f}".format(l[i]), end=" ")
    print("]")
    