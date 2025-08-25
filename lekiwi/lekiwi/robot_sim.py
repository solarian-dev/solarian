from pydantic import BaseModel
import numpy as np

import imageio
import os
import gymnasium as gym
import numpy as np
import gym_so100
import math
from threading import Lock


from proto.common.configs import ValueName
from proto.robots.robot import Robot
from proto.robots.observation import Observation
from proto.robots.action import Action

from proto.common.configs import ProjectConfig
from proto.common.utils import profile, manual_profile

from proto.conf import settings

from queue import Queue


# Must define robot
class LeKiwiRobotSim(Robot):
    def __init__(self, cfg: ProjectConfig):
        super().__init__(cfg)
        
        self.last_action = None

        print("Creating environment on main thread")
        self.env = gym.make("gym_so100/SO100PickCube-v0", obs_type="pixels_agent_pos")
        print("Environment created")
        self.observation, self.info = self.env.reset()
        print("Environment reset")

        # Queue for thread-safe communication
        self.observation_queue = Queue()

        print("LekiwiRobotSim initialized")

    # def process_leader_actions_fn(self, action):
    #     return real_positions_to_sim(action, **self.calib_kwgs)



    AXIS_DIRECTION = np.array([-1, -1, 1, 1, -1, 1])
    OFFSETS = np.array([0, 0, 0, 0, 90, 0])

    def real2sim(self, action: np.array):
        # Real actions come in degrees
        # Sim actions are returned in radians

        axis_directions = self.AXIS_DIRECTION
        offsets = self.OFFSETS

        # multiply action by axis_directions
        action = action * axis_directions + offsets
        # transform degrees to radians
        action = action * np.pi / 180
        return action
        
    def sim2real(self, action: np.array):
        axis_directions = self.AXIS_DIRECTION
        offsets = self.OFFSETS
        action = action * 180 / np.pi
        action = action * axis_directions - offsets
        return action


    def update(self):
        """Worker thread that processes actions from queue needs to run in main thread"""

        if self.last_action is not None:
            observation, reward, terminated, truncated, info = self.env.step(self.last_action)
            self.observation_queue.put(observation)
        else:
            self.observation_queue.put(None)


    def setup_actuators(self):
        pass

    def setup_sensors(self):
        pass

    def create_components(self):
        pass
    
    def get_module_observation(self, module_name: str) -> Observation:
        pass

    @profile
    def get_observation(self) -> Observation:
        """Create an observation with current robot state.
        For LeKiwiRobot:
        - Camera images from all cameras
        - Arm actuator positions
        - Base actuator speeds
        
        Returns:
            Observation: Current robot state
        """

        

        data = {
            settings.ModuleName.ARM: {},
            settings.ModuleName.BASE: {}
        }

        print("Getting observation from sim")
        observation = self.observation_queue.get()
        # print("Observation: ", observation)
        
        if observation is None:
            return None
        
        # Get camera images
        data[settings.ModuleName.BASE][settings.SensorName.FRONT] = {ValueName.IMAGE: observation["pixels"]["angle"]}
        data[settings.ModuleName.ARM][settings.SensorName.WRIST] = {ValueName.IMAGE: observation["pixels"]["wrist"]}
            
        # Get actuator readings
            
        # Get arm positions
        real_positions = self.sim2real(observation["agent_pos"])

        data[settings.ModuleName.ARM].update({joint_name: {ValueName.POS: pos} for joint_name, pos in zip(settings.JointName, real_positions)})

        # Get base speeds
        # data[settings.ModuleName.BASE].update({joint_name: {ValueName.SPEED: speed} for joint_name, speed in zip(settings.JointName, observation["agent_vel"])})
            
        return Observation(data=data)


    @profile
    def process_action(self, action: Action):# Visualization
            # if results and len(results[0].boxes) > 0:
            #     # Only show the tracked object
            #     boxes = results[0].boxes
            #     masks = results[0].masks
                
                
            #     # Find our tracked object
            #     for i, box in enumerate(boxes):
            #         if hasattr(box, 'id') and box.id.item() == self.tracked_id:
            #             tracked_box = box

            #             if self.show_mask:
            #                 tracked_mask = masks[i].data[0].cpu().numpy()
            #                 # Resize mask to match frame dimensions
            #                 tracked_mask = cv2.resize(
            #                     tracked_mask,
            #                     (frame.shape[1], frame.shape[0]),
            #                     interpolation=cv2.INTER_NEAREST
            #                 )
            #                 print("mask shape", tracked_mask.shape)
            #             break
                
            #     if tracked_box is not None:
            #         # Create a copy of the frame for drawing
            #         annotated_frame = frame.copy()
            #         if self.show_mask:
            #             # Create colored overlay for the mask
            #             color_mask = np.zeros_like(annotated_frame)
            #             color_mask[tracked_mask == 1] = [0, 255, 0]  # Green color for the mask
                        
            #             # Blend the mask with the frame
            #             alpha = 0.5  # Transparency factor
            #             mask_overlay = cv2.addWeighted(annotated_frame, 1, color_mask, alpha, 0)
            #             annotated_frame = mask_overlay
                    
            #         # Draw just the tracked box
            #         x1, y1, x2, y2 = tracked_box.xyxy[0].tolist()
            #         tracked_box = [x1, y1, x2, y2]
            #         # cv2.rectangle(annotated_frame, 
            #         #             (int(x1), int(y1)), 
            #         #             (int(x2), int(y2)), 
            #         #             (0, 255, 0), 2)
                    
            #         # cv2.putText(annotated_frame, f"Tracking ID: {self.tracked_id}", 
            #         #         (int(x1), int(y1) - 10), 
            #         #         cv2.FONT_HERSHEY_SIMPLEX, 
            #         #         0.9, (0, 255, 0), 2)
            #     else:
            #         print("Lost tracked object")
            #         self.tracking_active = False
            #         self.tracked_id = None
            #         annotated_frame = frame
            # else:
            #     annotated_frame = frame


            # frame = annotated_frame
        """Process received action"""
        
        action_list = [value[ValueName.POS] for value in action.data[settings.ModuleName.ARM].values()]
        action = np.array(action_list)
        # print("Action processed")
        # print("Action: ", action_list)
        
        action_formated = ""
        for v in action_list:
            # print with 2 decimal places
            action_formated += " {:.2f}".format(v)
        # print("Action formated: ", action_formated)       
        
        self.last_action = self.real2sim(action)
        # print("Action processed: ", self.last_action)
