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

from .settings import ModuleName, SensorName, JointName

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
        """Worker thread that processes actions from queue"""

        if self.last_action is not None:
            observation, reward, terminated, truncated, info = self.env.step(self.last_action)
            self.observation_queue.put(observation)


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
            ModuleName.ARM: {},
            ModuleName.BASE: {}
        }

        # print("Getting observation")
        observation = self.observation_queue.get()
        # print("Observation: ", observation)
        
        # Get camera images
        data[ModuleName.BASE][SensorName.FRONT] = {ValueName.IMAGE: observation["pixels"]["angle"]}
        data[ModuleName.ARM][SensorName.WRIST] = {ValueName.IMAGE: observation["pixels"]["wrist"]}
            
        # Get actuator readings
            
        # Get arm positions
        real_positions = self.sim2real(observation["agent_pos"])

        data[ModuleName.ARM].update({joint_name: {ValueName.POS: pos} for joint_name, pos in zip(JointName, real_positions)})

        # Get base speeds
        # data[ModuleName.BASE].update({joint_name: {ValueName.SPEED: speed} for joint_name, speed in zip(JointName, observation["agent_vel"])})
            
        return Observation(data=data)


    @profile
    def process_action(self, action: Action):
        """Process received action"""
        
        action_list = [value[ValueName.POS] for value in action.data[ModuleName.ARM].values()]
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
