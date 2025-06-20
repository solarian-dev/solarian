import time
import threading
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

from pydantic import BaseModel

from proto.control import Controller
from proto.control.ai.actions import HighLevelAction
from proto.control.ai.skills import Skill
from proto.control.ai.event import Event
from proto.robots.observation import Observation
from proto.control.ai.skills import PickObjectSkill

# Thoughts per second
# Represents the frequency at which system 2 runs
TPS = 1

class System2:

    def __init__(self, system2_model, system2_prompt, skills: List[Skill]):
        self.model = system2_model
        self.system2_prompt = system2_prompt
        self.skills = skills
        self.attention_queue = []
        self.action_queue: List[HighLevelAction] = []


    def idle(self):
        time.sleep(1/TPS)


    def call_model(self, events: List[Event], observation: Observation) -> str:
        # call model with events and observation
        # return a thought

        # TODO: implement
        # prepare prompt for model, depending of VLM used
        # thought = self.model(self.system2_prompt, events, observation)

        thought = 'Action: pick_object: {"object_description": "pink lego block"}'
        return thought

    def get_actions(self) -> List[HighLevelAction]:
        return self.action_queue


    def queue_action(self, action: HighLevelAction):
        for existing_action in self.action_queue:
            if type(existing_action.skill) == type(action.skill):
                return
        self.action_queue.append(action)


    def get_action_from_thought(self, thought: str) -> HighLevelAction:
        # parse a thought into a high level action
        # TODO: actually parse the thought

        return HighLevelAction(
            action_id="pick pink lego block",
            skill=PickObjectSkill(),
            params={"object_description": "pink lego block"}
        )

    def attend(self, events: List[Event], observation: Observation):        
        self.attention_queue.append((events, observation))

    def think(self):
        # Run system2 at TPS rate
        while True:
            
            # events and observations could be empty
            # in that case system2 might just reflect on its own
            # eg "I should go to sleep"
            # or "I should learn something new"
            # or "I should do something fun"
            # or "I should do something to help someone"
            # or "I should do something to help the environment"
            # or "I should recharge"
            events, observation = self.attention_queue.pop(0)
            thought = self.call_model(events, observation)
            action = self.get_action_from_thought(thought)
            self.queue_action(action)
        
            self.idle()


    def sleep(self):
        # Runs when not wakeup, could dream or just stay idle
        pass


    def dream(self):
        # Run simulations to train system 1 and system 2 policies
        pass






        
        