from typing import List, Dict
from proto.control.ai.actions import HighLevelAction, LowLevelAction, StopAction
from proto.control.ai.event import Event
from proto.robots.observation import Observation
from proto.control.ai.system2 import System2
from proto.control.ai.skills import PhysicalSkill


class System1:

    def __init__(self, system2: System2):
        self.system2 = system2
        self.active_actions: Dict[str, HighLevelAction] = {}
        self.action_queue = []
    
    def execute_action(self, action: HighLevelAction):
        self.action_queue.append(action)


    def merge_low_level_actions(self, actions: List[LowLevelAction]) -> LowLevelAction:
        # TODO: actually merge multiple actions into one, now just picking the first one
        return actions[0]


    def feel(self, observation: Observation) -> List[Event]:
        # receives observation from robot perception at FPS rate
        # detect (feel) events from observations, and return as events
        # Examples: something is moving, something is touching, etc
        # feel like reflecting? feel like learning?
        # heard something?
        # finished an action? change in action status?
        
        # TODO: implement
        # this should detect events from the observation
        # should run at FPS rate, that mean it runs fast, think of it as a fast reaction to the environment, like a reflex
        # the system 2 by itself will do its own thinking at a lower rate, which involves perceiving the envirionment.
        # for now just return a generic event
        return [Event(description="just pay attention to the environment")]


    def react(self, observation: Observation) -> LowLevelAction:
        # receives observation from robot perception at FPS rate
        
        events = self.feel(observation)
        if events:
            # if meaningful events are perceived, get attention of system 2
            self.system2.attend(events, observation)
            
        # get new actions from system2
        new_actions = self.system2.get_actions()
        self.active_actions.update(new_actions)

        # execute stop actions first
        for action_id, active_action in self.active_actions.items():
            if isinstance(active_action, StopAction):
                action_to_stop = self.active_actions[active_action.params["action_id"]]
                action_to_stop.skill.stop()
                # delete action to stop
                del self.active_actions[active_action.params["action_id"]]
                # delete stop action
                del self.active_actions[action_id]
                
        low_level_actions = []

        # execute active actions
        for action_id, active_action in self.active_actions.items():
            if isinstance(active_action.skill, PhysicalSkill):
                # execute physical skill
                params = {
                    "observation": observation,
                }
                low_level_action = active_action.skill.execute(params)
                low_level_actions.append(low_level_action)

            else:
                # TODO: handle non-physical skills
                pass

        # merge low level actions
        merged_action = self.merge_low_level_actions(low_level_actions)

        return merged_action