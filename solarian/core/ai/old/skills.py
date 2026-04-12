from abc import ABC, abstractmethod
from typing import Dict, Any
from pydantic import BaseModel
from solarian.control.ai.actions import LowLevelAction
from solarian.core import Observation
from solarian.control.ai.policy import Policy


class SkillConfig(BaseModel):
    pass


class Skill(ABC):
    def __init__(self, cfg: SkillConfig):
        self.cfg = cfg
        
    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> LowLevelAction:
        pass


class PhysicalSkill(Skill):
    def __init__(self, cfg: SkillConfig):
        super().__init__(cfg)
        self.policy: Policy = self.build_policy(cfg)

    @abstractmethod
    def build_policy(self, cfg: SkillConfig) -> Policy:
        ...

    @abstractmethod
    def build_policy_observation(self, observation: Observation) -> Any:
        ...
        

    def execute(self, params: Dict[str, Any]) -> LowLevelAction:
        policy_observation = self.build_policy_observation(params["observation"])
        return self.policy.select_action(policy_observation)


class PickObjectSkill(PhysicalSkill):
    def __init__(self, cfg: SkillConfig):
        super().__init__(cfg)
        

    def build_policy(self, cfg: SkillConfig) -> Policy:
        pass

    def build_policy_observation(self, observation: Observation) -> Any:
        pass