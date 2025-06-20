from pydantic import BaseModel
from typing import Dict, Any
from proto.control.ai.skills import Skill


class LowLevelAction:
    # Represents an action to be executed by the robot
    pass


class HighLevelAction(BaseModel):
    # Represents an action to be executed by system 1
    
    action_id: str
    skill: Skill
    params: Dict[str, Any]


class StopAction(HighLevelAction):
    pass