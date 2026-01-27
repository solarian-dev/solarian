import time
import threading
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from enum import Enum
from openai import OpenAI
from pydantic import BaseModel
import numpy as np
import base64
import json
import re


from proto.control import Controller
from proto.control.ai.actions import HighLevelAction
from proto.control.ai.skills import Skill
from proto.control.ai.event import Event
from proto.robots.observation import Observation
from proto.control.ai.skills import PickObjectSkill
from proto.conf import settings
from proto.common.configs import ValueName

# Thoughts per second
# Represents the frequency at which system 2 runs
TPS = 1

class ModelName(Enum):
    MINICPM_V_4_5 = "openbmb/MiniCPM-V-4_5"

class ModelApi(Enum):
    OPENAI = "openai"

MODEL_API_MAP = {
    ModelName.MINICPM_V_4_5: ModelApi.OPENAI,
}


from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, conlist, confloat, constr


# ---------------------------
# Utilities: image -> data URL
# ---------------------------

def guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "image/jpeg"

def image_path_to_data_url(image_path: str) -> str:
    b = Path(image_path).read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{guess_mime(image_path)};base64,{b64}"


# ---------------------------
# Pydantic models (contracts)
# ---------------------------

BBox = conlist(int, min_length=4, max_length=4)

class CameraInfo(BaseModel):
    name: str = "wrist_cam"
    image_width: int
    image_height: int

class SceneObject(BaseModel):
    id: constr(min_length=1)
    label: constr(min_length=1)
    bbox_xyxy: BBox
    confidence: confloat(ge=0.0, le=1.0)
    attributes: List[str] = Field(default_factory=list)
    notes: str = ""

class SceneRelation(BaseModel):
    subject_id: str
    relation: Literal["on_top_of", "inside", "left_of", "right_of", "in_front_of", "behind", "touching"]
    object_id: str
    confidence: confloat(ge=0.0, le=1.0)

class Uncertainty(BaseModel):
    kind: Literal["occlusion", "lighting", "out_of_view", "ambiguous"]
    detail: str

class SceneState(BaseModel):
    type: Literal["scene_state"]
    frame_id: str
    timestamp_ms: Optional[int] = Field(default_factory=lambda: int(time.time() * 1000))
    camera: CameraInfo
    objects: List[SceneObject] = Field(default_factory=list)
    relations: List[SceneRelation] = Field(default_factory=list)
    uncertainties: List[Uncertainty] = Field(default_factory=list)


class ToolCall(BaseModel):
    name: constr(min_length=1)
    args: Dict[str, Any] = Field(default_factory=dict)

class FinalBlock(BaseModel):
    summary: str = ""
    next_step_hint: str = ""

class SafetyBlock(BaseModel):
    confidence: confloat(ge=0.0, le=1.0)
    risk: Literal["low", "medium", "high"]
    reason: str

class Decision(BaseModel):
    type: Literal["decision"]
    decision_id: str
    status: Literal["tool_call", "no_action", "final", "refuse"]
    tool_call: Optional[ToolCall] = None
    final: Optional[FinalBlock] = None
    safety: SafetyBlock


# ---------------------------
# Tool registry + validation
# ---------------------------

class RobotState(BaseModel):
    tcp_pose: Dict[str, float]   # e.g. {"x":..., "y":..., "z":..., "qx":..., "qy":..., "qz":..., "qw":...}
    gripper_width_mm: float

class Pose(BaseModel):
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float

class GraspPlan(BaseModel):
    grasp_pose: Pose
    approach_pose: Pose
    width_mm: int

class ToolSpec(BaseModel):
    name: str
    description: str
    # A callable validator that can raise ValidationError or ValueError
    validator: Callable[[Dict[str, Any]], Dict[str, Any]]
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]


def validate_no_args(args: Dict[str, Any]) -> Dict[str, Any]:
    if args:
        raise ValueError("This tool takes no arguments.")
    return args

def validate_plan_grasp(args: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal schema enforcement.
    if "object_label" not in args or not isinstance(args["object_label"], str):
        raise ValueError("plan_grasp requires object_label: str")
    if "bbox_xyxy" not in args or not (isinstance(args["bbox_xyxy"], list) and len(args["bbox_xyxy"]) == 4):
        raise ValueError("plan_grasp requires bbox_xyxy: [x1,y1,x2,y2]")
    return args

def validate_execute_grasp(args: Dict[str, Any]) -> Dict[str, Any]:
    # Validate with Pydantic models for strong checks.
    plan = GraspPlan.model_validate(args)
    # Add simple safety bounds example:
    if plan.width_mm < 0 or plan.width_mm > 110:
        raise ValueError("width_mm out of bounds [0,110]")
    # Workspace checks (example bounds; replace with your robot):
    if not (0.0 <= plan.grasp_pose.z <= 1.5):
        raise ValueError("grasp_pose.z out of workspace bounds")
    return plan.model_dump()

def validate_execute_place(args: Dict[str, Any]) -> Dict[str, Any]:
    pose = Pose.model_validate(args)
    if not (0.0 <= pose.z <= 1.5):
        raise ValueError("place pose z out of bounds")
    return pose.model_dump()

# --- Tool implementations (stubs you replace with your robot control stack) ---

def tool_get_robot_state(_: Dict[str, Any]) -> Dict[str, Any]:
    # Replace with real robot state.
    state = RobotState(
        tcp_pose={"x": 0.4, "y": 0.0, "z": 0.3, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
        gripper_width_mm=80.0,
    )
    return {"ok": True, "state": state.model_dump()}

def tool_plan_grasp(args: Dict[str, Any]) -> Dict[str, Any]:
    # Replace with grasp planner (from depth, segmentation, etc.).
    # Here: dummy grasp in front of camera.
    _ = args["object_label"]
    plan = GraspPlan(
        grasp_pose=Pose(x=0.45, y=0.02, z=0.12, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        approach_pose=Pose(x=0.45, y=0.02, z=0.20, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        width_mm=55,
    )
    return {"ok": True, "plan": plan.model_dump()}

def tool_execute_grasp(args: Dict[str, Any]) -> Dict[str, Any]:
    plan = GraspPlan.model_validate(args)
    # Replace with motion execution and gripper close.
    return {"ok": True, "executed": "grasp", "plan": plan.model_dump()}

def tool_execute_place(args: Dict[str, Any]) -> Dict[str, Any]:
    pose = Pose.model_validate(args)
    # Replace with motion execution and gripper open.
    return {"ok": True, "executed": "place", "pose": pose.model_dump()}


def build_tool_registry() -> Dict[str, ToolSpec]:
    return {
        "get_robot_state": ToolSpec(
            name="get_robot_state",
            description="Get current TCP pose and gripper width.",
            validator=validate_no_args,
            fn=tool_get_robot_state,
        ),
        "plan_grasp": ToolSpec(
            name="plan_grasp",
            description="Plan a grasp for an object given its label and 2D bbox.",
            validator=validate_plan_grasp,
            fn=tool_plan_grasp,
        ),
        "execute_grasp": ToolSpec(
            name="execute_grasp",
            description="Execute a grasp using a provided grasp and approach pose and target width.",
            validator=validate_execute_grasp,
            fn=tool_execute_grasp,
        ),
        "execute_place": ToolSpec(
            name="execute_place",
            description="Execute a place at a target pose.",
            validator=validate_execute_place,
            fn=tool_execute_place,
        ),
    }


# ---------------------------
# VLM prompting (strict JSON)
# ---------------------------

PERCEPTION_SYSTEM = """You are a vision perception module for a robot. You will be given a scene seen from a camera mounted in the wrist of a robot arm, you will see the view from a parallel gripper with two orange fingers.
Return ONLY valid JSON matching this schema:
{
  "type":"scene_state",
  "frame_id": "...",
  "camera":{"name":"wrist_cam","image_width":<int>,"image_height":<int>},
  "objects":[{"id":"...","label":"...","bbox_xyxy":[x1,y1,x2,y2],"confidence":0..1,"attributes":[...],"notes":"..."}],
  "relations":[{"subject_id":"...","relation":"on_top_of|inside|left_of|right_of|in_front_of|behind|touching","object_id":"...","confidence":0..1}],
  "uncertainties":[{"kind":"occlusion|lighting|out_of_view|ambiguous","detail":"..."}]
}
Rules:
- No extra keys. No markdown. No explanation.
- If unsure, lower confidence and add an uncertainties entry.
"""

DECISION_SYSTEM = """You are a robot action selector.
Return ONLY valid JSON matching this schema:
{
  "type":"decision",
  "decision_id":"...",
  "status":"tool_call|no_action|final|refuse",
  "tool_call":{"name":"...","args":{...}} OR null,
  "final":{"summary":"...","next_step_hint":"..."} OR null,
  "safety":{"confidence":0..1,"risk":"low|medium|high","reason":"..."}
}
Rules:
- No explanation, no markdown, no <think>.
- Never invent tool results.
- Prefer a single tool_call per response.
- If perception confidence is low or action is unsafe, respond with status=no_action or refuse.
"""

def tool_list_for_prompt(tools: Dict[str, ToolSpec]) -> str:
    lines = []
    for t in tools.values():
        lines.append(f'- {t.name}: {t.description}')
    return "\n".join(lines)


# ---------------------------
# Agent implementation
# ---------------------------

@dataclass
class AgentConfig:
    model: str = "openbmb/MiniCPM-V-4_5"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "local"
    timeout_s: float = 10.0
    temperature: float = 0.0
    max_tokens_perception: int = 350
    max_tokens_decision: int = 220
    max_steps: int = 6
    min_confidence_to_act: float = 0.60


class VLMReActAgent:
    def __init__(self, cfg: AgentConfig, tools: Dict[str, ToolSpec]) -> None:
        self.cfg = cfg
        self.tools = tools
        self.attention_queue = []
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def idle(self):
        time.sleep(1/TPS)

    def _chat_one(self, system: str, user_text: str, image_data_url: str, max_tokens: int) -> str:

        user_content = [
            {"type": "text", "text": user_text}
        ]
        if image_data_url:
            user_content.append({"type": "image_url", "image_url": {"url": image_data_url}})

        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            temperature=self.cfg.temperature,
            max_tokens=max_tokens,
            timeout=self.cfg.timeout_s,
        )
        return (resp.choices[0].message.content or "").strip()

    def _parse_json(self, text: str) -> Dict[str, Any]:
        # Minimal hardening: strip common wrappers
        text = text.strip()
        # If model accidentally wraps in ```json ... ```
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()
        return json.loads(text)

    def perceive(self, image_data_url: str, command: str, width: int, height: int, frame_id: str) -> SceneState:
        user_text = (
            f"Task command: {command}\n"
            f"frame_id: {frame_id}\n"
            f"image_width: {width}\n"
            f"image_height: {height}\n"
            "Detect relevant objects for the command and their relations.\n"
            "Use labels like red_cup, red_saucer, etc.\n"
        )
        raw = self._chat_one(PERCEPTION_SYSTEM, user_text, image_data_url, self.cfg.max_tokens_perception)

        try:
            data = self._parse_json(raw)
            return SceneState.model_validate(data)
        except Exception:
            # Repair prompt once
            repair_text = (
                user_text
                + "\nINVALID OUTPUT. Return ONLY valid JSON exactly matching the schema. No markdown.\n"
            )
            raw2 = self._chat_one(PERCEPTION_SYSTEM, repair_text, image_data_url, self.cfg.max_tokens_perception)
            data2 = self._parse_json(raw2)
            return SceneState.model_validate(data2)

    def decide(self, image_data_url: str, command: str, scene: SceneState, step_idx: int, obs: List[Dict[str, Any]]) -> Decision:
        tools_txt = tool_list_for_prompt(self.tools)
        user_text = (
            f"Command: {command}\n"
            f"Step: {step_idx}\n\n"
            f"Available tools:\n{tools_txt}\n\n"
            f"SceneState JSON:\n{scene.model_dump_json()}\n\n"
            f"Observations JSON (tool results so far):\n{json.dumps(obs)}\n\n"
            "Choose next action. Prefer one tool_call.\n"
            "If you need a grasp, call plan_grasp first using the best bbox of the target object.\n"
        )
        raw = self._chat_one(DECISION_SYSTEM, user_text, image_data_url, self.cfg.max_tokens_decision)

        try:
            data = self._parse_json(raw)
            return Decision.model_validate(data)
        except Exception:
            repair_text = user_text + "\nINVALID OUTPUT. Return ONLY valid JSON matching the decision schema.\n"
            raw2 = self._chat_one(DECISION_SYSTEM, repair_text, None, self.cfg.max_tokens_decision)
            data2 = self._parse_json(raw2)
            return Decision.model_validate(data2)

    def run(self, command: str, width: int, height: int) -> Tuple[str, List[Dict[str, Any]]]:
        
        # scene = self.perceive(image_data_url, command, width, height, frame_id)
        observations: List[Dict[str, Any]] = []

        for step in range(1, self.cfg.max_steps + 1):
            events, observation = self.attention_queue.pop(0)
            # clear attention queue, for now we only use the last event and observation
            self.attention_queue = []
            image_data_url = self._image_to_data_url(observation.data[settings.ModuleName.GRIPPER][settings.SensorName.WRIST_CAM][ValueName.IMAGE])


            frame_id = str(int(time.time() * 1000))
            scene = self.perceive(image_data_url, command, width, height, frame_id)
            decision = self.decide(image_data_url, command, scene, step, observations)

            # Safety gate
            if decision.safety.confidence < self.cfg.min_confidence_to_act or decision.safety.risk == "high":
                return (
                    json.dumps(
                        {
                            "status": "no_action",
                            "reason": f"safety_gate: {decision.safety.reason}",
                            "safety": decision.safety.model_dump(),
                            "scene": scene.model_dump(),
                        },
                        indent=2,
                    ),
                    observations,
                )

            if decision.status in ("final", "refuse", "no_action"):
                return (
                    json.dumps(
                        {
                            "status": decision.status,
                            "final": (decision.final.model_dump() if decision.final else None),
                            "safety": decision.safety.model_dump(),
                            "scene": scene.model_dump(),
                        },
                        indent=2,
                    ),
                    observations,
                )

            if decision.status != "tool_call" or not decision.tool_call:
                return (
                    json.dumps(
                        {
                            "status": "no_action",
                            "reason": "decision missing tool_call",
                            "raw_decision": decision.model_dump(),
                        },
                        indent=2,
                    ),
                    observations,
                )

            # Execute tool with validation
            tool_name = decision.tool_call.name
            if tool_name not in self.tools:
                observations.append({"ok": False, "error": f"unknown_tool: {tool_name}"})
                continue

            spec = self.tools[tool_name]
            try:
                validated_args = spec.validator(decision.tool_call.args or {})
                result = spec.fn(validated_args)
            except (ValidationError, ValueError) as e:
                result = {"ok": False, "error": f"tool_validation_failed: {str(e)}", "tool": tool_name}
            except Exception as e:
                result = {"ok": False, "error": f"tool_execution_failed: {str(e)}", "tool": tool_name}

            observations.append({"tool": tool_name, "args": decision.tool_call.args, "result": result})

            self.idle()

            # Simple state update heuristic:
            # If we planned a grasp successfully, request execute_grasp next iteration.
            # (In a real system, you might update scene or capture a new image after execution.)
            # For now, keep the same image/scene to keep it single-image-per-request.
            #
            # If you DO capture a new image after a tool call, feed the new image into perceive() again.

        return (
            json.dumps({"status": "final", "final": {"summary": "max_steps_reached", "next_step_hint": ""}}, indent=2),
            observations,
        )


class System2:

    def __init__(self, system2_model, system2_prompt, skills: List[Skill], base_url: str, api_key: str, msg_buffer_size: int = 5):
        
        if system2_model not in MODEL_API_MAP:
            raise ValueError(f"Model {system2_model} not supported")
        
        # self.model_name = system2_model
        # self.system2_prompt = system2_prompt
        # self.skills = skills
        # self.attention_queue = []
        # self.action_queue: List[HighLevelAction] = []

        # self.client = OpenAI(base_url=base_url, api_key=api_key)
        # self.msg_buffer_size = msg_buffer_size
        # self.messages = []
        # self.observations = []
        # self.prompt_log = []
        # self.response_log = []

        self.cfg = AgentConfig(
            model=system2_model,
            base_url=base_url,
            api_key=api_key,
            min_confidence_to_act=0.60,
        )
        self.tools = build_tool_registry()
        self.agent = VLMReActAgent(self.cfg, self.tools)


    def idle(self):
        time.sleep(1/TPS)


    # def call_model(self, events: List[Event], observation: Observation) -> str:
    #     # call model with events and observation
    #     # return a thought

    #     # TODO: implement
    #     # prepare prompt for model, depending of VLM used
    #     # thought = self.model(self.system2_prompt, events, observation)

    #     thought = 'Action: pick_object: {"object_description": "pink lego block"}'
    #     return thought

    def _image_to_data_url(self, image: np.ndarray) -> str:
        image_bytes = image.tobytes()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime = "image/jpeg"
        return f"data:{mime};base64,{b64}"

#     def __call__(self, message, events: List[Event], observation: Observation):
        
#         data_url = self._image_to_data_url(observation.data[settings.ModuleName.GRIPPER][settings.SensorName.WRIST_CAM][settings.ValueName.IMAGE])

#         self.messages.append(
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": message},
#                     {"type": "image_url", "image_url": {"url": data_url}},
#                 ],
#             }
#         )
        
#         # if we are getting to max number of messages in history, delete older messages
#         if message.startswith("Question"):
#             self.messages = self.messages[-5:]
#         elif message.startswith("Observation"):
#             self.messages = self.messages[-3:]
        
#         result = self.execute()
        
#         self.messages.append({"role": "assistant", "content": result})
        
#         return result

#     def execute(self):

#         if self.model_name not in MODEL_API_MAP:
#             raise ValueError(f"Model {self.model_name} not supported")

#         if MODEL_API_MAP[self.model_name] == ModelApi.OPENAI:

#             messages = [{"role": "system", "content": self.system2_prompt}] + self.messages
#             self.prompt_log.append(messages)

#             completion = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=messages,
#                 max_tokens=128,
#                 temperature=0,
#                 timeout=10,
#                 stop=["\n\n", "\n4)", "\nExplanation", "<think>", "</think>"],
#             )
            
#             # Uncomment this to print out token usage each time, e.g.
#             # {"completion_tokens": 86, "prompt_tokens": 26, "total_tokens": 112}
#             # print(completion.usage)
#             self.response_log.append(completion)
#             return completion.choices[0].message.content

#     def query(self, message: str, events: List[Event], observation: Observation, max_turns: int = 10):
#         i = 0
#         num_errors = 0
#         next_prompt = "Question: " + message
#         while i < max_turns and num_errors < 3:
#             i += 1
#             result = self(next_prompt)
#             result = result.strip()
#             print("\n------------------ LLM COMPLETION START -----------------------")
#             print(result)
#             print("\n------------------ LLM COMPLETION END -----------------------")
            
#             pattern = r"Action: (\w+): (\{[^}]+\})"

#             # Use re.search to find the match
#             match = re.search(pattern, result)

#             if match:
#                 # Extract the function_name and function_input from the match
#                 action = match.group(1)
#                 action_input = match.group(2)
#                 # Load the JSON string into a Python dictionary
#                 input_data = json.loads(action_input)
#             elif "Answer:" in result:
#                 # final response, just return
#                 return
#             else:
#                 raise Exception("Unknown response format from LLM:\n" + result)
            
            
#             if action not in known_actions:
#                 raise Exception("Unknown action: {}: {}".format(action, action_input))

#             input_data = json.loads(action_input)

#             print("\n------------------------------------------------")
#             print(" -- running {} {}".format(action, input_data))
#             print("\n------------------------------------------------")
#             print("Observation{}:".format(len(self.observations) + 1))

            
#             query = input_data['query']

#             # if query is referencing to an observation, then retrieve that observation and use as query
#             if query.startswith("Observation"):
#                 query = self.observations[int(query[len("Observation"):]) - 1]

#             try:
#                 # run the action to get the observation
#                 observation = known_actions[action](query)
#             except Exception as e:
#                 # if there was an error in the action execution, capture the exception and send it
#                 # back to the agent as an observation, it should help the agent fix the issue and
#                 # run a new action
#                 observation = str(e)
                
#                 # clean up the observation error and limit its length
                
#                 # reduce vectors to a small replacement
#                 float_list_pattern = r'\[-?\d+\.\d+(e-\d\d)?(, -?\d+\.\d+(e-\d\d)?)*\]'
#                 replacement = "[0.0 ... 0.0]"
#                 observation = re.sub(float_list_pattern, replacement, observation, flags=re.DOTALL)
                
#                 # limit sice of observation
#                 observation = observation[:1024]
                
#                 self.observations.append(observation)
#                 num_errors += 1
#                 print(observation)
#             else:
                
#                 # Action ran without exceptions
#                 num_errors = 0

#                 # store the observation in the stack for reference in the future
#                 self.observations.append(observation)

#                 if action == "retrieve" or action == "get_themes":
#                     header, data = observation

#                     # render the data depending on the type of chart
#                     if input_data['chart'] == 'table':
#                         render_table(header, data)
#                     elif input_data['chart'] == 'bar_chart':
#                         render_bar_chart(header, data)
#                     elif input_data['chart'] == 'pie_chart':
#                         render_pie_chart(header, data)
#                     elif input_data['chart'] == 'line_chart':
#                         render_line_chart(header, data)
#                     else:
#                         print(data)
#                 else:
#                     print(observation)
            

#             # add observation as next prompt
#             # note we are tuncating the observation to avoid exhausting the model context
#             # we are also numbering the observations, so the agent can reference them in future actions
#             # this allows the agent to use the full data from an observation as input to another action
            
#             if not len(observation[1]):
#                 observation = "No matching records"
            
# #             next_prompt = "Observation{}: {}\nProvide your final answer, start with 'Answer:' unless you got an error, in that case, provide a Though: and Action:\n".format(len(self.observations), str(observation)[:MAX_OBSERVATION_SIZE_CHARS])
#             next_prompt = "Observation{}: {}\n".format(len(self.observations), str(observation)[:MAX_OBSERVATION_SIZE_CHARS])
            

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
        self.agent.attention_queue.append((events, observation))

    def think(self):
        # Run system2 at TPS rate
        while True:

            # get text input from user
            command = input("Enter command: ")
            
            final_json, observations = self.agent.run(
                command=command,
                width=224,
                height=224,
            )

            print(final_json)
            

            
            # self.query(message, events, observation)
            
            # events and observations could be empty
            # in that case system2 might just reflect on its own
            # eg "I should go to sleep"
            # or "I should learn something new"
            # or "I should do something fun"
            # or "I should do something to help someone"
            # or "I should do something to help the environment"
            # or "I should recharge"

            # # if there are no events and observations, just idle
            # if len(self.attention_queue) == 0:
            #     self.idle()
            #     continue
            
            # # just the last event and observation
            # # don't use events for now, just observation
            # events, observation = self.attention_queue.pop(0)
            # # clear attention queue, for now we only use the last event and observation
            # self.attention_queue = []


            # # call the VLM model to get a thought
            # thought = self.call_model(events, observation)
            # # parse the thought into a high level action
            # action = self.get_action_from_thought(thought)
            # # queue the action for execution
            # self.queue_action(action)
        
            # self.idle()


    def sleep(self):
        # Runs when not wakeup, could dream or just stay idle conserving energy (eg charging)
        pass


    def dream(self):
        # Run simulations to train system 1 and system 2 policies
        # these are randomly generated scenarios and actions, it useful to get to
        # examples that are not in the training data, to get to weird cases and fail states.
        pass






        
        