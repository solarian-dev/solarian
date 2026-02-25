"""
ThinkProcess — Runs the perceive-decide loop in a separate OS process.

The think process is the "slow brain" that operates at ~5 Hz.  It reads
camera images from shared-memory observation buffers, calls a VLM for
scene understanding (``perceive``), incorporates skill status reports and
user commands, then decides which skills to launch or stop (``decide``).

Architecture::

    observation_buffers (shared mem)
            │  read
            ▼
    ThinkProcess (5 Hz)
        perceive(image, command) → structured scene via VLM
        decide(scene, user_msg, skills_obs) → skills to start / stop
            │  launch / stop
            ▼
    Skill processes
"""

import time
import base64
import json
import multiprocessing
from multiprocessing import Process, Queue
from typing import Dict, List, Any, Optional

import numpy as np

from solarian.core.data import ImageData
from solarian.control.ai.system2.skill import Skill, SkillStatus


# ---------------------------------------------------------------------------
# VLM system prompts
# ---------------------------------------------------------------------------

PERCEPTION_SYSTEM = (
    "You are a vision perception module for a robot. "
    "You will be given a scene image from the robot's camera.\n"
    "Return ONLY valid JSON matching this schema:\n"
    "{\n"
    '  "type": "scene_state",\n'
    '  "frame_id": "...",\n'
    '  "camera": {"name": "...", "image_width": <int>, "image_height": <int>},\n'
    '  "objects": [\n'
    '    {"id": "...", "label": "...", "bbox_xyxy": [x1,y1,x2,y2],\n'
    '     "confidence": 0..1, "attributes": [...], "notes": "..."}\n'
    "  ],\n"
    '  "relations": [\n'
    '    {"subject_id": "...",\n'
    '     "relation": "on_top_of|inside|left_of|right_of|in_front_of|behind|touching",\n'
    '     "object_id": "...", "confidence": 0..1}\n'
    "  ],\n"
    '  "uncertainties": [\n'
    '    {"kind": "occlusion|lighting|out_of_view|ambiguous", "detail": "..."}\n'
    "  ]\n"
    "}\n"
    "Rules:\n"
    "- No extra keys. No markdown. No explanation.\n"
    "- If unsure, lower confidence and add an uncertainties entry.\n"
)

SKILL_DECISION_SYSTEM = (
    "You are a robot skill orchestrator.\n"
    "Given the current scene, user command, and skill execution status, "
    "decide which skills to start or stop.\n\n"
    "Return ONLY valid JSON matching this schema:\n"
    "{\n"
    '  "skills_to_start": [{"skill_name": "...", "params": {...}}],\n'
    '  "skills_to_stop": ["skill_name", ...],\n'
    '  "reasoning": "..."\n'
    "}\n\n"
    "Rules:\n"
    "- No explanation outside the JSON. No markdown.\n"
    "- Only reference skills from the available skills list.\n"
    "- Consider currently active skills before deciding.\n"
    "- If no action is needed, return empty lists.\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_data_url(image: np.ndarray) -> str:
    """Encode a numpy image as a JPEG base64 data URL for VLM input."""
    import cv2

    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Failed to JPEG-encode image")
    b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _parse_json(text: str) -> Dict[str, Any]:
    """Parse JSON from VLM output, stripping common markdown wrappers."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# ThinkProcess
# ---------------------------------------------------------------------------

class ThinkProcess(Process):
    """
    Runs the perceive-decide loop at :data:`THINK_FREQUENCY` Hz in a
    separate OS process.

    It reads camera images from shared-memory observation buffers, calls a VLM
    for scene understanding (``perceive``), incorporates skill status reports
    and user commands, then decides which skills to launch or stop (``decide``).

    Args:
        observation_buffers: Shared-memory ``ObservationRingBuffer`` dict
            (module_name → buffer).
        action_queue: Skills put action dicts here for the controller.
        skills_obs_queue: Skills put status dicts here for the think process.
        user_msg_queue: User commands arrive here.
        vlm_config: Dict with keys ``model``, ``base_url``, ``api_key``,
            ``temperature``, ``timeout_s``.
        available_skills: Registry mapping skill name → ``Skill`` subclass.
        running_event: Cleared to signal the process to shut down.
    """

    def __init__(
        self,
        observation_buffers: Dict[str, Any],
        action_queue: Queue,
        skills_obs_queue: Queue,
        user_msg_queue: Queue,
        vlm_config: Dict[str, Any],
        available_skills: Dict[str, type],
        frequency: float = 2.0,
        running_event: Any,
    ):
        super().__init__(name="ThinkProcess", daemon=True)
        self.observation_buffers = observation_buffers
        self.action_queue = action_queue
        self.skills_obs_queue = skills_obs_queue
        self.user_msg_queue = user_msg_queue
        self.vlm_config = vlm_config
        self.available_skills = available_skills
        self.frequency = frequency
        self.running_event = running_event

        # Populated at runtime inside the child process
        self.active_skills: Dict[str, Skill] = {}
        self.skills_obs_history: List[Dict] = []
        self.last_user_msg: Optional[str] = None
        self.client: Any = None

        self._create_vlm_client()

    # ------------------------------------------------------------------
    # VLM interaction
    # ------------------------------------------------------------------

    def _create_vlm_client(self):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=self.vlm_config["base_url"],
            api_key=self.vlm_config["api_key"],
        )

    def _chat(
        self,
        system: str,
        user_text: str,
        image_data_url: Optional[str] = None,
        max_tokens: int = 512,
    ) -> str:
        """Send a single-turn chat completion to the VLM."""
        user_content: list = [{"type": "text", "text": user_text}]
        if image_data_url:
            user_content.append(
                {"type": "image_url", "image_url": {"url": image_data_url}}
            )

        resp = self.client.chat.completions.create(
            model=self.vlm_config["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=self.vlm_config.get("temperature", 0.0),
            max_tokens=max_tokens,
            timeout=self.vlm_config.get("timeout_s", 10.0),
        )
        return (resp.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def perceive(self, image: np.ndarray, command: str) -> Optional[Dict]:
        """
        Use a VLM to produce a structured scene description from an image.

        The method encodes the image, sends it to the VLM along with the
        current user command, and parses the JSON response.

        Args:
            image: Camera image as a numpy array (H, W, C).
            command: Current user command / task description.

        Returns:
            A dict representing the scene state, or ``None`` on failure.
        """
        image_data_url = _image_to_data_url(image)
        h, w = image.shape[:2]
        frame_id = str(int(time.time() * 1000))

        user_text = (
            f"Task command: {command}\n"
            f"frame_id: {frame_id}\n"
            f"image_width: {w}\n"
            f"image_height: {h}\n"
            "Detect relevant objects and their spatial relations.\n"
        )

        raw = self._chat(PERCEPTION_SYSTEM, user_text, image_data_url, max_tokens=400)
        try:
            return _parse_json(raw)
        except Exception:
            repair = (
                user_text
                + "\nINVALID OUTPUT. Return ONLY valid JSON. No markdown.\n"
            )
            raw2 = self._chat(PERCEPTION_SYSTEM, repair, image_data_url, max_tokens=400)
            try:
                return _parse_json(raw2)
            except Exception as exc:
                print(f"[Think] Perception failed: {exc}")
                return None

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(
        self,
        scene: Optional[Dict],
        user_msg: str,
        skills_obs: List[Dict],
    ) -> Dict:
        """
        Decide which skills to start or stop.

        Uses the VLM to reason about the current scene state, the user
        command, and skill execution feedback.

        Returns:
            Dict with ``skills_to_start``, ``skills_to_stop``, ``reasoning``.
        """
        available_text = "\n".join(
            f"- {name}: {getattr(cls, '__doc__', '') or 'No description'}"
            for name, cls in self.available_skills.items()
        )
        active_text = (
            "\n".join(f"- {name}: running" for name in self.active_skills)
            or "None"
        )
        scene_text = json.dumps(scene, default=str) if scene else "Perception unavailable"
        obs_text = json.dumps(skills_obs[-10:], default=str) if skills_obs else "[]"

        user_text = (
            f"User command: {user_msg}\n\n"
            f"Available skills:\n{available_text}\n\n"
            f"Currently active skills:\n{active_text}\n\n"
            f"Scene:\n{scene_text}\n\n"
            f"Recent skill observations:\n{obs_text}\n\n"
            "Decide which skills to start or stop.\n"
        )

        raw = self._chat(SKILL_DECISION_SYSTEM, user_text, max_tokens=300)
        try:
            return _parse_json(raw)
        except Exception:
            repair = (
                user_text
                + "\nINVALID. Return ONLY valid JSON matching the schema.\n"
            )
            raw2 = self._chat(SKILL_DECISION_SYSTEM, repair, max_tokens=300)
            try:
                return _parse_json(raw2)
            except Exception as exc:
                print(f"[Think] Decision failed: {exc}")
                return {
                    "skills_to_start": [],
                    "skills_to_stop": [],
                    "reasoning": "decision_parse_failure",
                }

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    # TODO: could be more than one image and the images to be used should be configurable
    def _get_latest_image(self) -> Optional[np.ndarray]:
        """Extract the most recent camera image from any observation buffer."""
        for _module_name, buffer in self.observation_buffers.items():
            try:
                obs = buffer.get()
            except Exception:
                continue
            if obs is None:
                continue

            for sensor_name, sensor_obs in obs.items():
                for data_name, data in sensor_obs.items():
                    if isinstance(data, ImageData):
                        return data.value
        return None

    def _drain_queue(self, queue: Queue) -> List:
        """Non-blocking drain of all items currently in *queue*."""
        items: List = []
        while True:
            try:
                items.append(queue.get_nowait())
            except Exception:
                break
        return items

    # ------------------------------------------------------------------
    # Skill management
    # ------------------------------------------------------------------

    def _cleanup_finished_skills(self):
        finished = [n for n, s in self.active_skills.items() if not s.is_alive()]
        for name in finished:
            del self.active_skills[name]

    def _launch_skill(self, skill_name: str, params: Dict):
        if skill_name in self.active_skills:
            print(f"[Think] Skill '{skill_name}' already active, skipping")
            return

        cls = self.available_skills.get(skill_name)
        if cls is None:
            print(f"[Think] Unknown skill: {skill_name}")
            return

        skill = cls(
            name=skill_name,
            observation_buffers=self.observation_buffers,
            action_queue=self.action_queue,
            skills_obs_queue=self.skills_obs_queue,
            params=params,
        )
        skill.start()
        self.active_skills[skill_name] = skill
        print(f"[Think] Launched skill '{skill_name}' (PID {skill.pid})")

    def _stop_skill(self, skill_name: str):
        skill = self.active_skills.get(skill_name)
        if skill is not None and skill.is_alive():
            skill.request_stop()
            print(f"[Think] Requested stop for skill '{skill_name}'")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _think_step(self):
        """Execute one perceive-decide cycle."""
        # 1. Drain incoming queues
        for obs in self._drain_queue(self.skills_obs_queue):
            self.skills_obs_history.append(obs)

        user_msgs = self._drain_queue(self.user_msg_queue)
        if user_msgs:
            self.last_user_msg = user_msgs[-1]

        self._cleanup_finished_skills()

        # Nothing to do until the user sends a command
        if not self.last_user_msg:
            return

        # 2. Get latest image from observation buffers
        image = self._get_latest_image()
        if image is None:
            return

        # 3. Perceive — structured scene from image via VLM
        scene = self.perceive(image, self.last_user_msg)

        # 4. Decide — which skills to start / stop
        decision = self.decide(scene, self.last_user_msg, self.skills_obs_history)

        # 5. Act on the decision
        for spec in decision.get("skills_to_start", []):
            self._launch_skill(spec["skill_name"], spec.get("params", {}))

        for name in decision.get("skills_to_stop", []):
            self._stop_skill(name)

    def run(self):
        """Process entry point — loops ``_think_step`` at 5 Hz."""
        print(
            f"[Think] Process started (PID {multiprocessing.current_process().pid}), "
            f"frequency={self.frequency} Hz"
        )
        

        while self.running_event.is_set():
            t0 = time.monotonic()
            try:
                self._think_step()
            except Exception as exc:
                print(f"[Think] Error in think step: {exc}")

            elapsed = time.monotonic() - t0
            sleep_time = (1.0 / self.frequency) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Graceful shutdown — stop all active skills
        for name, skill in list(self.active_skills.items()):
            if skill.is_alive():
                skill.request_stop()
        for name, skill in list(self.active_skills.items()):
            skill.join(timeout=3.0)
            if skill.is_alive():
                skill.terminate()

        print("[Think] Process stopped")
