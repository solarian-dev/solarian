import threading

from proto.control.controller import Controller
from proto.control.ai.system2 import System2
from proto.control.ai.system1 import System1
from proto.robots.action import Action
from proto.robots.observation import Observation
from proto.common.configs import ControllerConfig


class AIController(Controller):
    def __init__(self, cfg: ControllerConfig):
        super().__init__(cfg)
        
        self.system2 = System2(cfg.system2_model, cfg.system2_prompt)
        self.system1 = System1(self.system2)


    def control_step(self, observation: Observation) -> Action:
        # system 1 runs at FPS rate, typically 30 Hz
        return self.system1.react(observation)


    def connect(self):
        super().connect()
        
        # system 2 runs in a parallel async thread, at TPS rate
        threading.Thread(target=self.system2.think, daemon=True).start()