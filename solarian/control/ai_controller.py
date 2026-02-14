import threading

from solarian.core.controller import Controller
from solarian.control.ai.system2 import System2
from solarian.control.ai.system1 import System1
from solarian.core.action import Action
# from solarian.core.observation import Observation
from solarian.conf.settings import Observation
from solarian.core.configs import ControllerConfig


class AIController(Controller):
    def __init__(self, cfg: ControllerConfig):
        super().__init__(cfg)
        
        # Creates system 2 and system 1 instances
        self.system2 = System2(cfg.system2_model, cfg.system2_prompt)
        self.system1 = System1(self.system2)


    def control_step(self, observation: Observation) -> Action:
        # system 1 runs at FPS rate, typically 30 Hz
        return self.system1.react(observation)


    def on_start(self):
        # System 2 runs in a parallel async thread, at TPS rate (~1 Hz)
        print("Starting System 2 think loop")
        threading.Thread(target=self.system2.think, daemon=True).start()