"""
Skill — Base class for skills that run as separate processes.

Skills are launched by the ThinkProcess when it decides an action should be taken.
Each skill runs at its own frequency, reads robot observations from shared-memory
observation buffers, and produces actions that get merged by the AIController.

Lifecycle::

    ThinkProcess.decide()  →  launches Skill
          ↓
    Skill.setup()
          ↓
    Skill.execute()  ← called in a loop at ``self.frequency`` Hz
          ↓
    Skill.teardown()

During ``execute()``, the skill should:
    1. Read observations via ``get_observation(module_name)``
    2. Compute actions (e.g. run a diffusion policy)
    3. Submit them via ``put_action(action_data)``
    4. Optionally report progress via ``put_skill_obs(status, data)``
"""

import time
import multiprocessing
from multiprocessing import Process, Queue
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .skill import Skill, SkillStatus


class PolicySkill(Skill):
    """
    Base class for skills that execute a policy as separate processes.

    Subclasses **must** implement :meth:`control_step`.
    """

    # ------------------------------------------------------------------
    # Process entry point
    # ------------------------------------------------------------------

    def run(self):
        """
        Main process entry point (called by ``Process.start()``).

        Runs :meth:`setup`, then loops :meth:`execute` at
        ``self.frequency`` Hz, and finally runs :meth:`teardown`.

        Subclasses that need full control over the loop may override this
        method entirely.
        """
        self.put_skill_obs(SkillStatus.RUNNING)
        try:
            self.setup()
            while not self._stop_event.is_set():
                t0 = time.monotonic()
                try:
                    result = self.control_step()
                    if result is False:
                        break
                except Exception as exc:
                    self.put_skill_obs(SkillStatus.FAILED, {"error": str(exc)})
                    break

                elapsed = time.monotonic() - t0
                sleep_time = (1.0 / self.frequency) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self.put_skill_obs(SkillStatus.COMPLETED)
        except Exception as exc:
            self.put_skill_obs(SkillStatus.FAILED, {"error": str(exc)})
        finally:
            self.teardown()


    @abstractmethod
    def control_step(self) -> Any:
        """
        Execute one step of the skill.

        This is called at ``self.frequency`` Hz. Implementations should:

        1. Read observations with :meth:`get_observation`
        2. Compute actions (e.g. via a diffusion policy)
        3. Submit them with :meth:`put_action`
        4. Optionally report progress via :meth:`put_skill_obs`

        Returns:
            ``False`` to signal the skill should stop, or any other value
            (including ``None``) to continue.
        """
