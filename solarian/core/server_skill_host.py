"""Skills hosted by the relay server (``ExecutionPlace.SERVER``).

The relay is usually the most capable machine in the loop and already sees
every observation, including the fields its own processors compute. A skill
placed here reads those observations from shared-memory buffers the relay
fills as it forwards state, and its actions are sent to the robot as one
batch per relay tick. That keeps the robot to capturing, publishing and
executing: no skill process and no policy round trip on a small board.

Skills are wired the way :meth:`Robot.start_skill` wires them, so a skill
does not know which of the two hosts it is running on.
"""

import logging
import os
import time
from multiprocessing.managers import SharedMemoryManager

from solarian.core.action import Action, ActionType, ModuleAction
from solarian.core.skill_registry import discover_skill_classes
from solarian.core.event import event_inbox_for
from solarian.core.execution import ExecutionPlace
from solarian.core.module import Module
from solarian.core.observation import Observation
from solarian.utils import StatusBoard, get_class
from solarian.utils.configs import ConfigError, load_config_yaml
from solarian.utils.shared_memory.action_queue import ModuleActionQueue
from solarian.utils.shared_memory.observation_buffer import ObservationBuffer
from solarian.utils.shared_memory.skill_status_buffer import SkillStatusBuffer

logger = logging.getLogger(__name__)

#: How long a stopping skill gets before it is terminated.
_STOP_TIMEOUT_S = 5.0


def _bind_robot_module_schemas() -> None:
    """Instantiate the robot's modules schema-only so image shapes are known.

    Same reason as the controller's ``_setup_remote_observation_schema``: the
    shared-memory buffers are sized from ``Observation.zeros()``, which reads
    shapes the module configs set on class-level descriptors. Nothing is
    connected.
    """
    from solarian.conf import settings

    from solarian.core.conf_paths import module_config_path

    robot_cls = get_class(settings.ROBOT_CLASS)
    for name, attr in list(robot_cls.__dict__.items()):
        if not isinstance(attr, Module):
            continue
        yaml_path = module_config_path(name)
        if not yaml_path.exists():
            logger.warning("No config for robot module '%s'; skipping its schema", name)
            continue
        attr.bind(load_config_yaml(str(yaml_path), type(attr).config_class), name)


class ServerSkillHost:
    def __init__(self):
        _bind_robot_module_schemas()
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        self.skills_status_buffer = SkillStatusBuffer(
            shm_manager=self.shm_manager,
            get_max_k=64,
            get_time_budget=0.2,
            put_desired_frequency=60,
        )
        example_observation = Observation.zeros()
        example_action = Action.zeros()
        self.observation_buffers: dict[str, ObservationBuffer] = {}
        self.action_queues: dict[str, ModuleActionQueue] = {}
        for name, module_obs in example_observation.items():
            self.observation_buffers[name] = ObservationBuffer.create_from_example(
                shm_manager=self.shm_manager,
                example_observation=module_obs,
                get_time_budget=0.2,
                put_desired_frequency=200,
            )
            try:
                example = example_action[name]
            except KeyError:
                continue
            self.action_queues[name] = ModuleActionQueue(
                shm_manager=self.shm_manager,
                example_action=example,
                buffer_size=1024,
            )
        self._zeros = example_observation
        self._local_skills: dict = {}
        self._stopping: dict = {}
        #: ``START_SKILL`` payloads that arrived while a skill of the same name
        #: was still stopping, started once :meth:`reap` retires it.
        self._deferred_starts: dict[str, dict] = {}
        self._skill_classes: dict[str, type] | None = None

    # -- observations ---------------------------------------------------------

    def put_observation(self, observation: Observation) -> None:
        """File each module's latest reading, completing it to the schema.

        The shared-memory ring is laid out from the full module schema, so a
        reading missing a field (a calculated one nobody computed yet) is
        completed with that field's zero rather than refused.
        """
        for module_name, module_obs in observation.items():
            buffer = self.observation_buffers.get(module_name)
            if buffer is None or module_obs is None:
                continue
            zeros = self._zeros[module_name]
            for field_name, zero in zeros.items():
                try:
                    module_obs[field_name]
                except KeyError:
                    module_obs[field_name] = zero
            try:
                buffer.put(module_obs, wait=False)
            except Exception:  # noqa: BLE001 - a dropped sample must not stall the relay
                logger.debug("Could not buffer '%s' observation", module_name, exc_info=True)

    # -- skills ---------------------------------------------------------------

    def skill_classes(self) -> dict[str, type]:
        if self._skill_classes is None:
            self._skill_classes = {
                skill_cls.registry_name(): skill_cls
                for skill_cls in discover_skill_classes().values()
                if isinstance(skill_cls, type)
            }
        return self._skill_classes

    def runs(self, name: str) -> bool:
        return name in self._local_skills

    def start_named_skill(self, data: dict) -> None:
        """Start the skill a ``START_SKILL`` addressed to the server names.

        Every read is a ``.get``: the payload was deserialized off the wire.
        """
        name = str(data.get("skill") or "")
        skill_cls = self.skill_classes().get(name)
        if skill_cls is None:
            logger.warning(
                "Cannot start unknown skill '%s'; installed: %s",
                name,
                sorted(self.skill_classes()),
            )
            return
        running = self._local_skills.get(name)
        if running is not None and running.is_alive():
            logger.info("Skill '%s' already running (pid=%s)", name, running.pid)
            return
        if name in self._stopping:
            logger.info("Skill '%s' is still stopping; starting it once it has", name)
            self._deferred_starts[name] = data
            return
        try:
            skill = skill_cls(
                params=data.get("params", {}),
                execution_place=ExecutionPlace.SERVER,
                status_mp_q=StatusBoard.get_mp_queue(),
            )
            skill.observations = self.observation_buffers
            skill.action_queues = self.action_queues
            skill.skills_status_buffer = self.skills_status_buffer
            skill.event_inbox = event_inbox_for(skill)
            skill.start()
        except ConfigError as exc:
            logger.error("Could not start skill '%s': %s", name, exc)
            return
        except Exception:
            logger.exception("Could not start skill '%s'", name)
            return
        self._local_skills[name] = skill
        logger.info("Started skill '%s' on the server (pid=%s)", name, skill.pid)

    def stop_skill(self, name: str) -> None:
        self._deferred_starts.pop(name, None)
        skill = self._local_skills.pop(name, None)
        if skill is None:
            return
        logger.info("Stopping skill '%s' (pid=%s)", name, skill.pid)
        skill.request_stop()
        self._stopping[name] = (skill, time.monotonic() + _STOP_TIMEOUT_S)

    def stop_all_skills(self) -> None:
        self._deferred_starts.clear()
        for name in list(self._local_skills):
            self.stop_skill(name)

    def reap(self) -> None:
        """Retire stopped skills and cancel the chunks they left scheduled.

        The waypoints a skill scheduled outlive it, so the robot is told to drop
        them once the skill is dead — its final tick dispatches one more chunk
        after the stop is requested.
        """
        now = time.monotonic()
        for name, (skill, deadline) in list(self._stopping.items()):
            if skill.is_alive():
                if now < deadline:
                    continue
                logger.warning("Skill '%s' did not stop; terminating", name)
                skill.terminate()
                skill.join(timeout=1.0)
            self._stopping.pop(name)
            for module_name in skill.commanded_module_names():
                queue = self.action_queues.get(module_name)
                if queue is not None:
                    queue.put(ModuleAction(action_type=ActionType.CANCEL_TRAJECTORY))
            logger.info("Skill '%s' stopped", name)
            deferred = self._deferred_starts.pop(name, None)
            if deferred is not None:
                self.start_named_skill(deferred)
        for name, skill in list(self._local_skills.items()):
            if not skill.is_alive():
                self._local_skills.pop(name)
                logger.info("Skill '%s' exited", name)

    def drain_actions(self) -> list[Action]:
        """Every queued module action, as ``MODULE_ACTION``s in queue order.

        Merged index-wise across modules, like the controller does: the
        arm's and the gripper's n-th waypoint of a chunk travel together, and
        a chunk of successive waypoints does not collapse to its last one.
        """
        slots: list[dict] = []
        for module_name, queue in self.action_queues.items():
            if queue.empty():
                continue
            for index, module_action in enumerate(queue.get_all()):
                while len(slots) <= index:
                    slots.append({})
                slots[index][module_name] = module_action.to_dict()
        return [Action(action_type=ActionType.MODULE_ACTION, data=slot) for slot in slots]

    def drain_status_records(self) -> list[dict]:
        records = self.skills_status_buffer.drain()
        return [
            {"skill_id": r.skill_id, "status": r.status, "timestamp": r.timestamp}
            for r in records
        ]

    def shutdown(self) -> None:
        self.stop_all_skills()
        deadline = time.monotonic() + _STOP_TIMEOUT_S
        while self._stopping and time.monotonic() < deadline:
            self.reap()
            time.sleep(0.05)
        self.shm_manager.shutdown()
