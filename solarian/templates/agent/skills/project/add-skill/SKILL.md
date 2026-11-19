---
name: add-skill
description: Write a Solarian Skill (a behaviour that reads observations and sends module actions at its own rate, scripted or policy-driven) in this project, give it a config, and start it from the controller or a key. Use when the user wants the robot to do something on its own.
---

# Add a skill

Docs: `solarian docs show skills`, `controllers`, `events`, `tutorials/skills/move`. To let the AI agent call it, also follow the `build-system1-tool` skill. A published skill may already exist: check `solarian docs show plugins` (e.g. `solarian install move`).

## 1. The class

```python
# {{ package }}/skills/open_gripper.py
from solarian.core.action import Action, ActionType
from solarian.core.execution import ExecutionMode
from solarian.core.skill import Skill


class OpenGripperSkill(Skill):
    execution_mode = ExecutionMode.INLINE
    config_name = "open_gripper"            # conf/skills/open_gripper.yaml, optional

    def setup(self):
        self.width_mm = float(self.load_config().get("width_mm", 80.0))
        self.sent = False

    def control_step(self, dt):
        gripper = self.params["gripper"]
        if not self.sent:
            self.sent = True
            return Action(action_type=ActionType.MODULE_ACTION, data={gripper: {
                "action_type": ActionType.MOVE_J, "data": {"gripper": {"value": self.width_mm}}, "params": {},
            }})
        obs = self.observations[gripper].latest()
        if obs is not None and obs["gripper"].value > 90:    # normalized: 100 is fully open
            self.finish()
        return None
```

Export it from `{{ package }}/skills/__init__.py`.

- `control_step(dt)` returns an `Action`, a list of them, or `None`. End with `self.finish()`, or `self.fail("reason")`.
- Read `self.observations[name].latest()` (`None` until a first sample; fields as `obs["<field>"].value`), `.last_k(n)`, and this tick's `self.events`. Never hold a `Module` object; address modules by the names in `self.params`.
- On the controller, only the modules named in `params["arm"]` and `params["gripper"]` get action queues. Those keys also decide preemption: starting a skill stops running skills that share a module.
- `frequency` defaults to 10 Hz and is not read from yaml; set it in `setup()` if needed. `INLINE` skills step on every controller tick.
- `execution_mode` (`PROCESS` by default, `INLINE`) and `execution_place` (`CONTROLLER` by default, `ROBOT`, `SERVER`) say where it runs. A skill placed on the robot or server must be importable there.

## 2. Its config (optional)

`self.load_config()` merges a `conf.yaml` next to the class with `conf/skills/<config_name>.yaml` (the project file wins) and resolves `${deployment.*}`. Named poses and task parameters go here, not in module yaml.

## 3. Start it

```python
# {{ package }}/controller.py
from solarian.core.skill import Tool

from {{ package }}.skills import OpenGripperSkill


class MyController(Controller):
    tools = {"open_gripper": Tool(OpenGripperSkill, {"gripper": "gripper"})}

    def on_start(self):
        self.start_tool("open_gripper")
```

Or `self.start_skill(OpenGripperSkill(params={"gripper": "gripper"}))`, or from a key handler (`@event_handler(EventType.KEY_DOWN)` from `solarian.core.event`). `stop_skill(name)` and `stop_all_skills()` stop them.

## 4. Check it

Unit-test `control_step` without a robot: fill an `ObservationBuffer.local()` (`solarian.utils.shared_memory.observation_buffer`) with `<ModuleObs>.zeros()`, set `skill.observations = {"gripper": buf}`, call `setup()`, and assert on `skill.control_step(0.1).to_dict()["data"]`. Then run it in simulation: `solarian run-server`, `solarian run-robot --sim --view`, `solarian run-controller`.
