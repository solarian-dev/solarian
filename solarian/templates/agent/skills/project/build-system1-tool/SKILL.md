---
name: build-system1-tool
description: Turn a Solarian skill into a tool the AI agentic controller (System 2, an LLM planner) can call - the description and JSON Schema parameters it reads, binding it in the controller's tools with agent_enabled, the agent block in conf/controller.yaml, and testing it in simulation and dry runs before hardware. Use when the user wants to drive the robot from chat or have the agent use a new behaviour.
---

# Build a System 1 tool for the agent

Docs: `solarian docs show agentic_controller`, `skills`, `controllers`. Write the skill itself with the `add-skill` skill first.

The agentic controller (`AgentHarness`, `solarian.core.ai.agent`) plans with a vision-language model and calls skills as tools. It never writes code on the robot; it only chooses among the skills you register.

## 1. Describe the skill to the model

Class attributes on the `Skill` subclass; docstrings are not read.

```python
class PlaceSkill(Skill):
    description = "Place the held object at a point in the wrist camera image."   # required
    parameters = {
        "type": "object",
        "properties": {"point": {"type": "array", "items": {"type": "number"},
                                 "minItems": 2, "maxItems": 2,
                                 "description": "[x, y] pixels in the wrist camera image."}},
        "required": ["point"],
    }
    image_points = ("point",)        # pixel params; the model must quote an observation_id
    examples = ({"params": {"point": [320, 260]}, "when": "Holding a battery over the tray."},)
    ends_by_itself = True            # it calls finish() itself
    max_duration_s = 60.0            # the agent stops it after this
```

- Without `description` the skill is left off the tool list.
- `parameters` lists only what the model chooses. Module names, config files and other wiring are bound in the controller and never shown.
- The harness checks only `required` keys, not types: validate in `setup()` or `control_step()` and `self.fail("...")` with a reason the model can act on.
- A skill with `ends_by_itself = False` needs `max_duration_s` or a `monitor` (camera questions asked while it runs).
- Override the classmethod `tool_spec(params)` when the description depends on config, e.g. an `enum` of named poses.

## 2. Register it

```python
# {{ package }}/controller.py
tools = {
    "place": Tool(PlaceSkill, {"arm": "arm", "gripper": "gripper", "camera": "wrist_camera"}, agent_enabled=True),
}
```

The key is the tool name the model sees. Only `agent_enabled=True` tools are offered; the others stay available to keys and code.

## 3. Turn the agent on

```yaml
# conf/controller.yaml
agent:
  planner: {provider: anthropic, model: <model>, api_key_env: ANTHROPIC_API_KEY}   # or openai, openai_responses
  monitor: null
  budgets: {max_turns: 40, max_minutes: 10, retries_per_step: 2}
  cameras: [wrist_camera]
  dry_run: false
```

Needs `pip install "solarian[agent]"` and the key in `.env`. Besides your tools the model gets `observe`, `stop_action`, `update_plan`, `ask_user` and `finish_task`.

## 4. Test, cheapest first

1. **Unit**: build the skill, fill `ObservationBuffer.local()` buffers with `<ModuleObs>.zeros()`, call `setup()` and `control_step(dt)`, assert on the action.
2. **Simulation**: `solarian run-server`, `solarian run-robot --sim --view`, `solarian run-controller`, then give the agent a task in the chat.
3. **Dry run**: `agent.dry_run: true` plans and points without moving anything.
4. **Hardware**: only with the user's go-ahead, the same commands with `-d <profile>` and no `--sim`.
