# Solarian: guide for coding agents working on the framework

This repository is the `solarian` package: the core framework, its CLI, the
templates that `solarian create-project` and `create-plugin` write, and the
docs. Robot projects and plugins built *with* Solarian get their own generated
`AGENTS.md`; this one is for changing Solarian itself.

## Layout

```text
solarian/
├── solarian/
│   ├── core/            # Robot, Module, Actuator, Sensor, Skill, Controller, RelayServer, Action, Observation
│   ├── core/ai/         # the agentic controller (AgentHarness)
│   ├── cli/             # manage.py (entry point), commands/, scaffolders, agent_files.py, docs_pages.py
│   ├── templates/       # project (*.tmpl), plugin/, agent/ (AGENTS.md, skills, bootstrap skill)
│   ├── datasets/  sim/  services/  perception/  tracking/  utils/
├── tests/               # pytest; one file per area
├── docs/                # the solarian.dev docs, one page.mdx per page
├── integrations/agent-plugin/   # Cursor and Claude Code plugin (bootstrap skill)
├── solarian_build.py    # build backend: atomics on Apple Silicon, docs bundled into the wheel
├── plugins/  projects/  # separate git repositories, ignored here; change them in their own repository
```

## Check a change

```bash
pip install -e ".[all]"
pytest tests/                       # or the files for the area you touched
```

## Conventions that tests enforce

- **CLI commands** are registered lazily in `solarian/cli/commands/__init__.py` and subclass `BaseCommand` (`solarian/cli/base.py`). `tests/test_cli_help.py` requires `help`, `description` and `examples` on each, examples that start with `solarian <name>`, every option named in `usage`, and the lazy help string equal to the class's `help`. List a new command in `docs/commands/page.mdx`.
- **Agent files.** `solarian/templates/agent/` is what every project and plugin gets on its next `solarian agent sync`, so a change there reaches users' repositories. Skills need `name` (the folder name) and `description` frontmatter. After changing `bootstrap/`, run `python integrations/sync_agent_plugin.py`; `tests/test_agent_files.py` fails until the plugin copy matches.
- **Docs as markdown.** `solarian/cli/docs_pages.py` (`solarian docs`) and `src/lib/docs-markdown.ts` in the solarian-docs repo (`/llms.txt`, `/docs/<page>.md`) convert MDX the same way. Change both together.

## Skills for this repository

Read the one that matches the task; they live in `.cursor/skills/`:

- `write-docs/SKILL.md`: writing or editing a page under `docs/`.
- `write-agent-skills/SKILL.md`: writing or editing the skills and `AGENTS.md` under `solarian/templates/agent/`, the bootstrap skill, or a plugin's `agent/` folder.
- `release-to-pypi/SKILL.md`: releasing core or a plugin.

## Code style

- **No single-use wrapper methods.** A method that only forwards to one call, at one call site, is boilerplate; inline it. A helper earns its name by having several callers or by naming a non-obvious concept. Delete methods with no callers, and overrides whose body is `pass` when the base hook already does nothing.
- **No defensive reads of attributes `__init__` always sets.** `getattr(self, "x", default)` and `(d or {}).get(k) or ""` claim the value might be missing. When it never is, fix the initialisation order if that made the guard necessary, then read directly. Keep the defensiveness where it is real (payloads off the wire, genuinely optional config, attributes only some subclasses declare) and say why when it is not obvious.
- **Use imported classes directly.** Do not alias one onto `self` (`self._arm_action_cls = RightArmAction`); call `RightArmAction(...)`.
- **Comments state constraints, never narrate.** A comment says what the code cannot: a unit, an invariant, why an order matters, a trade-off. It never restates the next line or explains a change to a reviewer; that belongs in the commit message.

## Skills and modules

- **A skill never holds a Module.** It reaches a module through two channels keyed by module name: `self.observations[name]` (an `ObservationBuffer`) and `self.action_queues[name]`. It is constructed with `params={"arm": "right_arm", "gripper": "right_gripper", ...}`; occupancy is those writer keys, while `leader` and `camera` are read-only and stay out of it. That is what lets one skill run on the controller or on the robot: a name crosses a link, a `Module` object cannot.

  ```python
  # BAD - holds the module, so it cannot be started remotely
  def control_step(self, dt):
      return self.params["arm"].waypoint(fields)

  # GOOD - addresses it by name and builds its own envelope
  def control_step(self, dt):
      return Action(
          action_type=ActionType.MODULE_ACTION,
          data={self.params["arm"]: {"action_type": ActionType.MOVE_L, "data": fields}},
      )
  ```

  So modules get no action-builder methods: a module interprets actions, it does not manufacture them for others.
- **Config belongs to whoever owns the concept.** A named pose ("home", "ground") is a skill's vocabulary, so it lives in the skill's `conf/skills/<config_name>.yaml`, not in `ModuleConfig`. A module knows its ends of travel and speed limits, because those are facts about the hardware. Ask who would have to change the value if the task changed.
- **Never reach into another object's private.** If a host has to set `other._x`, either `_x` should be public or the host should not set it. Both hosts wire a skill's channels in `start_skill` through public attributes on purpose: a skill built by an operator's key handler knows nothing about buffers, so it cannot receive them at construction.
