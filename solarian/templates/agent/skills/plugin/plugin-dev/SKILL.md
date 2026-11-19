---
name: plugin-dev
description: Develop this Solarian plugin - add a component (actuator, sensor, module, robot, command, policy, dataset op), bundle a default config, teach agents how to use it, test it in a project, and release it to PyPI. Use for any change to this plugin's code, entry points or packaging.
---

# Developing {{ name }}

Read `AGENTS.md` first. `solarian docs show plugins` is the reference for everything below.

## Add a component

```bash
solarian create-plugin <type> {{ short }}
pip install -e .
```

`<type>` is `actuator`, `sensor`, `module`, `robot`, `command`, `policy` or `op`. The command writes the stub under `{{ package }}/` and merges its entry points into `pyproject.toml`; re-running it with a new type never touches files that exist. Implement the stub, keeping the class name the entry point names.

| Type | Base class | Entry-point group(s) |
| --- | --- | --- |
| actuator | `Servo` / `Actuator`, plus an `ActuatorBus` | `solarian.actuators`, `solarian.actuator_buses` |
| sensor | `Sensor`, plus a `SensorBus` | `solarian.sensors`, `solarian.sensor_buses` |
| module | `Module` (or `ArmModule`, `GripperModule`) | `solarian.modules` |
| robot | `Robot` | `solarian.robots` |
| command | `BaseCommand` | `solarian.commands` |
| policy | trainer, service and skill | `solarian.trainers`, `solarian.services`, `solarian.skills` |
| op | `DatasetOp` | `solarian.dataset_ops` |

## Bundle a default config

Put it under `{{ package }}/templates/<folder>/<name>.yaml`, in the folder of `conf/` that reads it: `templates/modules/<name>.yaml` becomes the project's `conf/modules/<name>.yaml` when someone runs `solarian install {{ short }}`. The pattern `templates/*/*.yaml` must stay in `[tool.setuptools.package-data]`.

## Teach agents to use it

- `{{ package }}/agent/AGENTS.md`: a short section that lands in the `AGENTS.md` of every project with this plugin installed. Say what the plugin provides, the class to import, the config it seeds, and the mistake agents make with it.
- `{{ package }}/agent/skills/<skill>/SKILL.md`: a task recipe (calibrate this arm, train this policy). It is copied into projects as `{{ short }}-<skill>`. Give it `name` and `description` frontmatter; the description is what makes an agent pick it.

`agent/*.md` and `agent/skills/*/*` must stay in `[tool.setuptools.package-data]`.

## Test it in a project

```bash
solarian create-project plugin_check          # once, next to the plugin
cd plugin_check
solarian install ../{{ name }}                # editable install, conf/ seeded, dependency added
solarian run-robot --sim                      # after attaching the module to the robot
```

`solarian install` with a path installs editable, so edits to the plugin show up without reinstalling, except entry-point changes, which need `pip install -e ../{{ name }}` again.

## Release

Bump `version` in `pyproject.toml`, check the build (`python -m build` then `python -m twine check dist/*`, and `unzip -l` the wheel to confirm `templates/` and `agent/` are inside), then publish. `solarian docs show plugins` has the publishing section.
