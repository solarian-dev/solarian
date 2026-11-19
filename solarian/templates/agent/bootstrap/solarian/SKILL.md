---
name: solarian
description: Create or work on a Solarian robot project or Solarian plugin (robots, arms, grippers, cameras, modules, skills, policies, UMI, teleoperation, recording and training with the `solarian` CLI). Use when the user mentions Solarian, solarian.dev, a solarian-* package, or asks to build a robot project with it.
---

# Solarian

Solarian is a Python robotics framework, driven by the `solarian` CLI. Let the CLI
write the files it owns; do not hand-write a project or plugin layout.

## 1. Make sure Solarian is installed

```bash
solarian --version || pip install solarian
```

Use the user's virtual environment if the folder has one.

## 2. Start from the CLI

- New robot project: `solarian create-project <name>` (a valid Python identifier, such as `my_robot`), then `cd <name>` and `pip install -e ".[all]"` (or `.[robot]`, `.[server]`, `.[controller]` when the roles run on separate machines).
- New plugin (a driver, module, robot, skill or policy others can install): `solarian create-plugin <type> <name>`, where `<type>` is `actuator`, `sensor`, `module`, `robot`, `command`, `policy` or `op`. Run it again with another type to add a component to the same plugin.
- Adding hardware or a skill someone published: `solarian install <plugin>` inside the project, which also copies its config into `conf/`.

## 3. Then read AGENTS.md

`create-project` and `create-plugin` write an `AGENTS.md` and task skills (add a module, add a skill, record and train, ...) into what they create. Read `AGENTS.md` before changing anything, and follow the skill for the task.

In an existing project or plugin with no `AGENTS.md`, run `solarian agent sync` to write one.

## Docs

The docs for the installed version, offline:

```bash
solarian docs list
solarian docs show <page>          # e.g. skills, modules, tutorials/first-robot/project
solarian docs search <term>
```

Online: https://solarian.dev/llms.txt indexes every page; add `.md` to any docs URL for markdown, as in https://solarian.dev/docs/skills.md.
