---
name: write-agent-skills
description: Write or edit the agent files Solarian ships to coding agents - the task skills and AGENTS.md templates under solarian/templates/agent/, the bootstrap skill and its marketplace plugin copy, and the agent/ folder a plugin contributes. Use when adding, changing, renaming or removing one of those skills, or the AGENTS.md a project or plugin gets.
---

# Writing agent skills

Everything here reaches users' repositories on their next `solarian agent sync`,
so a mistake ships to every project. `solarian/cli/agent_files.py` does the writing;
`solarian docs show agents` is the user-facing page.

## Where each file goes

| Source | Ends up in | Rendered |
| --- | --- | --- |
| `solarian/templates/agent/skills/project/<name>/` | every project's `.agents/skills/<name>/` and `.claude/skills/<name>/` | Jinja |
| `solarian/templates/agent/skills/plugin/<name>/` | the same folders in every plugin | Jinja |
| `solarian/templates/agent/{project,plugin}/AGENTS.md.j2` | `AGENTS.md` (and a `CLAUDE.md` that imports it) | Jinja |
| `solarian/templates/agent/bootstrap/solarian/` | `~/.agents/skills`, `~/.claude/skills` via `solarian agent install`, and `integrations/agent-plugin/skills/` via `sync.py` | no |
| `solarian/templates/agent/NOTES.md` | `.agents/NOTES.md`, created once, never rewritten | no |
| a plugin's `<package>/agent/AGENTS.md` | a section of each project's `AGENTS.md`, HTML comments stripped | no |
| a plugin's `<package>/agent/skills/<name>/` | the project's skill folders as `<short>-<name>` | no |

A new folder under `skills/project/` or `skills/plugin/` is all it takes: the sync
picks it up and `AGENTS.md` lists it with its description. Deleting the folder
removes the skill from projects on their next sync, unless the user edited it.
A rename is a delete plus an add.

## The SKILL.md

```markdown
---
name: add-service
description: Add a Solarian service - ... Use when work is too heavy for a skill's own loop.
---
```

- `name` is the folder name. For plugin skills the sync rewrites it to `<short>-<name>`.
- `description` is the only part an agent always sees, so it decides whether the skill
  is used. Say what it does, then "Use when ..." with the words a user would type.
- Keep both on one line each. `frontmatter()` reads `key: value` lines; YAML folding
  (`>`, `|`, continuation lines) is not parsed.
- In the body, point at the docs (`solarian docs show <page>`) and the CLI
  (`solarian <command> --help`) rather than copying them; they match the installed
  version, a copy goes stale. Show the code the agent writes, and end with a
  "Check it" step that runs something.
- Follow the framework rules in `AGENTS.md` (a skill never holds a `Module`, config
  belongs to its owner) in every example. Agents copy examples literally.

## Jinja in core templates

Core skills and `AGENTS.md.j2` render with `StrictUndefined`, `trim_blocks` and
`lstrip_blocks`. Available in every template: `package` (import package, e.g.
`my_robot`), `name` (distribution name), `short` (`name` without `solarian-`),
`version`, `notes` (`.agents/NOTES.md`), `stamp`. `AGENTS.md.j2` also gets `skills`
(name and description), plus `plugins` for a project or `groups` (entry-point
groups) for a plugin.

- Use `{{ package }}/modules/...` for paths, never `my_robot/...`.
- A misspelt variable fails the sync, not silently. A literal `{{` or `{%` cannot
  appear in the rendered output at all: the tests reject it. Rephrase rather than
  escape.
- Only `SKILL.md` is rendered; other files in a skill folder are copied byte for byte.
- The bootstrap skill and plugin-shipped files are not rendered, so no Jinja there.

## The bootstrap skill

`bootstrap/solarian/SKILL.md` is installed outside any project, so it must stay
valid across Solarian versions: it only points at the CLI (`pip install solarian`,
`create-project`, `create-plugin`, `solarian docs`). Its description is what makes
an agent find Solarian from an empty folder; rerun the evals in
`integrations/agent-plugin/EVALS.md` after changing it.

After any change under `bootstrap/`:

```bash
python integrations/agent-plugin/sync.py           # copy into the marketplace plugin
python integrations/agent-plugin/sync.py --check   # exit 1 while it is out of date
```

Commit the copy with the change. `test_the_marketplace_plugin_matches_the_templates`
fails until they match, and the release (`release-to-pypi`) runs the sync again to
set the version.

## Skills a plugin ships

These live in the plugin's own repository, under its import package:

```text
solarian_myarm/agent/
├── AGENTS.md                  # a few lines: what it provides, the class to import, the usual mistake
└── skills/calibrate/SKILL.md  # becomes myarm-calibrate
```

The plugin's `pyproject.toml` must list them, or the wheel ships without them:

```toml
[tool.setuptools.package-data]
solarian_myarm = ["agent/*.md", "agent/skills/*/*"]
```

`solarian create-plugin` writes both. Plugin files are not rendered, so write the
plugin's own package name out.

## Check it

```bash
pytest tests/test_agent_files.py
mkdir -p tmp && cd tmp && solarian create-project skillbot && cat skillbot/.agents/skills/<name>/SKILL.md
```

Read the rendered file as the agent will: is the description enough to pick it, and
does every command and path in it work in `skillbot`? Delete `tmp/skillbot` after.
For a plugin skill, run `solarian agent sync` in a project with the plugin installed
from source (`pip install -e`).
