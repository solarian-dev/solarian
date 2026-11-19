# Solarian plugin for coding agents

This plugin gives Cursor and Claude Code the `solarian` skill, so that a request
such as "create a Solarian project" or "write a Solarian plugin for my servo"
finds the `solarian` CLI in any folder. It is the same skill that
`solarian agent install` copies into the user's skill folders.

The skill only points at the CLI. The detailed guides arrive with the project:
`solarian create-project` and `solarian create-plugin` write an `AGENTS.md` and
task skills for the installed Solarian version, and `solarian agent sync`
refreshes them. See [Using Solarian with coding agents](https://solarian.dev/docs/agents).

## Install

Claude Code:

```bash
/plugin marketplace add solarian-dev/solarian
/plugin install solarian@solarian
```

Cursor reads the marketplace manifest in `.cursor-plugin/marketplace.json` at
the repository root. To try the plugin locally, copy this folder to
`~/.cursor/plugins/local/solarian` and reload the window.

## Layout

```text
.cursor-plugin/plugin.json    # Cursor manifest
.claude-plugin/plugin.json    # Claude Code manifest
skills/solarian/SKILL.md      # copied from solarian/templates/agent/bootstrap/
sync.py
EVALS.md                      # the discovery evals and their last results
```

`skills/` is generated. Edit `solarian/templates/agent/bootstrap/` instead, then run:

```bash
python integrations/agent-plugin/sync.py           # copy the skill, set the version
python integrations/agent-plugin/sync.py --check   # exit 1 if the plugin is out of date
```
