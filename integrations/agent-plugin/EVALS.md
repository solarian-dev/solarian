# Discovery evals

Four requests, each given to a fresh agent, check that coding agents find and use
the `solarian` CLI and the generated guides. Run them after changing the bootstrap
skill, the agent templates or the CLI's agent-facing output.

| # | Folder | Request | Passes when |
| --- | --- | --- | --- |
| 1 | empty | "Create a Solarian project called demo." | made with `solarian create-project`, `AGENTS.md` read |
| 2 | empty, bootstrap skill installed | same | as 1, with the skill as the first source |
| 3 | inside a new project | "Add a gripper module to this robot." | follows `AGENTS.md` and the `add-module` skill, `solarian doctor` clean |
| 4 | empty | "Create a Solarian plugin for a new servo." | made with `solarian create-plugin`, its `AGENTS.md` and `plugin-dev` skill followed |

## Running them

Use folders outside any Solarian checkout, so the repository's own `AGENTS.md`
is not in context, and a venv so the agents' installs stay out of your Python:

```bash
E=/tmp/solarian-evals && mkdir -p $E/{1,2,3,4}
python -m venv --system-site-packages $E/venv
printf '#!%s\nimport sys\nfrom solarian.cli.manage import main\nif __name__ == "__main__":\n    sys.exit(main())\n' \
  $E/venv/bin/python > $E/venv/bin/solarian && chmod +x $E/venv/bin/solarian
export PATH=$E/venv/bin:$PATH
(cd $E/3 && solarian create-project demo)

agent login    # once
(cd $E/1 && agent -p --force --trust --output-format json "Create a Solarian project called demo.")
(cd $E/2 && agent -p --force --trust --plugin-dir integrations/agent-plugin --output-format json "Create a Solarian project called demo.")
(cd $E/3/demo && agent -p --force --trust --output-format json "Add a gripper module to this robot.")
(cd $E/4 && agent -p --force --trust --output-format json "Create a Solarian plugin for a new servo.")
```

Run them one after another, or give each its own venv: an install in one changes
what the others see.

## Results, 2026-11-19

The `agent` CLI was not logged in, so these ran as Cursor subagents told to treat
the folder as their workspace. That is weaker than the procedure above: for
evals 2 and 3 the skill list and `AGENTS.md` were given in the prompt instead of
being loaded by the harness, and every agent's context still listed this
repository. Solarian was already installed in the venv, so eval 1 did not test
finding Solarian from nothing.

| # | Used the CLI | Guides read | Tool calls | Human needed |
| --- | --- | --- | --- | --- |
| 1 | yes: `--help` footer, then `create-project` | `AGENTS.md`, `NOTES.md`, `doctor` | ~22 | no |
| 2 | yes: the skill, then `create-project` | skill, `AGENTS.md`, `NOTES.md` | ~14 | no |
| 3 | yes: `install gripper1`, `docs`, `doctor` | `AGENTS.md`, `add-module`, 8 docs pages | ~50 | no |
| 4 | yes: `--help` footer, `docs show plugins`, `create-plugin` | `AGENTS.md`, `plugin-dev`, `NOTES.md` | ~39 | no |

Fixed after this run: the `add-module` skill now says what to do when a plugin
seeds no config and shows the per-actuator `port: ${actuator_port}`; the guides no
longer promise an `aperture_mm` field the docs do not explain; the plugin
`agent/AGENTS.md` stub no longer assumes a module; the `plugins` docs page shows
the agent files; the `create-plugin actuator` bus stub now has
`drive_to_encoder_center`, which `ActuatorBus` declares abstract; a new project's
`all` extra no longer requires `solarian-base` (not on PyPI), which moved to its
own `base` extra.

Open, outside the agent files:

- `solarian install` of a plugin that is already installed, from a source folder that no longer exists, reports success and changes nothing.
- `run-controller --help` shows `ui=external`, which Hydra rejects; it needs `+ui=external`.
