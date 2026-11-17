---
name: write-docs
description: Conventions for writing and editing Solarian documentation pages (MDX under solarian/docs, rendered by solarian-docs). Use when writing, editing, or reviewing any docs page or tutorial, especially one that tells the reader to change a file in their project or adds an external link.
---

# Writing Solarian docs

Docs live in `solarian/docs/**/page.mdx`. `solarian-docs/src/app/(docs)/docs` is a
symlink to that directory, so edit only under `solarian/docs`. Navigation entries are
in `solarian-docs/src/components/Navigation.tsx`; update them when a page is added,
removed, or renamed.

## Title every snippet that edits a file

When a snippet is something the reader puts into a specific file, name that file in
the code fence title. The site renders it as a header over the code block:

````mdx
```python {{ title: 'my_robot/controller.py' }}
from solarian_move import MoveSkill
```
````

- Use the path relative to the project root: `my_robot/robot.py`,
  `conf/modules/arm.yaml`, `conf/skills/move.yaml`.
- This applies to every language: `python`, `yaml`, `toml`, `xml`.
- Do not also write "In `my_robot/robot.py`:" in the prose before the block. The title
  says it. The prose says why the change is needed.
- Terminal commands that need their own terminal use the same mechanism:
  `` ```bash {{ title: 'Terminal 1' }} ``. A single one-off command needs no title.
- Snippets that are not a file edit (a generic API example on a reference page, sample
  output) have no title.

## Show where in the file the lines go

A bare indented fragment does not tell the reader where to put it. Include the
enclosing class or function, and elide the parts that are not changing with a
`# ...` comment that names what is there:

```python {{ title: 'my_robot/action.py' }}
class MyArmAction(ModuleAction):
    # ... the six joint fields ...
    x = ActuatorField(MyRobotRobot.arm.x.pos)
```

Imports that the edit needs go at the top of the same block, separated from the
class by blank lines as in the real file, rather than in a separate block.

When one edit touches two places in the same file (a `tools` entry and a key
branch), put both in one titled block with `# ...` between them.

## External links open in a new tab

Every link that leaves the docs site (any `http://` or `https://` URL) opens in a new
browser tab, so the reader does not lose their place in a tutorial. Markdown link
syntax cannot set a target, so write the anchor as JSX:

```mdx
The project site is <a href="https://umi-gripper.github.io/" target="_blank" rel="noreferrer">umi-gripper.github.io</a>.
```

- Always pair `target="_blank"` with `rel="noreferrer"`.
- Internal links (`/docs/...`, `#anchor`) stay as markdown links and open in the
  same tab.
- When editing a page, convert any external markdown link `[text](https://...)` you
  touch to this form.

## Keep the docs and the example project in sync

The tutorials are followed against `my_robot`. When a page changes what the reader
types, check that the code in the page matches what actually runs, and that every
file the reader must touch is shown. A missing step (an import, a schema field, a
tool entry) is the most common tutorial bug.
