"""``solarian docs``: the bundled docs pages read as markdown."""

from __future__ import annotations

import pytest

from solarian.cli.docs_pages import docs_root, find_page, list_pages, mdx_to_markdown, normalize_slug, search

MDX = """\
import { Note } from '@/components/mdx'

export const metadata = {
  title: 'Skills',
  description: 'How skills work.',
}

export const sections = [
  { title: 'Config', id: 'config' },
]

# Skills {{ className: 'lead' }}

{/* a comment for editors */}
<Note>
  A skill reads <code>observations</code> and returns <strong>actions</strong>.
</Note>

<Properties>
  <Property name="frequency" type="float">
    Hz.
  </Property>
</Properties>

<Arm1Figure src="/a.png" />

<Button href="/docs/commands" arrow="right">Commands</Button>

```python {{ title: 'skill.py' }}
<Note>stays literal in code</Note>
export const x = 1
```
"""


@pytest.mark.parametrize("ref", [
    "skills", "/docs/skills", "docs/skills.md", "https://solarian.dev/docs/skills", "/docs/skills#config",
])
def test_every_spelling_of_a_page_names_one_slug(ref):
    assert normalize_slug(ref) == "skills"


@pytest.mark.parametrize("ref", ["", "/docs", "https://solarian.dev/docs.md"])
def test_the_docs_root_is_the_empty_slug(ref):
    assert normalize_slug(ref) == ""


def test_mdx_becomes_plain_markdown():
    md = mdx_to_markdown(MDX)

    assert "import" not in md.split("```")[0]
    assert "metadata" not in md and "sections" not in md
    assert md.startswith("# Skills\n")
    assert "a comment for editors" not in md
    assert "A skill reads `observations` and returns **actions**." in md
    assert "- **frequency** (float):" in md
    assert "Arm1Figure" not in md and "<Properties>" not in md
    assert "[Commands](/docs/commands)" in md
    assert '```python title="skill.py"\n<Note>stays literal in code</Note>\nexport const x = 1\n```' in md


def test_the_checkout_docs_are_found_and_readable():
    if docs_root() is None:
        pytest.skip("no docs next to this Solarian")
    slugs = {page.slug for page in list_pages()}
    assert {"", "agents", "skills", "commands"} <= slugs
    page = find_page("/docs/agents")
    assert page.title and page.description
    assert "solarian agent sync" in page.markdown()
    assert any(hit[0].slug == "agents" for hit in search("solarian agent install", limit=200))
