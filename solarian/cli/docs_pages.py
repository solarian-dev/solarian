"""The docs pages of the installed Solarian, read as markdown.

A wheel carries them under ``solarian/_docs`` (copied there by solarian_build);
a source checkout reads ``docs/`` beside the package. Both hold the MDX the
website renders, so the text matches the code that is installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import solarian

SITE_URL = "https://solarian.dev"


@dataclass(frozen=True)
class DocPage:
    slug: str  # path below /docs, "" for the docs root
    title: str
    description: str
    path: Path

    def markdown(self) -> str:
        return mdx_to_markdown(self.path.read_text(encoding="utf-8"))


def docs_root() -> Path | None:
    package = Path(solarian.__file__).parent
    for candidate in (package / "_docs", package.parent / "docs"):
        if (candidate / "page.mdx").is_file():
            return candidate
    return None


def _metadata(source: str, key: str) -> str:
    match = re.search(rf"{key}:\s*(['\"`])(.*?)\1", source, re.S)
    return " ".join(match.group(2).split()) if match else ""


def list_pages() -> list[DocPage]:
    root = docs_root()
    if root is None:
        return []
    pages = []
    for path in sorted(root.rglob("page.mdx")):
        source = path.read_text(encoding="utf-8")
        slug = path.parent.relative_to(root).as_posix()
        pages.append(DocPage(
            slug="" if slug == "." else slug,
            title=_metadata(source, "title"),
            description=_metadata(source, "description"),
            path=path,
        ))
    return pages


def normalize_slug(ref: str) -> str:
    """``skills``, ``/docs/skills``, ``docs/skills.md`` and the site URL all name one page."""
    ref = ref.strip().split("#")[0]
    ref = ref.removeprefix(SITE_URL).strip("/")
    ref = ref.removesuffix(".md").removesuffix("/page.mdx")
    if ref == "docs" or ref.startswith("docs/"):
        ref = ref[len("docs"):].strip("/")
    return ref


def find_page(ref: str) -> DocPage | None:
    slug = normalize_slug(ref)
    return next((page for page in list_pages() if page.slug == slug), None)


def _convert_prose(text: str) -> str:
    text = re.sub(r"\{/\*.*?\*/\}", "", text, flags=re.S)
    text = re.sub(r"</?>", "", text)
    text = re.sub(r"<[A-Z][A-Za-z0-9]*\b[^>]*?/>", "", text, flags=re.S)
    text = re.sub(
        r'<Property\s+name="([^"]*)"(?:\s+type="([^"]*)")?[^>]*>',
        lambda m: f"- **{m[1]}** ({m[2]}):" if m[2] else f"- **{m[1]}**:",
        text,
    )
    text = re.sub(r'<Button\s[^>]*href="([^"]*)"[^>]*>(.*?)</Button>',
                  lambda m: f"[{m[2].strip()}]({m[1]})", text, flags=re.S)
    text = re.sub(r"^[ \t]*</?[A-Z][A-Za-z0-9]*\b[^>]*>[ \t]*\n", "", text, flags=re.M)
    text = re.sub(r"</?[A-Z][A-Za-z0-9]*\b[^>]*>", "", text)
    text = re.sub(r'<a\s[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r"[\2](\1)", text, flags=re.S)
    text = re.sub(r"<code>(.*?)</code>", r"`\1`", text, flags=re.S)
    text = re.sub(r"<h[1-6]\b[^>]*>(.*?)</h[1-6]>", r"**\1**", text, flags=re.S)
    text = re.sub(r"</?(strong|b)>", "**", text)
    text = re.sub(r"</?(em|i)>", "*", text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"</?(div|span|p|figure|figcaption|video)\b[^>]*>", "", text)
    text = text.replace("{' '}", " ")
    text = re.sub(r"[ \t]*\{\{[^}]*\}\}[ \t]*$", "", text, flags=re.M)
    return re.sub(r"[ \t]+$", "", text, flags=re.M)


def mdx_to_markdown(source: str) -> str:
    """MDX as plain markdown: no exports or imports, components unwrapped, code untouched.

    The website's /docs/<page>.md route applies the same rules
    (solarian-docs ``src/lib/docs-markdown.ts``), except that it makes links absolute.
    """
    out: list[str] = []
    prose: list[str] = []
    fence: str | None = None
    skip_depth = 0

    def flush() -> None:
        if prose:
            out.append(_convert_prose("\n".join(prose)))
            prose.clear()

    for line in source.split("\n"):
        if fence is not None:
            out.append(line)
            if line.lstrip().startswith(fence):
                fence = None
            continue
        opening = re.match(r"^(\s*)(`{3,}|~{3,})(.*)$", line)
        if opening:
            flush()
            indent, marker, rest = opening.groups()
            lang = re.sub(r"\{\{[^}]*\}\}", "", rest).strip()
            title = re.search(r"title:\s*'([^']*)'", rest)
            out.append(f"{indent}{marker}{lang}" + (f' title="{title[1]}"' if title else ""))
            fence = marker
            continue
        if skip_depth > 0 or line.startswith("export "):
            skip_depth = max(0, skip_depth + len(re.findall(r"[{\[(]", line)) - len(re.findall(r"[}\])]", line)))
            continue
        if re.match(r"^import\s.+\sfrom\s+['\"]", line):
            continue
        prose.append(line)
    flush()
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def search(term: str, limit: int = 40) -> list[tuple[DocPage, int, str]]:
    """``(page, line number, line)`` for each markdown line containing *term*, case-insensitive."""
    needle = term.lower()
    hits = []
    for page in list_pages():
        for number, line in enumerate(page.markdown().split("\n"), start=1):
            if needle in line.lower():
                hits.append((page, number, line.strip()))
                if len(hits) >= limit:
                    return hits
    return hits
