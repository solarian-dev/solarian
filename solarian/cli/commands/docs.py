"""``solarian docs``: the docs of the installed version, as markdown, offline."""

from __future__ import annotations

import argparse

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry

ACTIONS = ("list", "show", "search", "path")


@CommandRegistry.register("docs")
class DocsCommand(BaseCommand):
    category = "Project"
    uses_config = False
    setup_modules = ()
    help = "Read the docs of the installed Solarian version as markdown, offline"
    description = """
    Prints the Solarian docs that shipped with this install, so the text matches
    the code it describes. Coding agents read pages here rather than the website,
    which follows the latest source.

    list prints every page as its slug, title and summary. show prints one page
    as markdown; a slug, /docs/<slug>, or a solarian.dev URL all work. search
    prints the lines that contain a term. path prints the folder the pages are
    read from.
    """
    usage = "[list | show <page> | search <term> | path]"
    options = {
        "list": "Every page: slug, title and summary. The default.",
        "show": "Print one page as markdown, e.g. show skills or show tutorials/first-robot/arm.",
        "search": "Lines that contain a term, case-insensitive, with their page.",
        "path": "The folder the pages are read from.",
    }
    examples = (
        "solarian docs list",
        "solarian docs show skills",
        "solarian docs search MOVE_L",
    )

    def parse_args(self, argv: list[str]) -> list[str]:
        parser = argparse.ArgumentParser(prog="solarian docs")
        parser.add_argument("action", nargs="?", default="list", choices=ACTIONS)
        parser.add_argument("arg", nargs="*")
        self.ns = parser.parse_args(argv)
        return []

    def handle(self, cfg: DictConfig) -> None:
        from solarian.cli.docs_pages import docs_root, find_page, list_pages, search

        root = docs_root()
        if root is None:
            raise SystemExit("This install has no docs. Read them at https://solarian.dev/llms.txt")
        action, arg = self.ns.action, " ".join(self.ns.arg)

        if action == "path":
            print(root)
        elif action == "list":
            for page in list_pages():
                print(f"{page.slug or '(index)'}  {page.title}")
                if page.description:
                    print(f"    {page.description}")
        elif action == "show":
            if not arg:
                raise SystemExit("solarian docs show <page>; `solarian docs list` names them")
            page = find_page(arg)
            if page is None:
                raise SystemExit(f"No docs page {arg!r}; `solarian docs list` names them")
            print(page.markdown(), end="")
        else:
            if not arg:
                raise SystemExit("solarian docs search <term>")
            hits = search(arg)
            if not hits:
                raise SystemExit(f"No page mentions {arg!r}")
            for page, number, line in hits:
                print(f"{page.slug or '(index)'}:{number}: {line}")
