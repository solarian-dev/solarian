"""Render what ``solarian --help`` and ``solarian <command> --help`` print.

Commands are driven by Hydra ``key=value`` overrides rather than flags, so
argparse can only describe the global options. What each command accepts is
declared on its :class:`~solarian.cli.base.BaseCommand` subclass and formatted
here.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solarian.cli.base import BaseCommand

#: Root-parser flags as ``(flags, argparse kwargs)``. They live here because the
#: root parser and every command's help both need them: each one changes which
#: conf directory, deployment profile or settings module a command reads.
GLOBAL_OPTIONS: tuple[tuple[tuple[str, ...], dict], ...] = (
    (
        ("--conf",),
        {
            "default": "conf",
            "metavar": "DIR",
            "help": "Conf directory holding the yaml configs. DIR/NAME composes "
                    "NAME.yaml instead of the command's own config (default: conf).",
        },
    ),
    (
        ("--deployment", "-d"),
        {
            "default": None,
            "metavar": "NAME",
            "help": "Deployment profile in conf/deployment/ (relay address, device "
                    "ports, keys). Defaults to $SOLARIAN_DEPLOYMENT, then localhost.",
        },
    ),
    (
        ("--settings",),
        {
            "default": None,
            "metavar": "PATH",
            "help": "Path to the project's settings.py (auto-discovered if not given).",
        },
    ),
)

#: Width the help is wrapped to.
WIDTH = 88

_INDENT = "  "
#: Widest term that still shares its line with a description; longer ones get a
#: line of their own rather than pushing every description to the right.
_TERM_WIDTH = 22


def _prose(text: str, indent: str = "") -> list[str]:
    """Wrap paragraphs to :data:`WIDTH`, leaving indented or bulleted blocks as written."""
    out: list[str] = []
    for block in textwrap.dedent(text).strip("\n").split("\n\n"):
        lines = block.splitlines()
        if any(line[:1].isspace() or line.lstrip()[:1] in "-*$" for line in lines):
            out += [indent + line if line.strip() else "" for line in lines]
        else:
            out += textwrap.wrap(
                " ".join(block.split()),
                width=WIDTH,
                initial_indent=indent,
                subsequent_indent=indent,
                # `calibrate-umi-camera` is one word, not two.
                break_on_hyphens=False,
            )
        out.append("")
    return out


def _usage(name: str, spec: str) -> list[str]:
    """Lay out the usage line, continuing under the command name when it is long.

    Wraps between arguments only, so ``[--keys-file PATH]`` stays in one piece.
    """
    prefix = f"usage: solarian {name}"
    indent = " " * (len(prefix) + 1)
    lines: list[str] = []
    line = prefix
    for group in re.findall(r"\[[^\]]*\]|\S+", spec):
        if len(line) + 1 + len(group) > WIDTH:
            lines.append(line)
            line = indent + group
        else:
            line = f"{line} {group}"
    lines.append(line)
    return lines


def _table(title: str, rows: Mapping[str, str]) -> list[str]:
    """Format ``term -> description`` pairs as an aligned two-column block."""
    if not rows:
        return []
    width = min(max(len(term) for term in rows), _TERM_WIDTH)
    gutter = len(_INDENT) + width + 2
    out = [f"{title}:"]
    for term, description in rows.items():
        wrapped = textwrap.wrap(
            description, width=WIDTH - gutter, break_on_hyphens=False
        ) or [""]
        if len(term) > width:
            out.append(_INDENT + term)
        else:
            out.append(_INDENT + term.ljust(width) + "  " + wrapped.pop(0))
        out += [" " * gutter + line for line in wrapped]
    out.append("")
    return out


def format_command_help(
    name: str,
    cmd: BaseCommand,
    *,
    config_name: str | None,
    uses_config: bool,
) -> str:
    """Build the help page for one command.

    ``uses_config`` is False for the commands that run before a project exists,
    or straight against hardware, and so ignore the conf directory entirely.
    """
    out = [*_usage(name, cmd.usage), ""]
    if cmd.help:
        out += _prose(cmd.help)
    if cmd.description:
        out += _prose(cmd.description)
    out += _table("arguments", cmd.options)
    out += _table("config overrides (key=value)", cmd.params)

    if uses_config and config_name:
        out.append("config:")
        out += _prose(
            f"conf/{config_name}.yaml, merged with the deployment profile. Any key "
            "either file declares can be overridden as key=value as well.",
            indent=_INDENT,
        )

    options = {"-h, --help": "Show this help and exit."}
    if uses_config:
        for flags, kwargs in GLOBAL_OPTIONS:
            term = ", ".join(flags)
            options[f"{term} {kwargs['metavar']}"] = kwargs["help"]
    out += _table("options", options)

    if cmd.examples:
        out.append("examples:")
        out += [_INDENT + line for line in cmd.examples]
        out.append("")
    return "\n".join(out).rstrip() + "\n"
