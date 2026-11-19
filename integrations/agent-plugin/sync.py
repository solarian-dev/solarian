"""Copy the bootstrap skill and Solarian's version into this plugin.

The skill's source is solarian/templates/agent/bootstrap/, the same files
`solarian agent install` copies. Run this after changing them or bumping the
version in pyproject.toml; `--check` exits 1 when the plugin is out of date.

    python integrations/agent-plugin/sync.py [--check]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PLUGIN = Path(__file__).resolve().parent
REPO = PLUGIN.parents[1]
SOURCE = REPO / "solarian" / "templates" / "agent" / "bootstrap"
MANIFESTS = (PLUGIN / ".cursor-plugin" / "plugin.json", PLUGIN / ".claude-plugin" / "plugin.json")


def version() -> str:
    return re.search(r'^version\s*=\s*"([^"]+)"', (REPO / "pyproject.toml").read_text(), re.M)[1]


def desired() -> dict[Path, str]:
    files = {
        PLUGIN / "skills" / path.relative_to(SOURCE): path.read_text(encoding="utf-8")
        for path in sorted(SOURCE.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    for manifest in MANIFESTS:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["version"] = version()
        files[manifest] = json.dumps(data, indent=2) + "\n"
    return files


def stale() -> list[Path]:
    """Files that differ from what a sync writes, and skills the source no longer has."""
    want = desired()
    out = [path for path, text in want.items() if not path.is_file() or path.read_text(encoding="utf-8") != text]
    out += [path for path in (PLUGIN / "skills").rglob("*") if path.is_file() and path not in want]
    return out


def main() -> int:
    if "--check" in sys.argv[1:]:
        paths = stale()
        for path in paths:
            print(f"out of date: {path.relative_to(REPO)}")
        return 1 if paths else 0
    want = desired()
    for path in (PLUGIN / "skills").rglob("*"):
        if path.is_file() and path not in want:
            path.unlink()
    for path, text in want.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(f"Synced {len(want)} files into {PLUGIN.relative_to(REPO)} (solarian {version()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
