"""``KEY=VALUE`` files (``.env``) into the process environment."""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path) -> list[str]:
    """Set each ``KEY=VALUE`` line of *path* that the environment does not already set.

    The shell wins, so ``OPENAI_API_KEY=... solarian run-controller`` overrides
    the file. Returns the keys it set.
    """
    loaded = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")
        loaded.append(key)
    return loaded
