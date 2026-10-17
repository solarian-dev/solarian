"""Where model weights live on this machine."""

from __future__ import annotations

import os
from pathlib import Path


def models_dir(*parts: str) -> Path:
    """``$SOLARIAN_MODELS_DIR/<parts>``, by default under ``~/.cache/solarian/models``.

    One root for every plugin's weights, so a host that runs ops for others
    (solarian_base) points it at its own store and nobody passes paths around.
    """
    root = os.environ.get("SOLARIAN_MODELS_DIR") or "~/.cache/solarian/models"
    return Path(root).expanduser().joinpath(*parts)
