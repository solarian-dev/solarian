"""Where model weights live on this machine, and fetching the ones that are missing."""

from __future__ import annotations

import logging
import os
import shutil
import urllib.request
from pathlib import Path

from solarian.core.conf_paths import conf_dir

logger = logging.getLogger(__name__)


def models_dir(*parts: str) -> Path:
    """Where a service keeps and downloads its weights: ``<root>/<parts>``.

    The root is ``$SOLARIAN_MODELS_DIR`` when set, so a host that runs ops for
    others (solarian_base) points every plugin at its own store. Otherwise it
    is the active project's ``models/``, next to its ``conf/``, and
    ``~/.cache/solarian/models`` outside a project.
    """
    root = os.environ.get("SOLARIAN_MODELS_DIR")
    if root:
        base = Path(root).expanduser()
    elif conf_dir().is_dir():
        base = conf_dir().parent / "models"
    else:
        base = Path("~/.cache/solarian/models").expanduser()
    return base.joinpath(*parts)


def download_missing(path: str | os.PathLike, url: str, user_agent: str = "solarian") -> Path:
    """``path``, downloading ``url`` there first when the file is absent."""
    path = Path(path).expanduser()
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # A dropped connection must not leave a file the next start would load.
    partial = path.with_name(path.name + ".partial")
    logger.info("Weights %s are missing; downloading %s", path, url)
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return path
