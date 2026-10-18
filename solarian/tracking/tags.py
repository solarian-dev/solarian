"""Tag hygiene shared by every backend."""

from __future__ import annotations

from typing import Iterable

#: wandb rejects longer tags with HTTP 400; Base keeps the same limit so a run
#: looks the same in both.
MAX_TAG_LENGTH = 64


def clean_tags(tags: Iterable[object] | None) -> list[str]:
    """Drop empty tags, truncate long ones, keep the first of duplicates.

    Configs build tags with interpolations that resolve to ``""`` or ``None``
    when a field does not apply (``real_size:`` without real data).
    """
    out: list[str] = []
    for tag in tags or ():
        if tag is None:
            continue
        text = str(tag).strip()
        if not text or text in ("None", "null"):
            continue
        if len(text) > MAX_TAG_LENGTH:
            print(f"[tracking] truncating tag to {MAX_TAG_LENGTH} chars: {text!r} -> {text[:MAX_TAG_LENGTH]!r}")
            text = text[:MAX_TAG_LENGTH]
        if text not in out:
            out.append(text)
    return out
