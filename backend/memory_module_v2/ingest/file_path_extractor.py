"""Extract file paths from exchange text using regex (no LLM)."""

from __future__ import annotations

import os
import re

_FILE_PATH_PATTERNS = [
    re.compile(
        r"""(?:^|[\s"'`(,])"""
        r"((?:[A-Za-z]:)?(?:[/\\])?"
        r"(?:[\w.@-]+[/\\])+[\w.@-]+\.\w{1,10})"
        r"""(?:[\s"'`),;:]|$)""",
    ),
    re.compile(
        r"\b([\w.-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|php|css|scss|html"
        r"|vue|svelte|json|yaml|yml|toml|md|sql|sh|bat|ps1|env|cfg|ini"
        r"|txt|xml|csv|proto))\b"
    ),
]


def extract_file_paths(text: str) -> list[str]:
    """Extract and deduplicate file paths from text."""
    if not text:
        return []

    found: set[str] = set()
    for pattern in _FILE_PATH_PATTERNS:
        for match in pattern.finditer(text):
            path = match.group(1).strip()
            path = _normalize_path(path)
            if path and _is_plausible_path(path):
                found.add(path)

    return sorted(found)


def _normalize_path(path: str) -> str:
    """Normalize path separators and remove trailing punctuation."""
    path = path.replace("\\", "/")
    path = path.rstrip(".,;:\"'`)")
    return path


def _is_plausible_path(path: str) -> bool:
    """Filter out false positives."""
    if len(path) < 3:
        return False
    if path.startswith("http://") or path.startswith("https://"):
        return False
    _, ext = os.path.splitext(path)
    if not ext:
        return False
    return True
