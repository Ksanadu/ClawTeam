"""MVP text cleaner: identify and optionally strip tool output / terminal noise."""

from __future__ import annotations

import re

_NOISE_PREFIXES = [
    "CategoryInfo",
    "FullyQualifiedErrorId",
    "所在位置",
    "PSMessageDetails",
    "+ ~",
]

_NOISE_PATTERNS = [
    re.compile(r"^\s*\+\s*~+\s*$"),
    re.compile(r"^.*CommandNotFoundException.*$"),
    re.compile(r"^.*ParameterBindingException.*$"),
    re.compile(r"^\s*at\s+\S+\s+in\s+\S+:line\s+\d+", re.IGNORECASE),
]

_TOOL_OUTPUT_BLOCK = re.compile(
    r"^(Exit code: \d+|Command output:|```\n).*?```$",
    re.MULTILINE | re.DOTALL,
)


def clean_text(text: str, *, strip_noise: bool = True) -> str:
    """Remove obvious terminal/tool noise from text.

    When strip_noise=False, returns text unchanged (passthrough).
    """
    if not strip_noise or not text:
        return text

    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in _NOISE_PREFIXES):
            continue
        if any(pat.match(stripped) for pat in _NOISE_PATTERNS):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def is_tool_output(text: str) -> bool:
    """Heuristic: check if text looks like pure tool/terminal output."""
    if not text or not text.strip():
        return True
    stripped = text.strip()
    if stripped.startswith("Exit code:"):
        return True
    if stripped.startswith("```") and stripped.endswith("```"):
        return True
    noise_line_count = 0
    lines = stripped.splitlines()
    for line in lines:
        s = line.strip()
        if any(s.startswith(p) for p in _NOISE_PREFIXES):
            noise_line_count += 1
        elif any(pat.match(s) for pat in _NOISE_PATTERNS):
            noise_line_count += 1
    if lines and noise_line_count / len(lines) > 0.6:
        return True
    return False
