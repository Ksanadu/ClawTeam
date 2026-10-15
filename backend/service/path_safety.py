"""Pure path policy for the editable-document API (not a general file browser)."""
from __future__ import annotations

import re
from pathlib import Path, PurePosixPath


EDITABLE_DOCUMENTS = frozenset({
    "workspace/AGENTS.md",
    "workspace/IDENTITY.md",
    "workspace/SOUL.md",
    "workspace/USER.md",
    "memory/MEMORY.md",
    "memory_module_v1/long_term_memory/MEMORY.md",
    "skills/SKILLS_SNAPSHOT.md",
    *(f"workspace/roles/{role}.md" for role in (
        "MEDICAL_ONCOLOGIST", "PATHOLOGIST", "PHARMACIST", "PHYSICIAN",
        "RADIATION_ONCOLOGIST", "RADIOLOGIST", "SURGEON",
    )),
})
ALIASES = {
    "SKILLS_SNAPSHOT.md": "skills/SKILLS_SNAPSHOT.md",
    "MEMORY.md": "memory_module_v1/long_term_memory/MEMORY.md",
}
_SKILL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def normalize_editable_path(relative_path: str) -> str:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("An editable relative document path is required")
    normalized = relative_path.replace("\\", "/")
    # Check before PurePath/resolve can discard traversal, drive or empty segments.
    parts = normalized.split("/")
    if any(
        not part or part in {".", ".."} or ":" in part or "\x00" in part
        or part != part.strip() or part.endswith(".")
        for part in parts
    ):
        raise ValueError("Absolute paths, traversal and ambiguous path segments are not allowed")
    return ALIASES.get(normalized, normalized)


def _is_editable_document(relative_path: str) -> bool:
    if relative_path in EDITABLE_DOCUMENTS:
        return True
    parts = PurePosixPath(relative_path).parts
    # Dynamic UI entries are skill instructions, never their executable scripts,
    # dependencies, dotfiles, credentials, or arbitrary files under skills/.
    return (
        len(parts) == 3 and parts[0] == "skills" and parts[2] == "SKILL.md"
        and _SKILL_NAME.fullmatch(parts[1]) is not None
    )


def resolve_editable_path(base_dir: Path, relative_path: str) -> Path:
    normalized = normalize_editable_path(relative_path)
    if not _is_editable_document(normalized):
        raise ValueError("Path is not in the editable document whitelist")
    base = Path(base_dir).resolve()
    try:
        candidate = (base / normalized).resolve()
        canonical_relative = candidate.relative_to(base).as_posix()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("Path escapes the editable document whitelist") from exc
    # This second check matters: an allowed lexical prefix may resolve through a
    # symlink to code/configuration/secrets elsewhere *inside* the backend root.
    if not _is_editable_document(canonical_relative):
        raise ValueError("Resolved path is not in the editable document whitelist")
    return candidate
