"""Read sessions from memory_module_v1/sessions/*.json and normalize."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..domain.models import NormalizedMessage

logger = logging.getLogger(__name__)

_SESSIONS_DIR: Path | None = None


def _default_sessions_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "memory_module_v1" / "sessions"


def get_sessions_dir() -> Path:
    global _SESSIONS_DIR
    if _SESSIONS_DIR is None:
        _SESSIONS_DIR = _default_sessions_dir()
    return _SESSIONS_DIR


def list_session_ids() -> list[str]:
    sessions_dir = get_sessions_dir()
    if not sessions_dir.exists():
        return []
    return [
        p.stem
        for p in sorted(sessions_dir.glob("*.json"))
        if not p.name.startswith(".")
    ]


def load_session_raw(session_id: str) -> dict[str, Any] | None:
    path = get_sessions_dir() / f"{session_id}.json"
    if not path.exists():
        logger.warning("Session file not found: %s", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read session %s: %s", session_id, exc)
        return None


def normalize_messages(session_data: dict[str, Any]) -> list[NormalizedMessage]:
    """Convert raw session messages to NormalizedMessage list with msg_index."""
    raw_messages = session_data.get("messages", [])
    result: list[NormalizedMessage] = []
    for idx, msg in enumerate(raw_messages):
        result.append(NormalizedMessage(
            msg_index=idx,
            role=msg.get("role", "unknown"),
            content=msg.get("content", ""),
            tool_calls=msg.get("tool_calls"),
        ))
    return result


def read_session(session_id: str) -> list[NormalizedMessage]:
    """Read a session and return normalized messages."""
    data = load_session_raw(session_id)
    if data is None:
        return []
    return normalize_messages(data)


def get_session_updated_at(session_id: str) -> float | None:
    data = load_session_raw(session_id)
    if data is None:
        return None
    return data.get("updated_at")
