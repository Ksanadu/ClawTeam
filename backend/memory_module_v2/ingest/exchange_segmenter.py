"""Segment normalized messages into exchanges based on msg_index ply ranges."""

from __future__ import annotations

import hashlib
import logging

from ..domain.models import Exchange, NormalizedMessage
from .text_cleaner import clean_text

logger = logging.getLogger(__name__)


def make_exchange_id(session_id: str, ply_start: int, ply_end: int) -> str:
    """Deterministic exchange_id: sha1(session_id:ply_start:ply_end)."""
    raw = f"{session_id}:{ply_start}:{ply_end}"
    return hashlib.sha1(raw.encode()).hexdigest()


def _is_substantive_assistant(content: str, *, min_chars: int = 80) -> bool:
    """Determine if an assistant message is a 'substantive reply'."""
    stripped = clean_text(content).strip()
    return len(stripped) >= min_chars


def _render_verbatim(
    messages: list[NormalizedMessage],
    *,
    min_assistant_chars_for_snippet: int,
) -> tuple[str, str]:
    """Render messages into verbatim_text (full) and verbatim_snippet (substantive-only)."""
    text_parts: list[str] = []
    snippet_parts: list[str] = []

    for msg in messages:
        prefix = msg.role.upper()
        content = msg.content or ""
        text_parts.append(f"[{prefix}] {content}")

        if msg.role == "user":
            snippet_parts.append(f"USER: {content}")
        elif msg.role == "assistant" and _is_substantive_assistant(
            content,
            min_chars=min_assistant_chars_for_snippet,
        ):
            cleaned = clean_text(content)
            snippet_parts.append(f"ASSISTANT: {cleaned}")

    verbatim_text = "\n\n".join(text_parts)
    verbatim_snippet = "\n\n".join(snippet_parts)
    return verbatim_text, verbatim_snippet


def segment_exchanges(
    session_id: str,
    messages: list[NormalizedMessage],
    *,
    min_exchange_chars: int = 100,
    max_ply_len: int = 20,
    min_assistant_chars: int = 80,
) -> list[Exchange]:
    """Segment messages into exchanges following the ply-based protocol."""
    if not messages:
        return []

    raw_ranges: list[tuple[int, int, bool]] = []
    start: int | None = None
    has_substantive = False

    for msg in messages:
        idx = msg.msg_index
        if msg.role == "user":
            if start is not None and has_substantive:
                raw_ranges.append((start, prev_idx, True))
                start = idx
                has_substantive = False
            elif start is None:
                start = idx
        if msg.role == "assistant":
            if start is None:
                start = idx
            if _is_substantive_assistant(msg.content, min_chars=min_assistant_chars):
                has_substantive = True
        prev_idx = idx

    if start is not None:
        raw_ranges.append((start, messages[-1].msg_index, has_substantive))

    exchanges: list[Exchange] = []
    msg_by_idx = {m.msg_index: m for m in messages}

    for ply_start, ply_end, substantive in raw_ranges:
        length = ply_end - ply_start + 1
        if length > max_ply_len:
            for chunk_start in range(ply_start, ply_end + 1, max_ply_len):
                chunk_end = min(chunk_start + max_ply_len - 1, ply_end)
                _add_exchange(
                    exchanges, session_id, chunk_start, chunk_end,
                    msg_by_idx, min_exchange_chars, substantive, min_assistant_chars,
                )
        else:
            _add_exchange(
                exchanges, session_id, ply_start, ply_end,
                msg_by_idx, min_exchange_chars, substantive, min_assistant_chars,
            )

    return exchanges


def _add_exchange(
    exchanges: list[Exchange],
    session_id: str,
    ply_start: int,
    ply_end: int,
    msg_by_idx: dict[int, NormalizedMessage],
    min_exchange_chars: int,
    has_substantive: bool,
    min_assistant_chars: int,
) -> None:
    msgs = [msg_by_idx[i] for i in range(ply_start, ply_end + 1) if i in msg_by_idx]
    if not msgs:
        return

    verbatim_text, verbatim_snippet = _render_verbatim(
        msgs,
        min_assistant_chars_for_snippet=min_assistant_chars,
    )
    total_chars = sum(len(m.content or "") for m in msgs)

    if total_chars < min_exchange_chars:
        return

    exchange_id = make_exchange_id(session_id, ply_start, ply_end)
    exchanges.append(Exchange(
        exchange_id=exchange_id,
        session_id=session_id,
        ply_start=ply_start,
        ply_end=ply_end,
        messages=msgs,
        verbatim_text=verbatim_text,
        verbatim_snippet=verbatim_snippet,
        message_count=len(msgs),
        has_substantive_assistant=has_substantive,
    ))
