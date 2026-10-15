from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.sessions import _session_operation
from service.session_runtime import SessionBusyError, claim_session

router = APIRouter()
MAX_SUMMARIZATION_MESSAGES = 200
MAX_SUMMARIZATION_CHARS = 24000


def _get_agent_manager():
    from graph.agent import agent_manager
    return agent_manager


def _summary_input(messages: list[dict]) -> list[dict[str, str]]:
    # Never send metadata, raw audit events, attachments, or unbounded message
    # text to the summarizer. The full originals remain in the atomic archive.
    remaining = MAX_SUMMARIZATION_CHARS
    result: list[dict[str, str]] = []
    for index, item in enumerate(messages):
        if remaining <= 0:
            break
        # Reserve room for every selected message, rather than allowing one
        # large tool result to crowd all later user goals out of the summary.
        allowance = remaining // (len(messages) - index)
        content = str(item.get("content", "") or "")
        if len(content) > allowance:
            marker = "\n...[truncated]"
            content = content[:max(0, allowance - len(marker))] + marker[:allowance]
        result.append({"role": str(item.get("role", "assistant")), "content": content})
        remaining -= len(content)
    return result


@router.post("/sessions/{session_id}/compress")
async def compress_session(session_id: str) -> dict[str, int]:
    manager = _get_agent_manager()
    session_manager = manager.session_manager
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")
    _session_operation(session_manager.validate_session_id, session_id)
    try:
        with claim_session(session_id):
            record = _session_operation(session_manager.get_history, session_id)
            messages = record["messages"]
            if len(messages) < 4:
                raise HTTPException(status_code=400, detail="At least 4 messages are required")
            n_messages = min(MAX_SUMMARIZATION_MESSAGES, max(4, len(messages) // 2))
            summary_messages = _summary_input(messages[:n_messages])
            summary = await manager.summarize_history(summary_messages)
            if not isinstance(summary, str) or not summary.strip():
                raise HTTPException(status_code=502, detail="Unable to generate a usable history summary")
            result = _session_operation(
                session_manager.compress_history, session_id, summary, n_messages,
                expected_message_ids=[item["id"] for item in messages[:n_messages]],
                expected_compressed_context=record["compressed_context"],
            )
            await manager.invalidate_session_checkpoint(session_id)
            return result
    except SessionBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
