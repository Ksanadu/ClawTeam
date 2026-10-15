from __future__ import annotations

from functools import lru_cache
from typing import Any

import tiktoken
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.files import _resolve_path
from service.prompt_builder import build_system_prompt

router = APIRouter()


def _get_agent_manager():
    from graph.agent import agent_manager
    return agent_manager


@lru_cache(maxsize=1)
def _encoder():
    return tiktoken.get_encoding("cl100k_base")


class FileTokensRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)


def _count_tokens(text: str) -> int:
    return len(_encoder().encode(text or ""))


@router.get("/tokens/session/{session_id}")
async def session_tokens(session_id: str) -> dict[str, int]:
    agent_manager = _get_agent_manager()
    session_manager = agent_manager.session_manager
    if session_manager is None or agent_manager.base_dir is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")

    try:
        record = session_manager.get_history(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    system_prompt = build_system_prompt(agent_manager.base_dir)
    message_text = []
    for item in record.get("messages", []):
        message_text.append(str(item.get("content", "")))
        for tool_call in item.get("tool_calls", []) or []:
            message_text.append(str(tool_call))

    system_tokens = _count_tokens(system_prompt)
    message_tokens = _count_tokens("\n".join(message_text))
    return {
        "system_tokens": system_tokens,
        "message_tokens": message_tokens,
        "total_tokens": system_tokens + message_tokens,
    }


@router.post("/tokens/files")
async def file_tokens(payload: FileTokensRequest) -> dict[str, Any]:
    if _get_agent_manager().base_dir is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")

    files: list[dict[str, Any]] = []
    total = 0
    for relative_path in payload.paths:
        path = _resolve_path(relative_path)
        if not path.exists() or path.is_dir():
            continue
        count = _count_tokens(path.read_text(encoding="utf-8"))
        total += count
        files.append({"path": relative_path, "tokens": count})

    return {"files": files, "total_tokens": total}
