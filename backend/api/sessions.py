from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from service.session_manager import SessionConflictError, SessionDataError
from service.session_runtime import SessionBusyError, claim_session

router = APIRouter()


def _get_agent_manager():
    from graph.agent import agent_manager
    return agent_manager


def _session_operation(operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return operation(*args, **kwargs)
    except SessionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SessionDataError as exc:
        raise HTTPException(status_code=500, detail="Stored session data is invalid") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Unable to access session storage") from exc


def _get_session_manager():
    session_manager = _get_agent_manager().session_manager
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")
    return session_manager


class CreateSessionRequest(BaseModel):
    title: str = "新会话"


class RenameSessionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)


class GenerateTitleRequest(BaseModel):
    message: str | None = None


@router.get("/sessions")
async def list_sessions() -> list[dict[str, Any]]:
    return _session_operation(_get_session_manager().list_sessions)


@router.post("/sessions")
async def create_session(payload: CreateSessionRequest) -> dict[str, Any]:
    return _session_operation(_get_session_manager().create_session, title=payload.title)


@router.put("/sessions/{session_id}")
async def rename_session(session_id: str, payload: RenameSessionRequest) -> dict[str, Any]:
    return _session_operation(_get_session_manager().rename_session, session_id, payload.title)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, bool]:
    session_manager = _get_session_manager()
    _session_operation(session_manager.validate_session_id, session_id)
    try:
        with claim_session(session_id):
            _session_operation(session_manager.delete_session, session_id)
            await _get_agent_manager().invalidate_session_checkpoint(session_id)
    except SessionBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True}


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str) -> dict[str, Any]:
    session_manager = _get_session_manager()
    base_dir = _get_agent_manager().base_dir
    if base_dir is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")
    messages = _session_operation(session_manager.load_session, session_id)
    from service.prompt_builder import build_system_prompt
    return {"system_prompt": build_system_prompt(base_dir), "messages": messages}


@router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str) -> dict[str, Any]:
    return _session_operation(_get_session_manager().get_history, session_id)


@router.post("/sessions/{session_id}/generate-title")
async def generate_title(session_id: str, payload: GenerateTitleRequest) -> dict[str, str]:
    session_manager = _get_session_manager()
    # Validate/existence-check before invoking an external title model, including
    # requests that supplied their own seed message.
    messages = _session_operation(session_manager.load_session, session_id)
    seed = payload.message or next(
        (item["content"] for item in messages if item.get("role") == "user"), ""
    )
    title = await _get_agent_manager().generate_title(seed or "新会话")
    _session_operation(session_manager.set_title, session_id, title)
    return {"session_id": session_id, "title": title}
