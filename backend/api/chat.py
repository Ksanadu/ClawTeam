from __future__ import annotations

import asyncio
import json
import logging
import traceback
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from graph.context import build_request_context
from graph.agent import agent_manager
from graph.checkpointer import reconnect_checkpointer_async
from memory_module_v2.service.config import get_memory_backend
from service.experiment import ExperimentMode

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str
    stream: bool = True
    experiment_mode: str = Field(
        default="single",
        description="实验模式: single | multi_no_memory | multi_memory | multi_full",
    )
    attachments: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional image attachments for multimodal specialist agents.",
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _new_segment() -> dict[str, Any]:
    return {"content": "", "tool_calls": []}


def _is_recoverable_checkpointer_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        exc.__class__.__name__ == "OperationalError"
        or "could not receive data from server" in text
        or "software caused connection abort" in text
    )


@router.post("/chat")
async def chat(payload: ChatRequest):
    session_manager = agent_manager.session_manager
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")

    history_record = session_manager.load_session_record(payload.session_id)
    is_first_user_message = not any(
        message.get("role") == "user"
        for message in history_record.get("messages", [])
    )
    request_context = build_request_context(thread_id=payload.session_id)
    # 对话历史由 checkpointer 管理，不再从 SessionManager 传入
    history_for_agent: list[dict[str, Any]] = []

    # 解析实验模式
    try:
        experiment_mode = ExperimentMode(payload.experiment_mode)
    except ValueError:
        experiment_mode = ExperimentMode.SINGLE_AGENT

    def _select_stream():
        """根据实验模式选择单 Agent 或多 Agent 流。"""
        if experiment_mode.use_multi_agent:
            return agent_manager.astream_multi_agent(
                payload.message,
                history_for_agent,
                context=request_context,
                experiment_mode=experiment_mode,
                attachments=payload.attachments,
            )
        return agent_manager.astream(
            payload.message, history_for_agent, context=request_context
        )

    async def event_generator():
        retried = False
        while True:
            segments: list[dict[str, Any]] = []
            current_segment = _new_segment()
            emitted_any_event = False
            try:
                async for event in _select_stream():
                    emitted_any_event = True
                    event_type = event["type"]

                    if event_type == "token":
                        current_segment["content"] += event.get("content", "")
                    elif event_type == "synthesis_token":
                        # 多 Agent 模式下，融合结论的流式 token
                        current_segment["content"] += event.get("content", "")
                    elif event_type in ("role_opinion", "routing", "guardian_blocked"):
                        # 多 Agent 新事件：透传给前端，不进入 segment content
                        pass
                    elif event_type == "tool_start":
                        current_segment["tool_calls"].append(
                            {
                                "tool": event.get("tool", "tool"),
                                "input": event.get("input", ""),
                                "output": "",
                            }
                        )
                    elif event_type == "tool_end":
                        if current_segment["tool_calls"]:
                            current_segment["tool_calls"][-1]["output"] = event.get("output", "")
                    elif event_type == "new_response":
                        if current_segment["content"].strip() or current_segment["tool_calls"]:
                            segments.append(current_segment)
                        current_segment = _new_segment()
                    elif event_type == "done":
                        if not current_segment["content"].strip() and event.get("content"):
                            current_segment["content"] = event["content"]
                        if current_segment["content"].strip() or current_segment["tool_calls"]:
                            segments.append(current_segment)

                        session_manager.save_message(payload.session_id, "user", payload.message)
                        for segment in segments:
                            session_manager.save_message(
                                payload.session_id,
                                "assistant",
                                segment["content"],
                                tool_calls=segment["tool_calls"] or None,
                            )

                    data = {key: value for key, value in event.items() if key != "type"}
                    yield _sse(event_type, data)

                    if event_type == "done":
                        if is_first_user_message:
                            title = await agent_manager.generate_title(payload.message)
                            session_manager.set_title(payload.session_id, title)
                            yield _sse(
                                "title",
                                {"session_id": payload.session_id, "title": title},
                            )

                        if get_memory_backend() == "v2":
                            asyncio.create_task(
                                _distill_session_background(payload.session_id)
                            )
                return
            except Exception as exc:
                if (
                    not retried
                    and not emitted_any_event
                    and _is_recoverable_checkpointer_error(exc)
                ):
                    retried = True
                    logger.warning(
                        "checkpointer connection dropped, reconnect and retry once: %s", exc
                    )
                    await reconnect_checkpointer_async()
                    continue

                print("[chat] error in event_generator", repr(exc))
                traceback.print_exc()
                yield _sse("error", {"error": str(exc)})
                return

    if payload.stream:
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    final_text = ""
    async for raw_event in event_generator():
        if raw_event.startswith("event: done"):
            final_text = raw_event
    return JSONResponse({"content": final_text})


async def _distill_session_background(session_id: str) -> None:
    """Run distillation in background after conversation ends.

    Idempotent: only processes new exchanges (deterministic exchange_id).
    Uses DISTILL_PROVIDER/MODEL if configured, otherwise falls back to main LLM.
    """
    try:
        from memory_module_v2.service.api import distill_session
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, distill_session, session_id)
        if result.exchanges_new > 0:
            logger.info(
                "Distilled session %s: %d new exchanges → %d objects",
                session_id[:12],
                result.exchanges_new,
                result.objects_created,
            )
        if result.errors:
            logger.warning(
                "Distillation errors for session %s: %d",
                session_id[:12],
                len(result.errors),
            )
    except Exception as exc:
        logger.warning("Background distillation failed for %s: %s", session_id[:12], exc)
