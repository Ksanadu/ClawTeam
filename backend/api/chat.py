from __future__ import annotations

import asyncio
import json
import logging
from contextlib import aclosing
from copy import deepcopy
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from graph.context import build_request_context
from graph.agent import agent_manager
from memory_module_v2.service.config import get_memory_backend
from service.experiment import ExperimentMode
from service.session_runtime import SessionBusyError, claim_session
from starlette.background import BackgroundTask

logger = logging.getLogger(__name__)
router = APIRouter()
_background_tasks: set[asyncio.Task] = set()


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


class ChatRun:
    """One accumulator for SSE, JSON and durable history; correlate tools by ID."""

    def __init__(self, payload: ChatRequest):
        self.run_id = uuid4().hex
        self.segments: list[dict[str, Any]] = []
        self.current = _new_segment()
        self.tools: dict[str, dict[str, Any]] = {}
        self.content = ""
        self.error = ""
        self.finished = False
        self.metadata: dict[str, Any] = {
            "run_id": self.run_id,
            "experiment_mode": payload.experiment_mode,
            "status": "running",
            "failed_roles": [],
            "attachments": deepcopy(payload.attachments),
            "role_opinions": [],
            "routing": None,
            "retrievals": [],
            "progress": [],
            "guardian_blocked": None,
            "collaboration_metric": None,
            "events": [],
        }

    def _boundary(self):
        if self.current["content"] or self.current["tool_calls"]:
            self.segments.append(self.current)
        self.current = _new_segment()

    def apply(self, event: dict[str, Any]) -> dict[str, Any]:
        event = deepcopy(event)
        kind = event["type"]
        if kind in {"token", "synthesis_token"}:
            self.current["content"] += event.get("content", "")
        elif kind == "tool_start":
            call_id = str(event.get("tool_call_id") or uuid4().hex)
            event["tool_call_id"] = call_id
            tool = self.tools.get(call_id)
            if tool is None:
                tool = {"tool_call_id": call_id, "tool": event.get("tool", "tool"), "input": event.get("input", ""), "output": "", "status": "running"}
                self.current["tool_calls"].append(tool)
                self.tools[call_id] = tool
        elif kind == "tool_end":
            call_id = str(event.get("tool_call_id") or uuid4().hex)
            event["tool_call_id"] = call_id
            tool = self.tools.get(call_id)
            if tool is None:
                tool = {"tool_call_id": call_id, "tool": event.get("tool", "tool"), "input": ""}
                self.current["tool_calls"].append(tool)
                self.tools[call_id] = tool
            tool.update(output=event.get("output", ""), status=event.get("status", "success"))
        elif kind == "new_response":
            self._boundary()
        elif kind == "role_opinion":
            self.metadata["role_opinions"].append({key: value for key, value in event.items() if key != "type"})
        elif kind in {"routing", "guardian_blocked", "collaboration_metric"}:
            self.metadata[kind] = {key: value for key, value in event.items() if key != "type"}
        elif kind in {"retrieval", "retrieval_v2"}:
            self.metadata["retrievals"].append(deepcopy(event))
        elif kind == "progress":
            self.metadata["progress"].append({key: value for key, value in event.items() if key != "type"})
        elif kind == "role_error":
            self.metadata["failed_roles"].append({key: value for key, value in event.items() if key != "type"})
        elif kind == "error":
            self.error = str(event.get("error", "Run failed"))
            self.metadata["status"] = "failed"
        elif kind == "done":
            self.content = str(event.get("content", ""))
            self.metadata["status"] = event.get("status") or ("failed" if self.error else "success")
            self.metadata["failed_roles"] = event.get("failed_roles", self.metadata["failed_roles"])
            # The authoritative done answer replaces only the final response, not
            # earlier tool/narration segments. No raw SSE is ever returned as JSON.
            if self.content:
                self.current["content"] = self.content
            self._boundary()
            self.finished = True
            event.update(run_id=self.run_id, status=self.metadata["status"], failed_roles=self.metadata["failed_roles"])
        if kind not in {"token", "synthesis_token"}:
            self.metadata["events"].append(deepcopy(event))
        return event

    def interrupt(self, status: str):
        if self.finished:
            return
        self.metadata["status"] = status
        for tool in self.tools.values():
            if tool.get("status") == "running":
                tool["status"] = "cancelled" if status == "cancelled" else "failed"
        self.content = self.current["content"]
        self._boundary()

    def persist(self, manager, session_id: str):
        segments = self.segments or [_new_segment()]
        for index, segment in enumerate(segments):
            # Keep rich run-level panels on the final message only.
            metadata = deepcopy(self.metadata) if index == len(segments) - 1 else {
                key: self.metadata[key] for key in ("run_id", "experiment_mode", "status")
            }
            manager.save_message(session_id, "assistant", segment["content"], tool_calls=segment["tool_calls"] or None, metadata=metadata)

    def response(self):
        return {"content": self.content, **deepcopy(self.metadata), "segments": deepcopy(self.segments), **({"error": self.error} if self.error else {})}


@router.post("/chat")
async def chat(payload: ChatRequest):
    claim = claim_session(payload.session_id)
    try:
        claim.__enter__()
    except SessionBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    released = False

    def release():
        nonlocal released
        if not released:
            released = True
            claim.__exit__(None, None, None)

    try:
        response = await _chat_claimed(payload, release)
        if isinstance(response, StreamingResponse):
            response.background = BackgroundTask(release)
        else:
            release()
        return response
    except BaseException:
        release()
        raise


async def _chat_claimed(payload: ChatRequest, release):
    session_manager = agent_manager.session_manager
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")
    try:
        experiment_mode = ExperimentMode(payload.experiment_mode)
    except ValueError:
        raise HTTPException(status_code=422, detail="Unknown experiment_mode")

    try:
        history_record = session_manager.load_session_record(payload.session_id)
        # Load before persisting this turn, so the current input appears once.
        history_for_agent = session_manager.load_session_for_agent(payload.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("Could not read session")
        raise HTTPException(status_code=500, detail="Could not read session") from exc

    if experiment_mode.use_multi_agent:
        # Multi turns bypass the single-agent graph. Clear its cached state under
        # the same session claim so the next single turn rehydrates durable JSON
        # (including this consultation), instead of silently missing the turn.
        try:
            await agent_manager.invalidate_session_checkpoint(payload.session_id)
        except Exception as exc:
            logger.exception("Could not invalidate single-agent checkpoint")
            raise HTTPException(status_code=500, detail="Could not reset session context") from exc

    is_first_user_message = not history_record.get("compressed_context") and not any(
        message.get("role") == "user" for message in history_record.get("messages", [])
    )
    request_context = build_request_context(thread_id=payload.session_id)
    run = ChatRun(payload)
    try:
        session_manager.save_message(
            payload.session_id, "user", payload.message,
            metadata={"run_id": run.run_id, "experiment_mode": payload.experiment_mode, "attachments": payload.attachments},
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Could not save message") from exc

    def select_stream():
        if experiment_mode.use_multi_agent:
            return agent_manager.astream_multi_agent(
                payload.message, history_for_agent, context=request_context,
                experiment_mode=experiment_mode, attachments=payload.attachments,
            )
        return agent_manager.astream(payload.message, history_for_agent, context=request_context)

    async def events():
        persisted = False
        try:
            # Closing the stream on disconnect also closes/cancels specialist work.
            async with aclosing(select_stream()) as source:
                async for event in source:
                    event = run.apply(event)
                    if event["type"] == "done":
                        run.persist(session_manager, payload.session_id)
                        persisted = True
                    yield event
                    if event["type"] == "done":
                        break
            if not run.finished:
                raise RuntimeError("Agent ended without a final result")
            if is_first_user_message and run.metadata["status"] in {"success", "partial", "blocked"}:
                try:
                    title = await asyncio.wait_for(agent_manager.generate_title(payload.message), timeout=10)
                    session_manager.set_title(payload.session_id, title)
                    yield {"type": "title", "session_id": payload.session_id, "title": title}
                except Exception:
                    logger.warning("Title generation failed", exc_info=True)
            if get_memory_backend() == "v2" and run.metadata["status"] in {"success", "partial"}:
                task = asyncio.create_task(_distill_session_background(payload.session_id))
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)
        except (asyncio.CancelledError, GeneratorExit):
            run.interrupt("cancelled")
            raise
        except Exception as exc:
            # Never replay an agent invocation: a tool may have already committed
            # a side effect even when no token/event reached the client.
            logger.exception("Chat run failed")
            run.interrupt("failed")
            yield run.apply({"type": "error", "error": str(exc) or exc.__class__.__name__, "run_id": run.run_id})
        finally:
            try:
                if not persisted:
                    run.persist(session_manager, payload.session_id)
            except Exception:
                logger.exception("Could not persist failed/interrupted chat run")
            finally:
                release()

    if payload.stream:
        async def event_generator():
            async with aclosing(events()) as source:
                async for event in source:
                    yield _sse(event["type"], {key: value for key, value in event.items() if key != "type"})
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    async for _event in events():
        pass
    return JSONResponse(run.response(), status_code=502 if run.metadata["status"] == "failed" else 200)


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
                session_id[:12], result.exchanges_new, result.objects_created,
            )
        if result.errors:
            logger.warning("Distillation errors for session %s: %d", session_id[:12], len(result.errors))
    except Exception as exc:
        logger.warning("Background distillation failed for %s: %s", session_id[:12], exc)
