from __future__ import annotations

import asyncio
import importlib
import json
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from graph import agent as agent_module, checkpointer
from graph.agent import AgentManager
from graph.complexity_assessor import CaseComplexity, ComplexityDecision
from graph.coordinator import Coordinator
from graph.roles.base_role import RoleOpinion, RoleType
from service.session_manager import SessionManager
from service.session_runtime import SessionBusyError, claim_session

chat_module = importlib.import_module("api.chat")


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    manager = SessionManager(tmp_path)
    session_id = manager.create_session()["id"]
    # A prior turn ensures ordinary followups don't trigger a live title model.
    manager.save_message(session_id, "user", "prior question")
    manager.save_message(session_id, "assistant", "prior answer")
    monkeypatch.setattr(chat_module.agent_manager, "session_manager", manager)
    monkeypatch.setattr(chat_module, "get_memory_backend", lambda: "none")
    return manager, session_id


async def body_events(response):
    result = []
    async for raw in response.body_iterator:
        if isinstance(raw, bytes):
            raw = raw.decode()
        event, data, _ = raw.split("\n", 2)
        result.append({"type": event.removeprefix("event: "), **json.loads(data.removeprefix("data: "))})
    return result


def rich_events():
    return [
        {"type": "progress", "stage": "round1", "status": "running", "label": "working"},
        {"type": "routing", "roles": ["surgeon"], "reason": "case"},
        {"type": "retrieval_v2", "query": "q", "context": "memory"},
        {"type": "role_opinion", "role": "surgeon", "role_label": "surgeon", "content": "expert", "round": 1, "evidence": ["report"]},
        {"type": "token", "content": "checking"},
        {"type": "tool_start", "tool_call_id": "a", "tool": "same", "input": "one"},
        {"type": "tool_start", "tool_call_id": "b", "tool": "same", "input": "two"},
        {"type": "tool_end", "tool_call_id": "b", "tool": "same", "output": "second"},
        {"type": "tool_end", "tool_call_id": "a", "tool": "same", "output": "first"},
        {"type": "new_response"},
        {"type": "collaboration_metric", "revision_rate": 0.5, "disagreement_count": 1},
        {"type": "role_error", "role": "pathologist", "round": 1, "stage": "round1", "error": "timeout"},
        {"type": "synthesis_token", "content": "final answer"},
        {"type": "done", "content": "final answer", "status": "partial"},
    ]


@pytest.mark.parametrize("streaming", [True, False])
async def test_chat_roundtrip_rich_metadata_and_tool_ids(sessions, monkeypatch, streaming):
    manager, session_id = sessions
    history_seen = []
    attachment = {"name": "scan.png", "url": "data:image/png;base64,AA", "content_type": "image/png"}

    async def source(message, history, **kwargs):
        history_seen.extend(history)
        assert kwargs["attachments"] == [attachment]
        for event in rich_events():
            yield event

    monkeypatch.setattr(chat_module.agent_manager, "astream_multi_agent", source)
    app = FastAPI()
    app.include_router(chat_module.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/chat", json={"session_id": session_id, "message": "follow-up", "stream": streaming, "experiment_mode": "multi_memory", "attachments": [attachment]})
    assert response.status_code == 200
    if streaming:
        assert "event: done\n" in response.text
        assert '"tool_call_id": "a"' in response.text
    else:
        body = response.json()
        assert body["content"] == "final answer"
        assert body["status"] == "partial"
        assert body["routing"]["roles"] == ["surgeon"]
        assert "event: done" not in body["content"]
    assert [item["content"] for item in history_seen] == ["prior question", "prior answer"]
    messages = manager.load_session(session_id)
    assert len(messages) == 5
    assert messages[2]["role"] == "user" and messages[2]["attachments"] == [attachment]
    tools = messages[3]["tool_calls"]
    assert [(tool["tool_call_id"], tool["output"]) for tool in tools] == [("a", "first"), ("b", "second")]
    final = messages[-1]
    assert final["content"] == "final answer" and final["status"] == "partial"
    assert final["attachments"] == [attachment]
    assert final["routing"]["roles"] == ["surgeon"]
    assert final["role_opinions"][0]["content"] == "expert"
    assert final["retrievals"][0]["context"] == "memory"
    assert final["progress"][0]["stage"] == "round1"
    assert final["collaboration_metric"]["revision_rate"] == 0.5
    assert final["failed_roles"][0]["role"] == "pathologist"
    assert final["events"][-1]["type"] == "done"
    assert len({message["id"] for message in messages}) == len(messages)
    assert messages[2]["run_id"] == messages[3]["run_id"] == final["run_id"]


async def test_no_replay_after_side_effect_even_before_first_event(sessions, monkeypatch):
    manager, session_id = sessions
    calls = []

    class OperationalError(RuntimeError):
        pass

    async def source(*args, **kwargs):
        calls.append("side-effect")
        raise OperationalError("connection dropped after write")
        yield

    monkeypatch.setattr(chat_module.agent_manager, "astream", source)
    response = await chat_module.chat(chat_module.ChatRequest(session_id=session_id, message="write", stream=False))
    body = json.loads(response.body)
    assert response.status_code == 502
    assert calls == ["side-effect"]
    assert body["status"] == "failed" and body["error"]
    messages = manager.load_session(session_id)
    assert messages[-2]["role"] == "user" and messages[-2]["content"] == "write"
    assert messages[-1]["status"] == "failed"
    # The claim was released after failure.
    with claim_session(session_id):
        pass


async def test_disconnect_closes_agent_persists_partial_and_releases_claim(sessions, monkeypatch):
    manager, session_id = sessions
    closed = asyncio.Event()

    async def source(*args, **kwargs):
        try:
            yield {"type": "token", "content": "partial"}
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(chat_module.agent_manager, "astream", source)
    response = await chat_module.chat(chat_module.ChatRequest(session_id=session_id, message="followup"))
    with pytest.raises(SessionBusyError):
        with claim_session(session_id):
            pass
    await anext(response.body_iterator)
    await response.body_iterator.aclose()
    assert closed.is_set()
    assert manager.load_session(session_id)[-1]["status"] == "cancelled"
    assert manager.load_session(session_id)[-1]["content"] == "partial"
    with claim_session(session_id):
        pass


async def test_cancellation_during_pending_agent_await_joins_cleanup(sessions, monkeypatch):
    manager, session_id = sessions
    entered, closed = asyncio.Event(), asyncio.Event()

    async def source(*args, **kwargs):
        try:
            entered.set()
            await asyncio.Event().wait()
            yield
        finally:
            closed.set()

    monkeypatch.setattr(chat_module.agent_manager, "astream", source)
    response = await chat_module.chat(chat_module.ChatRequest(session_id=session_id, message="followup"))
    task = asyncio.create_task(anext(response.body_iterator))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    assert manager.load_session(session_id)[-1]["status"] == "cancelled"
    with claim_session(session_id):
        pass


async def test_active_chat_rejects_concurrent_chat(sessions, monkeypatch):
    manager, session_id = sessions

    async def source(*args, **kwargs):
        yield {"type": "done", "content": "ok"}

    monkeypatch.setattr(chat_module.agent_manager, "astream", source)
    response = await chat_module.chat(chat_module.ChatRequest(session_id=session_id, message="first"))
    with pytest.raises(HTTPException) as error:
        await chat_module.chat(chat_module.ChatRequest(session_id=session_id, message="second"))
    assert error.value.status_code == 409
    await body_events(response)
    assert [message["content"] for message in manager.load_session(session_id) if message["role"] == "user"] == ["prior question", "first"]


@pytest.mark.parametrize("session_id,status", [("../invalid", 400), (uuid4().hex, 404)])
async def test_invalid_missing_session_fail_before_stream(sessions, session_id, status):
    with pytest.raises(HTTPException) as error:
        await chat_module.chat(chat_module.ChatRequest(session_id=session_id, message="hello"))
    assert error.value.status_code == status


async def test_mixed_single_multi_single_rehydrates_durable_context(sessions, tmp_path, monkeypatch):
    session_manager, session_id = sessions
    manager = AgentManager()
    manager.base_dir = tmp_path
    manager.session_manager = session_manager
    manager.coordinator = Coordinator(tmp_path)
    saver = InMemorySaver()
    monkeypatch.setattr(checkpointer, "_default_saver", saver)
    monkeypatch.setattr(chat_module, "agent_manager", manager)
    monkeypatch.setattr(agent_module, "get_memory_backend", lambda: "none")
    monkeypatch.setattr(agent_module.experiment_logger, "log", lambda _: None)
    observed = []

    async def respond(state):
        texts = [message.content for message in state["messages"]]
        observed.append(texts)
        return {"messages": [AIMessage(content=f"answer to {texts[-1]}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", respond)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    graph = builder.compile(checkpointer=saver)
    monkeypatch.setattr(manager, "_build_agent", lambda: graph)
    config = {"configurable": {"thread_id": session_id}}

    class Expert:
        async def aconsult_round1(self, case, memory_context, attachments=None):
            assert "single-one" in case and "answer to single-one" in case
            # Actual single checkpoint was cleared while the chat claim is held.
            assert not (await graph.aget_state(config)).values
            with pytest.raises(SessionBusyError):
                with claim_session(session_id):
                    pass
            return RoleOpinion(RoleType.SURGEON, "surgeon", "multi expert")

    async def assess(*args, **kwargs):
        return ComplexityDecision(CaseComplexity.MODERATE, "test", "manual")

    async def aggregate(*args, **kwargs):
        return "multi answer"

    manager.coordinator._roles = {RoleType.SURGEON: Expert()}
    monkeypatch.setattr(agent_module, "assess_complexity", assess)
    monkeypatch.setattr(manager.coordinator, "_aggregate", aggregate)
    for mode, message in [("single", "single-one"), ("multi_no_memory", "multi-two"), ("single", "single-three")]:
        response = await chat_module.chat(chat_module.ChatRequest(
            session_id=session_id, message=message, experiment_mode=mode, stream=False,
        ))
        assert response.status_code == 200
    assert observed[0] == ["prior question", "prior answer", "single-one"]
    assert observed[1] == [
        "prior question", "prior answer", "single-one", "answer to single-one",
        "multi-two", "multi answer", "single-three",
    ]
    assert [message.content for message in (await graph.aget_state(config)).values["messages"]] == observed[1] + ["answer to single-three"]


async def test_multi_checkpoint_invalidation_failure_aborts_before_save(sessions, monkeypatch):
    manager, session_id = sessions

    async def fail(_session_id):
        raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr(chat_module.agent_manager, "invalidate_session_checkpoint", fail)
    with pytest.raises(HTTPException) as error:
        await chat_module.chat(chat_module.ChatRequest(
            session_id=session_id, message="multi", experiment_mode="multi_no_memory", stream=False,
        ))
    assert error.value.status_code == 500
    assert len(manager.load_session(session_id)) == 2
    with claim_session(session_id):
        pass


async def test_stream_without_done_is_failed_and_partial_text_is_saved(sessions, monkeypatch):
    manager, session_id = sessions

    async def source(*args, **kwargs):
        yield {"type": "token", "content": "not finished"}

    monkeypatch.setattr(chat_module.agent_manager, "astream", source)
    response = await chat_module.chat(chat_module.ChatRequest(session_id=session_id, message="hello", stream=False))
    assert response.status_code == 502
    body = json.loads(response.body)
    assert body["content"] == "not finished" and body["status"] == "failed"
    assert manager.load_session(session_id)[-1]["content"] == "not finished"
