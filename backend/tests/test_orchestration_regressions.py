from __future__ import annotations

import asyncio
import threading
from contextlib import aclosing
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from graph import agent as agent_module
from graph import checkpointer
from graph.agent import AgentManager, bounded_history
from graph.complexity_assessor import CaseComplexity, ComplexityDecision
from graph.context import RequestContext
from graph.coordinator import Coordinator, stream_role_round
from graph.roles.base_role import RoleAgent, RoleOpinion, RoleType, run_model_worker
from service.experiment import ExperimentMode


@pytest.fixture
def manager(tmp_path, monkeypatch):
    manager = AgentManager()
    manager.base_dir = tmp_path
    manager.coordinator = Coordinator(tmp_path)
    monkeypatch.setattr(agent_module, "get_memory_backend", lambda: "none")
    monkeypatch.setattr(agent_module.experiment_logger, "log", lambda _: None)
    return manager


def test_history_bound_keeps_summary_and_recent_without_tool_replay(monkeypatch):
    monkeypatch.setenv("AGENT_HISTORY_MAX_MESSAGES", "4")
    monkeypatch.setenv("AGENT_HISTORY_MAX_CHARS", "512")
    history = [{"role": "assistant", "content": "[以下是之前对话的摘要]\ncompressed"}]
    history += [{"role": "user", "content": f"old-{i}" + "x" * 200, "tool_calls": [{"id": "danger"}]} for i in range(10)]
    bounded = bounded_history(history)
    assert bounded[0]["content"].endswith("compressed")
    assert len(bounded) <= 4
    assert sum(len(item["content"]) for item in bounded) <= 512
    assert "old-9" in bounded[-1]["content"]
    assert all(set(item) == {"role", "content"} for item in bounded)
    assert history[-1]["tool_calls"]


async def test_single_real_checkpoint_seed_only_when_empty_and_restart(manager, monkeypatch):
    received = []

    async def respond(state):
        received.append([message.content for message in state["messages"]])
        return {"messages": [AIMessage(content="answer")]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", respond)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    saver = InMemorySaver()
    graph = builder.compile(checkpointer=saver)
    monkeypatch.setattr(manager, "_build_agent", lambda: graph)
    context = RequestContext(thread_id="history-test")
    history = [{"role": "assistant", "content": "[以下是之前对话的摘要]\nsummary"}, {"role": "user", "content": "prior"}]
    await collect(manager.astream("first", history, context))
    await collect(manager.astream("second", history, context))
    assert received[0] == [history[0]["content"], "prior", "first"]
    assert received[1].count("prior") == 1
    assert received[1][-1] == "second"
    # Restarting the memory saver restores the durable summary and history.
    graph = builder.compile(checkpointer=InMemorySaver())
    await collect(manager.astream("after-restart", history, context))
    assert received[-1] == [history[0]["content"], "prior", "after-restart"]


async def test_streamed_parallel_tools_have_ids_and_one_response_boundary(manager, monkeypatch):
    class Graph:
        async def aget_state(self, config):
            return SimpleNamespace(values={})

        async def astream(self, *args, **kwargs):
            yield "messages", (AIMessageChunk(content="checking"), {"langgraph_node": "agent"})
            yield "updates", {"agent": {"messages": [AIMessage(content="checking", tool_calls=[
                {"id": "a", "name": "write", "args": {"x": 1}},
                {"id": "b", "name": "write", "args": {"x": 2}},
            ])]}}
            yield "updates", {"tools": {"messages": [ToolMessage(content="second", tool_call_id="b", name="write")]}}
            yield "updates", {"tools": {"messages": [ToolMessage(content="first", tool_call_id="a", name="write")]}}
            yield "messages", (AIMessageChunk(content="final"), {"langgraph_node": "agent"})
            yield "updates", {"agent": {"messages": [AIMessage(content="final")]}}

    monkeypatch.setattr(manager, "_build_agent", lambda: Graph())
    events = await collect(manager.astream("hello", []))
    assert [event["tool_call_id"] for event in events if event["type"] == "tool_start"] == ["a", "b"]
    assert [event["tool_call_id"] for event in events if event["type"] == "tool_end"] == ["b", "a"]
    assert sum(event["type"] == "new_response" for event in events) == 1
    boundary = next(i for i, event in enumerate(events) if event["type"] == "new_response")
    assert events[boundary - 1]["tool_call_id"] == "a"
    assert events[boundary + 1] == {"type": "token", "content": "final"}
    assert events[-1]["content"] == "final"


async def test_round_timeout_and_close_cancel_and_join_children(monkeypatch):
    monkeypatch.setenv("MDT_ROLE_TIMEOUT_SECONDS", "0.02")
    monkeypatch.setenv("MDT_ROUND_TIMEOUT_SECONDS", "0.2")
    stopped = asyncio.Event()

    async def hang():
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    results = await collect(stream_role_round({RoleType.SURGEON: hang}, 1))
    assert results[0][1] is None and "Timeout" in results[0][2]
    assert stopped.is_set()
    stopped.clear()
    monkeypatch.setenv("MDT_ROLE_TIMEOUT_SECONDS", "10")

    async def ready():
        await asyncio.sleep(0)
        return RoleOpinion(RoleType.PATHOLOGIST, "pathologist", "ok")

    async with aclosing(stream_role_round({RoleType.SURGEON: hang, RoleType.PATHOLOGIST: ready}, 1)) as stream:
        assert (await anext(stream))[1].content == "ok"
    assert stopped.is_set()


async def test_round_deadline_preserves_completed_expert(monkeypatch):
    monkeypatch.setenv("MDT_ROLE_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("MDT_ROUND_TIMEOUT_SECONDS", "0.02")
    stopped = asyncio.Event()

    async def fast():
        return RoleOpinion(RoleType.SURGEON, "surgeon", "good")

    async def slow():
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    results = await collect(stream_role_round({RoleType.SURGEON: fast, RoleType.PATHOLOGIST: slow}, 1))
    assert results[0][1].content == "good"
    assert results[1][1] is None and "Round 1 timed out" == results[1][2]
    assert stopped.is_set()


@pytest.mark.parametrize("all_fail", [False, True])
async def test_multi_partial_failure_preserves_round1_and_never_claims_success(manager, monkeypatch, all_fail):
    observed = []

    class Expert:
        def __init__(self, role):
            self.role = role

        async def aconsult_round1(self, case, memory_context, attachments=None):
            observed.append(case)
            if all_fail or self.role == RoleType.PATHOLOGIST:
                return RoleOpinion(self.role, self.role.value, "failure", status="failed", error="unavailable")
            return RoleOpinion(self.role, self.role.value, "successful round1")

        async def aconsult_round2(self, *args, **kwargs):
            raise RuntimeError("revision unavailable")

    async def assess(case, method):
        observed.append(case)
        return ComplexityDecision(CaseComplexity.COMPLEX, "test", "manual")

    async def synthesize(case, opinions, weights):
        assert "summary" in case and "prior patient" in case
        assert len(opinions) == 1 and opinions[0].content == "successful round1"
        return "partial synthesis"

    monkeypatch.setattr(agent_module, "assess_complexity", assess)
    manager.coordinator._roles = {role: Expert(role) for role in (RoleType.SURGEON, RoleType.PATHOLOGIST)}
    monkeypatch.setattr(manager.coordinator, "_arbitrate", synthesize)
    history = [{"role": "assistant", "content": "[以下是之前对话的摘要]\nsummary"}, {"role": "user", "content": "prior patient"}]
    events = await collect(manager.astream_multi_agent("follow-up", history, experiment_mode=ExperimentMode.MULTI_WITH_MEMORY))
    assert all("summary" in case and "prior patient" in case and "follow-up" in case for case in observed)
    assert events[-1]["status"] == ("failed" if all_fail else "partial")
    assert events[-1]["failed_roles"]
    assert any(event["type"] == "error" for event in events) == all_fail


async def test_cancelled_model_worker_does_not_allow_parallel_or_queued_gpu_work():
    started = threading.Event()
    finish = threading.Event()
    second_started = threading.Event()

    def blocking():
        started.set()
        finish.wait(timeout=5)
        return "first"

    task = asyncio.create_task(run_model_worker(blocking))
    try:
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.005)
        assert started.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        second = asyncio.create_task(run_model_worker(lambda: second_started.set()))
        await asyncio.sleep(0.03)
        assert not second_started.is_set()
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
    finally:
        finish.set()
    # A healthy later request still works after the cancelled request finishes.
    assert await asyncio.wait_for(run_model_worker(lambda: "healthy"), timeout=2) == "healthy"
    assert not second_started.is_set()


async def test_role_local_load_and_generate_off_event_loop_and_keep_memory(tmp_path, monkeypatch):
    main_thread = threading.get_ident()
    observed = []

    class LocalModel:
        def generate(self, **kwargs):
            observed.append((threading.get_ident(), kwargs))
            return "opinion"

    role = RoleAgent(tmp_path)
    role._system_prompt = "system"

    def build(temperature):
        observed.append((threading.get_ident(), {}))
        return LocalModel(), "local"

    monkeypatch.setattr(role, "_build_llm_with_source", build)
    opinion = await role.aconsult_round1("case", "saved memory")
    assert opinion.status == "ok"
    assert all(thread_id != main_thread for thread_id, _ in observed)
    assert "saved memory" in observed[-1][1]["system_prompt"]


async def test_synthesis_timeout_is_partial_and_preserves_successful_experts(manager, monkeypatch):
    monkeypatch.setenv("MDT_STAGE_TIMEOUT_SECONDS", "0.02")
    stopped = asyncio.Event()

    class Expert:
        async def aconsult_round1(self, *args, **kwargs):
            return RoleOpinion(RoleType.SURGEON, "surgeon", "successful expert")

    async def assess(*args, **kwargs):
        return ComplexityDecision(CaseComplexity.MODERATE, "test", "manual")

    async def synthesize(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    manager.coordinator._roles = {RoleType.SURGEON: Expert()}
    monkeypatch.setattr(agent_module, "assess_complexity", assess)
    monkeypatch.setattr(manager.coordinator, "_aggregate", synthesize)
    events = await collect(manager.astream_multi_agent("case", [], experiment_mode=ExperimentMode.MULTI_NO_MEMORY))
    assert stopped.is_set()
    assert events[-1]["status"] == "partial"
    assert "successful expert" in events[-1]["content"]
    assert any(event["type"] == "role_error" and event["stage"] == "synthesis" for event in events)


async def test_synthesis_failure_not_hidden_as_success(manager, monkeypatch):
    from graph import coordinator as coordinator_module

    class FailingModel:
        async def ainvoke(self, messages):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(coordinator_module, "get_llm", lambda config: FailingModel())
    with pytest.raises(RuntimeError, match="model unavailable"):
        await manager.coordinator._aggregate("case", [RoleOpinion(RoleType.SURGEON, "surgeon", "opinion")], {})
    with pytest.raises(RuntimeError, match="No usable expert"):
        await manager.coordinator._aggregate("case", [], {})


async def test_multi_generator_close_joins_running_specialists(manager, monkeypatch):
    stopped = asyncio.Event()

    class Expert:
        def __init__(self, slow=False):
            self.slow = slow

        async def aconsult_round1(self, *args, **kwargs):
            if self.slow:
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()
            await asyncio.sleep(0)
            return RoleOpinion(RoleType.SURGEON, "surgeon", "ok")

    async def assess(*args, **kwargs):
        return ComplexityDecision(CaseComplexity.COMPLEX, "test", "manual")

    monkeypatch.setattr(agent_module, "assess_complexity", assess)
    manager.coordinator._roles = {RoleType.SURGEON: Expert(), RoleType.PATHOLOGIST: Expert(slow=True)}
    stream = manager.astream_multi_agent("case", [], experiment_mode=ExperimentMode.MULTI_NO_MEMORY)
    async for event in stream:
        if event["type"] == "role_opinion":
            break
    await stream.aclose()
    assert stopped.is_set()


async def test_checkpoint_invalidation_uses_production_saver(manager, monkeypatch):
    saver = InMemorySaver()
    monkeypatch.setattr(checkpointer, "_default_saver", saver)
    builder = StateGraph(MessagesState)
    builder.add_node("agent", lambda _: {"messages": [AIMessage(content="answer")]})
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "compress-test"}}
    await graph.ainvoke({"messages": [{"role": "user", "content": "original"}]}, config)
    assert (await graph.aget_state(config)).values["messages"]
    await manager.invalidate_session_checkpoint("compress-test")
    assert not (await graph.aget_state(config)).values


async def collect(stream):
    return [event async for event in stream]
