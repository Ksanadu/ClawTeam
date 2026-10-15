from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from graph import agent_factory, guardian


@pytest.mark.parametrize("enabled,summarize,expected", [(True, True, 2), (True, False, 1), (False, True, 1), (False, False, 0)])
def test_real_factory_middleware_composition(monkeypatch, enabled, summarize, expected):
    captured = {}
    monkeypatch.setattr(agent_factory, "create_agent", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(agent_factory, "SummarizationMiddleware", lambda **kwargs: SimpleNamespace(**kwargs))
    agent_factory.create_agent_from_config(agent_factory.AgentConfig(
        llm=object(), tools=[], system_prompt="", guardian_enabled=enabled, use_summarization=summarize,
    ))
    middleware = list(captured["middleware"])
    assert len(middleware) == expected
    if enabled:
        assert isinstance(middleware[0], guardian.GuardianMiddleware)


async def test_async_guardian_hook_never_uses_sync_evaluator(monkeypatch):
    monkeypatch.setenv("GUARDIAN_ENABLED", "true")
    monkeypatch.setattr(guardian, "evaluate_guardian_input", lambda _: pytest.fail("sync Guardian called"))
    observed = []

    async def evaluate(text):
        observed.append(text)
        return guardian.GuardianRuntimeResult(True, "危险", "test", "blocked")

    monkeypatch.setattr(guardian, "aevaluate_guardian_input", evaluate)
    result = await guardian.GuardianMiddleware().abefore_agent({"messages": [HumanMessage(content="latest")]}, None)
    assert observed == ["latest"]
    assert result["jump_to"] == "end"
    assert isinstance(result["messages"][0], AIMessage)


async def test_guardian_async_timeout_is_fail_closed(monkeypatch):
    import asyncio
    monkeypatch.setenv("GUARDIAN_ENABLED", "true")
    monkeypatch.setenv("GUARDIAN_TIMEOUT_MS", "100")
    cancelled = asyncio.Event()

    async def request(_):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(guardian, "_arequest_guardian_label", request)
    result = await guardian.aevaluate_guardian_input("test")
    assert result.is_blocked
    assert result.reason_code == "upstream_timeout"
    assert cancelled.is_set()
