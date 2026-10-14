from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

if "langchain_core.callbacks" not in sys.modules:
    callbacks_mod = types.ModuleType("langchain_core.callbacks")
    callbacks_mod.BaseCallbackHandler = type("BaseCallbackHandler", (), {})
    sys.modules["langchain_core.callbacks"] = callbacks_mod

if "langchain_core.language_models.chat_models" not in sys.modules:
    chat_models_mod = types.ModuleType("langchain_core.language_models.chat_models")
    chat_models_mod.BaseChatModel = type("BaseChatModel", (), {})
    sys.modules["langchain_core.language_models.chat_models"] = chat_models_mod

if "langchain_core.tools" not in sys.modules:
    tools_mod = types.ModuleType("langchain_core.tools")
    tools_mod.BaseTool = type("BaseTool", (), {})
    sys.modules["langchain_core.tools"] = tools_mod

if "langchain.agents.middleware" not in sys.modules:
    middleware_mod = types.ModuleType("langchain.agents.middleware")
    middleware_mod.SummarizationMiddleware = type("SummarizationMiddleware", (), {})
    sys.modules["langchain.agents.middleware"] = middleware_mod

if "langchain.agents" not in sys.modules:
    agents_mod = types.ModuleType("langchain.agents")

    def _fake_create_agent(**kwargs):
        return kwargs

    agents_mod.create_agent = _fake_create_agent
    sys.modules["langchain.agents"] = agents_mod

if "graph.checkpointer" not in sys.modules:
    checkpointer_mod = types.ModuleType("graph.checkpointer")
    checkpointer_mod.get_checkpointer = lambda: None
    sys.modules["graph.checkpointer"] = checkpointer_mod

if "service.prompt_builder" not in sys.modules:
    prompt_builder_mod = types.ModuleType("service.prompt_builder")
    prompt_builder_mod.build_system_prompt = lambda _base_dir: ""
    sys.modules["service.prompt_builder"] = prompt_builder_mod

if "graph.llm" not in sys.modules:
    llm_mod = types.ModuleType("graph.llm")
    llm_mod.build_llm_config_from_settings = lambda *_args, **_kwargs: {}
    llm_mod.get_llm = lambda *_args, **_kwargs: object()
    sys.modules["graph.llm"] = llm_mod

if "tools" not in sys.modules:
    tools_module = types.ModuleType("tools")
    tools_module.get_all_tools = lambda _base_dir: []
    sys.modules["tools"] = tools_module

if "service.memory_indexer" not in sys.modules:
    memory_indexer_mod = types.ModuleType("service.memory_indexer")
    memory_indexer_mod.memory_indexer = SimpleNamespace(retrieve=lambda *_args, **_kwargs: [])
    sys.modules["service.memory_indexer"] = memory_indexer_mod

if "service.session_manager" not in sys.modules:
    session_manager_mod = types.ModuleType("service.session_manager")
    session_manager_mod.SessionManager = type("SessionManager", (), {})
    sys.modules["service.session_manager"] = session_manager_mod

if "memory_module_v2.service.config" not in sys.modules:
    memory_config_mod = types.ModuleType("memory_module_v2.service.config")
    memory_config_mod.get_memory_backend = lambda: "none"
    memory_config_mod.get_memory_v2_inject_mode = lambda: "tool"
    sys.modules["memory_module_v2.service.config"] = memory_config_mod

if "memory_module_v2.integrations.middleware" not in sys.modules:
    memory_mw_mod = types.ModuleType("memory_module_v2.integrations.middleware")
    memory_mw_mod.build_memory_context = lambda _query: ""
    sys.modules["memory_module_v2.integrations.middleware"] = memory_mw_mod

from backend.graph.agent_factory import AgentConfig, create_agent_from_config


def test_guardian_middleware_before_summarization_when_both_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeSummarizationMiddleware:
        def __init__(self, **_kwargs):
            pass

    def _fake_create_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("backend.graph.agent_factory.SummarizationMiddleware", _FakeSummarizationMiddleware)
    monkeypatch.setattr("backend.graph.agent_factory.create_agent", _fake_create_agent)

    config = AgentConfig(
        llm=object(),
        tools=[],
        system_prompt="",
        guardian_enabled=True,
        use_summarization=True,
    )
    create_agent_from_config(config)

    middleware = list(captured["middleware"])
    assert len(middleware) == 2
    assert middleware[0].__class__.__name__ == "GuardianMiddleware"
    assert middleware[1].__class__.__name__ == "_FakeSummarizationMiddleware"


def test_only_guardian_middleware_when_summarization_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_create_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("backend.graph.agent_factory.create_agent", _fake_create_agent)

    config = AgentConfig(
        llm=object(),
        tools=[],
        system_prompt="",
        guardian_enabled=True,
        use_summarization=False,
    )
    create_agent_from_config(config)

    middleware = list(captured["middleware"])
    assert len(middleware) == 1
    assert middleware[0].__class__.__name__ == "GuardianMiddleware"


def test_no_guardian_middleware_when_disabled_and_no_summarization(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_create_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("backend.graph.agent_factory.create_agent", _fake_create_agent)

    config = AgentConfig(
        llm=object(),
        tools=[],
        system_prompt="",
        guardian_enabled=False,
        use_summarization=False,
    )
    create_agent_from_config(config)

    assert captured["middleware"] == ()


def test_only_summarization_when_guardian_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeSummarizationMiddleware:
        def __init__(self, **_kwargs):
            pass

    def _fake_create_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("backend.graph.agent_factory.SummarizationMiddleware", _FakeSummarizationMiddleware)
    monkeypatch.setattr("backend.graph.agent_factory.create_agent", _fake_create_agent)

    config = AgentConfig(
        llm=object(),
        tools=[],
        system_prompt="",
        guardian_enabled=False,
        use_summarization=True,
    )
    create_agent_from_config(config)

    middleware = list(captured["middleware"])
    assert len(middleware) == 1
    assert middleware[0].__class__.__name__ == "_FakeSummarizationMiddleware"
