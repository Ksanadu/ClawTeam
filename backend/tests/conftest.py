"""Tests never consume developer credentials or load the project's .env file."""
from __future__ import annotations

import os

import dotenv
import pytest

# Must run before production config is imported during test collection.
_isolation = pytest.MonkeyPatch()
_isolation.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
for _key in list(os.environ):
    if any(word in _key for word in ("API_KEY", "SECRET", "TOKEN", "PASSWORD")) or _key.startswith(("GUARDIAN_", "LLM_", "EMBEDDING_", "LANGFUSE_", "LANGCHAIN_", "LANGSMITH_", "USE_LORA_", "USE_LOCAL_BASE_", "MDT_", "AGENT_HISTORY_", "MEMORY_")):
        _isolation.delenv(_key, raising=False)
_isolation.setenv("LANGCHAIN_TRACING_V2", "false")
_isolation.setenv("LANGSMITH_TRACING", "false")
_isolation.setenv("MEMORY_BACKEND", "none")


@pytest.fixture(autouse=True)
def clear_settings_cache():
    from config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def pytest_unconfigure(config):
    _isolation.undo()
