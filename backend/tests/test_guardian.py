from __future__ import annotations

import pytest

from backend.config.config import get_settings
from backend.graph.guardian import (
    GuardianMiddleware,
    GuardianRuntimeResult,
    build_guardian_request_payload,
    classify_guardian_error,
    parse_guardian_label,
    parse_or_fallback_guardian_label,
    resolve_guardian_fallback,
)


@pytest.fixture()
def _clear_guardian_env(monkeypatch):
    for key in (
        "GUARDIAN_ENABLED",
        "GUARDIAN_PROVIDER",
        "GUARDIAN_MODEL",
        "GUARDIAN_API_KEY",
        "GUARDIAN_BASE_URL",
        "GUARDIAN_TIMEOUT_MS",
        "GUARDIAN_FAIL_MODE",
        "GUARDIAN_BLOCK_MESSAGE",
    ):
        monkeypatch.setenv(key, "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_load_guardian_defaults(_clear_guardian_env) -> None:
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.guardian_enabled is True
    assert settings.guardian_provider == "openai"
    assert settings.guardian_model == "gpt-4.1-mini"
    assert settings.guardian_api_key is None
    assert settings.guardian_base_url == "https://api.openai.com/v1"
    assert settings.guardian_timeout_ms == 1500
    assert settings.guardian_fail_mode == "closed"
    assert settings.guardian_block_message == "检测到潜在提示词攻击风险，本次请求已被拦截。"


def test_guardian_enabled_defaults_true_when_missing(monkeypatch, _clear_guardian_env) -> None:
    monkeypatch.setenv("GUARDIAN_ENABLED", "")
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.guardian_enabled is True

    get_settings.cache_clear()


def test_guardian_parse_accepts_only_safe_or_danger() -> None:
    assert parse_guardian_label("安全") == "安全"
    assert parse_guardian_label("危险") == "危险"
    with pytest.raises(ValueError):
        parse_guardian_label("不确定")


def test_guardian_timeout_uses_closed_mode_as_block() -> None:
    verdict = resolve_guardian_fallback(error=TimeoutError(), fail_mode="closed")
    assert verdict == "危险"


def test_guardian_open_mode_fallback_allows() -> None:
    verdict = resolve_guardian_fallback(error=RuntimeError("x"), fail_mode="open")
    assert verdict == "安全"


def test_guardian_request_temperature_is_zero() -> None:
    payload = build_guardian_request_payload("x", model="gpt-4.1-mini")
    assert payload["temperature"] == 0


def test_guardian_maps_http_429_to_fail_mode() -> None:
    label, reason = classify_guardian_error(status_code=429, fail_mode="closed")
    assert label == "危险"
    assert reason == "upstream_rate_limited"


def test_guardian_maps_http_401_to_fail_mode() -> None:
    label, reason = classify_guardian_error(status_code=401, fail_mode="closed")
    assert label == "危险"
    assert reason == "upstream_auth_error"


def test_guardian_maps_http_503_to_fail_mode() -> None:
    label, reason = classify_guardian_error(status_code=503, fail_mode="closed")
    assert label == "危险"
    assert reason == "upstream_unavailable"


def test_guardian_maps_timeout_to_fail_mode() -> None:
    label, reason = classify_guardian_error(
        status_code=None,
        fail_mode="closed",
        error=TimeoutError(),
    )
    assert label == "危险"
    assert reason == "upstream_timeout"


def test_guardian_handles_malformed_response_as_fallback() -> None:
    label = parse_or_fallback_guardian_label("", fail_mode="closed")
    assert label == "危险"


def test_guardian_middleware_before_agent_blocks(monkeypatch: pytest.MonkeyPatch, _clear_guardian_env) -> None:
    monkeypatch.setenv("GUARDIAN_ENABLED", "true")
    get_settings.cache_clear()

    def _fake_eval(_text: str) -> GuardianRuntimeResult:
        return GuardianRuntimeResult(
            is_blocked=True,
            label="危险",
            reason_code="guardian_dangerous",
            block_message="拦截测试",
        )

    monkeypatch.setattr("backend.graph.guardian.evaluate_guardian_input", _fake_eval)
    mw = GuardianMiddleware()
    out = mw.before_agent({"messages": [{"role": "user", "content": "x"}]}, None)  # type: ignore[arg-type]
    assert out is not None
    assert out["jump_to"] == "end"
    assert len(out["messages"]) == 1
    assert getattr(out["messages"][0], "content", None) == "拦截测试"


def test_guardian_middleware_before_agent_passes_when_safe(monkeypatch: pytest.MonkeyPatch, _clear_guardian_env) -> None:
    monkeypatch.setenv("GUARDIAN_ENABLED", "true")
    get_settings.cache_clear()

    def _safe(_text: str) -> GuardianRuntimeResult:
        return GuardianRuntimeResult(
            is_blocked=False,
            label="安全",
            reason_code="ok",
            block_message="",
        )

    monkeypatch.setattr("backend.graph.guardian.evaluate_guardian_input", _safe)
    mw = GuardianMiddleware()
    assert mw.before_agent({"messages": [{"role": "user", "content": "hi"}]}, None) is None  # type: ignore[arg-type]


