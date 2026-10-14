from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

LLM_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "zhipu": {
        "model": "glm-5",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
    },
    "bailian": {
        "model": "qwen3.5-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "deepseek": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
    },
    "openai": {
        "model": "gpt-4.1-mini",
        "base_url": "https://api.openai.com/v1",
    },
}

EMBEDDING_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "bailian": {
        "model": "text-embedding-v4",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "openai": {
        "model": "text-embedding-3-small",
        "base_url": "https://api.openai.com/v1",
    },
}

PROVIDER_ALIASES = {
    "glm": "zhipu",
    "zhipuai": "zhipu",
    "bigmodel": "zhipu",
    "aliyun": "bailian",
    "dashscope": "bailian",
    "qwen": "bailian",
    "openai-compatible": "openai",
    "compatible": "openai",
}


@dataclass(frozen=True)
class Settings:
    config_dir: Path
    backend_dir: Path
    project_root: Path
    llm_provider: str
    llm_model: str
    llm_api_key: str | None
    llm_base_url: str
    embedding_provider: str
    embedding_model: str
    embedding_api_key: str | None
    embedding_base_url: str
    guardian_enabled: bool = True
    guardian_provider: str = "openai"
    guardian_model: str = LLM_PROVIDER_DEFAULTS["openai"]["model"]
    guardian_api_key: str | None = None
    guardian_base_url: str = LLM_PROVIDER_DEFAULTS["openai"]["base_url"]
    guardian_timeout_ms: int = 1500
    guardian_fail_mode: str = "closed"
    guardian_block_message: str = "检测到潜在提示词攻击风险，本次请求已被拦截。"
    component_char_limit: int = 20_000
    terminal_timeout_seconds: int = 30


@dataclass(frozen=True)
class ProjectPaths:
    config_dir: Path
    backend_dir: Path
    project_root: Path


def _load_env_file() -> ProjectPaths:
    config_dir = Path(__file__).resolve().parent
    backend_dir = config_dir.parent
    project_root = backend_dir.parent
    load_dotenv(config_dir / ".env")
    return ProjectPaths(
        config_dir=config_dir,
        backend_dir=backend_dir,
        project_root=project_root,
    )


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _normalize_provider(
    value: str | None,
    *,
    default: str,
    defaults: dict[str, dict[str, str]],
) -> str:
    normalized = (value or default).strip().lower()
    normalized = PROVIDER_ALIASES.get(normalized, normalized)
    if normalized in defaults:
        return normalized
    return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _resolve_guardian_fail_mode() -> str:
    value = (os.getenv("GUARDIAN_FAIL_MODE") or "closed").strip().lower()
    if value in {"open", "closed"}:
        return value
    return "closed"


def _resolve_guardian_model(provider: str) -> str:
    return _first_env("GUARDIAN_MODEL") or LLM_PROVIDER_DEFAULTS[provider]["model"]


def _resolve_guardian_base_url(provider: str) -> str:
    return _first_env("GUARDIAN_BASE_URL") or LLM_PROVIDER_DEFAULTS[provider]["base_url"]


def _resolve_llm_api_key(provider: str) -> str | None:
    if provider == "zhipu":
        return _first_env("LLM_API_KEY", "ZHIPU_API_KEY", "ZHIPUAI_API_KEY")
    if provider == "bailian":
        return _first_env("LLM_API_KEY", "BAILIAN_API_KEY", "DASHSCOPE_API_KEY")
    if provider == "deepseek":
        return _first_env("LLM_API_KEY", "DEEPSEEK_API_KEY")
    return _first_env("LLM_API_KEY", "OPENAI_API_KEY")


def _resolve_llm_model(provider: str) -> str:
    if provider == "zhipu":
        return _first_env("LLM_MODEL", "ZHIPU_MODEL") or LLM_PROVIDER_DEFAULTS[provider]["model"]
    if provider == "bailian":
        return _first_env("LLM_MODEL", "BAILIAN_MODEL") or LLM_PROVIDER_DEFAULTS[provider]["model"]
    if provider == "deepseek":
        return _first_env("LLM_MODEL", "DEEPSEEK_MODEL") or LLM_PROVIDER_DEFAULTS[provider]["model"]
    return _first_env("LLM_MODEL") or LLM_PROVIDER_DEFAULTS[provider]["model"]


def _resolve_llm_base_url(provider: str) -> str:
    if provider == "zhipu":
        return _first_env("LLM_BASE_URL", "ZHIPU_BASE_URL") or LLM_PROVIDER_DEFAULTS[provider]["base_url"]
    if provider == "bailian":
        return _first_env("LLM_BASE_URL", "BAILIAN_BASE_URL") or LLM_PROVIDER_DEFAULTS[provider]["base_url"]
    if provider == "deepseek":
        return _first_env("LLM_BASE_URL", "DEEPSEEK_BASE_URL") or LLM_PROVIDER_DEFAULTS[provider]["base_url"]
    return _first_env("LLM_BASE_URL", "OPENAI_BASE_URL") or LLM_PROVIDER_DEFAULTS[provider]["base_url"]


def _resolve_embedding_api_key(provider: str) -> str | None:
    if provider == "bailian":
        return _first_env("EMBEDDING_API_KEY", "BAILIAN_API_KEY", "DASHSCOPE_API_KEY")
    return _first_env("EMBEDDING_API_KEY", "OPENAI_API_KEY")


def _resolve_embedding_model(provider: str) -> str:
    return _first_env("EMBEDDING_MODEL") or EMBEDDING_PROVIDER_DEFAULTS[provider]["model"]


def _resolve_embedding_base_url(provider: str) -> str:
    if provider == "bailian":
        return (
            _first_env("EMBEDDING_BASE_URL", "BAILIAN_BASE_URL")
            or EMBEDDING_PROVIDER_DEFAULTS[provider]["base_url"]
        )
    return (
        _first_env("EMBEDDING_BASE_URL", "OPENAI_BASE_URL")
        or EMBEDDING_PROVIDER_DEFAULTS[provider]["base_url"]
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    paths = _load_env_file()

    llm_provider = _normalize_provider(
        os.getenv("LLM_PROVIDER"),
        default="zhipu",
        defaults=LLM_PROVIDER_DEFAULTS,
    )
    embedding_provider = _normalize_provider(
        os.getenv("EMBEDDING_PROVIDER"),
        default="bailian",
        defaults=EMBEDDING_PROVIDER_DEFAULTS,
    )
    guardian_provider = _normalize_provider(
        os.getenv("GUARDIAN_PROVIDER"),
        default="openai",
        defaults=LLM_PROVIDER_DEFAULTS,
    )

    return Settings(
        config_dir=paths.config_dir,
        backend_dir=paths.backend_dir,
        project_root=paths.project_root,
        llm_provider=llm_provider,
        llm_model=_resolve_llm_model(llm_provider),
        llm_api_key=_resolve_llm_api_key(llm_provider),
        llm_base_url=_resolve_llm_base_url(llm_provider),
        embedding_provider=embedding_provider,
        embedding_model=_resolve_embedding_model(embedding_provider),
        embedding_api_key=_resolve_embedding_api_key(embedding_provider),
        embedding_base_url=_resolve_embedding_base_url(embedding_provider),
        guardian_enabled=_env_bool("GUARDIAN_ENABLED", True),
        guardian_provider=guardian_provider,
        guardian_model=_resolve_guardian_model(guardian_provider),
        guardian_api_key=_first_env("GUARDIAN_API_KEY"),
        guardian_base_url=_resolve_guardian_base_url(guardian_provider),
        guardian_timeout_ms=_env_int("GUARDIAN_TIMEOUT_MS", 1500),
        guardian_fail_mode=_resolve_guardian_fail_mode(),
        guardian_block_message=(
            os.getenv("GUARDIAN_BLOCK_MESSAGE")
            or "检测到潜在提示词攻击风险，本次请求已被拦截。"
        ),
    )


class RuntimeConfigManager:
    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._lock = threading.RLock()
        self._default_config = {"rag_mode": False}

    def load(self) -> dict[str, Any]:
        with self._lock:
            if not self._config_path.exists():
                self.save(self._default_config)
            try:
                return json.loads(self._config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.save(self._default_config)
                return dict(self._default_config)

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            merged = dict(self._default_config)
            merged.update(payload)
            self._config_path.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return merged

    def get_rag_mode(self) -> bool:
        return bool(self.load().get("rag_mode", False))

    def set_rag_mode(self, enabled: bool) -> dict[str, Any]:
        return self.save({"rag_mode": enabled})


runtime_config = RuntimeConfigManager(get_settings().config_dir / "config.json")
