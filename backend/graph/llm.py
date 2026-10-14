from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

try:
    from langchain_deepseek import ChatDeepSeek
except ImportError:  # pragma: no cover - optional dependency at runtime
    ChatDeepSeek = None

from config import Settings


@dataclass(frozen=True)
class ResolvedLLMConfig:
    provider: str
    model: str
    api_key: str | None
    base_url: str
    temperature: float = 0.6
    streaming: bool = False


@dataclass(frozen=True)
class ResolvedEmbeddingConfig:
    provider: str
    model: str
    api_key: str | None
    base_url: str


def _ensure_api_key(config: ResolvedLLMConfig) -> None:
    if not config.api_key:
        raise RuntimeError(f"Missing API key for provider {config.provider}")


def _build_openai_compatible_chat(config: ResolvedLLMConfig) -> BaseChatModel:
    _ensure_api_key(config)
    kwargs: dict = {}
    if config.streaming:
        kwargs["stream_options"] = {"include_usage": True}
    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        streaming=config.streaming,
        **kwargs,
    )


def _build_tongyi_chat(config: ResolvedLLMConfig) -> BaseChatModel:
    """DashScope（通义千问）官方 SDK 客户端（非 OpenAI 兼容模式）。"""
    _ensure_api_key(config)
    # ChatTongyi 不使用 base_url；通过 dashscope SDK 直连阿里云 DashScope
    return ChatTongyi(
        model=config.model,
        api_key=config.api_key,
        streaming=True,
        model_kwargs={"temperature": config.temperature},
    )


def _build_deepseek_chat(config: ResolvedLLMConfig) -> BaseChatModel:
    if ChatDeepSeek is None:
        return _build_openai_compatible_chat(config)
    _ensure_api_key(config)
    return ChatDeepSeek(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
    )


LLM_REGISTRY: Dict[str, Callable[[ResolvedLLMConfig], BaseChatModel]] = {
    # OpenAI 及兼容模式
    "openai": _build_openai_compatible_chat,
    "zhipu": _build_openai_compatible_chat,
    # DashScope / Bailian / Qwen：使用官方 dashscope SDK（ChatTongyi）
    "bailian": _build_openai_compatible_chat,
    # "bailian": _build_tongyi_chat,
    # "dashscope": _build_tongyi_chat,
    # "qwen": _build_tongyi_chat,
    # DeepSeek 专用客户端
    "deepseek": _build_deepseek_chat,
}


def build_llm_config_from_settings(
    settings: Settings,
    *,
    temperature: float = 0.0,
    streaming: bool = False,
) -> ResolvedLLMConfig:
    return ResolvedLLMConfig(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=temperature,
        streaming=streaming,
    )


def get_llm(config: ResolvedLLMConfig) -> BaseChatModel:
    provider = config.provider
    if provider not in LLM_REGISTRY:
        raise RuntimeError(f"Unsupported LLM provider: {provider}")
    factory = LLM_REGISTRY[provider]
    return factory(config)


def build_embedding_config_from_settings(settings: Settings) -> ResolvedEmbeddingConfig:
    return ResolvedEmbeddingConfig(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
    )


def get_embedding_model(config: ResolvedEmbeddingConfig):
    """
    构建统一的 Embedding 实例。

    - openai: 使用 OpenAIEmbeddings（支持 base_url）
    - bailian/dashscope/qwen: 使用 DashScopeEmbeddings（官方 dashscope SDK）
    """
    if not config.api_key:
        raise RuntimeError(f"Missing embedding API key for provider {config.provider}")

    provider = (config.provider or "").strip().lower()
    if provider in {"bailian", "dashscope", "qwen"}:
        # DashScopeEmbeddings 不使用 base_url
        return DashScopeEmbeddings(model=config.model, dashscope_api_key=config.api_key)

    return OpenAIEmbeddings(model=config.model, api_key=config.api_key, base_url=config.base_url)

