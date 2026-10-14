"""Request-scoped context for agent calls: thread_id, callbacks, optional metadata."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


def _build_langfuse_callbacks() -> list[BaseCallbackHandler]:
    """若已配置 Langfuse，返回包含 Langfuse handler 的列表；否则返回空列表。不抛错。"""
    secret = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    public = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    if not secret or not public:
        print("[langfuse] LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY 未配置，跳过 Langfuse callback。")
        return []
    try:
        # Langfuse v3 官方推荐写法：通过环境变量或 Langfuse client 配置，
        # 这里的 CallbackHandler 不再接收 public_key / secret_key / host 等参数。
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
        # Initialize Langfuse client
        langfuse = get_client()
        base_url = (os.getenv("LANGFUSE_BASE_URL") or "").strip() or None
        environment = (os.getenv("LANGFUSE_ENV") or "").strip() or "default"
        print(
            "[langfuse] 初始化 LangfuseCallbackHandler",
            "base_url=",
            base_url,
            "environment=",
            environment,
        )
        # 依赖环境变量中的 LANGFUSE_* 来完成鉴权和环境配置
        handler = LangfuseCallbackHandler()
        print("[langfuse] LangfuseCallbackHandler 创建成功：", handler)
        return [handler]
    except Exception as exc:
        print("[langfuse] 初始化 LangfuseCallbackHandler 失败：", repr(exc))
        return []


@dataclass
class RequestContext:
    """单次请求的上下文，供 API 与 graph 共享。"""

    thread_id: str
    """与 session_id 一致，用于 checkpointer 与 trace 关联。"""
    callbacks: list[BaseCallbackHandler] = field(default_factory=list)
    """本请求使用的 callback（如 Langfuse），在 invoke/astream 时传入。"""
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_langfuse(self) -> RequestContext:
        """在现有 callbacks 基础上追加 Langfuse handler（若已配置）。返回新实例。"""
        extra = _build_langfuse_callbacks()
        if not extra:
            return self
        return RequestContext(
            thread_id=self.thread_id,
            callbacks=[*self.callbacks, *extra],
            request_id=self.request_id,
            metadata=dict(self.metadata),
        )


def build_request_context(
    thread_id: str,
    *,
    request_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    include_langfuse: bool = True,
) -> RequestContext:
    """构造 RequestContext，可选自动加入 Langfuse。"""
    ctx = RequestContext(
        thread_id=thread_id,
        callbacks=[],
        request_id=request_id,
        metadata=metadata or {},
    )
    if include_langfuse:
        ctx = ctx.with_langfuse()
    return ctx
