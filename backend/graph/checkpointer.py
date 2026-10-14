"""短期记忆：LangChain/LangGraph checkpointer，按 thread_id 持久化对话状态。

支持 InMemorySaver（默认）与 AsyncPostgresSaver（CHECKPOINTER=postgres）。
使用 Postgres 时需在应用启动时调用 init_checkpointer_async() 完成建表与连接。
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

# 默认使用内存存储；CHECKPOINTER=postgres 时使用 AsyncPostgresSaver
_default_saver: Any = None
_postgres_cm: Any = None  # 保持 AsyncPostgresSaver 的 context manager 引用，避免连接被关
_init_lock = asyncio.Lock()


def _use_postgres() -> bool:
    return os.getenv("CHECKPOINTER", "memory").strip().lower() == "postgres"


def _postgres_dsn() -> str:
    dsn = os.getenv("POSTGRES_DSN", "").strip()
    if dsn:
        return dsn
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    db = os.getenv("POSTGRES_DB", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def get_checkpointer() -> Any:
    """返回当前配置的 checkpointer 单例（InMemorySaver 或 AsyncPostgresSaver）。
    使用 Postgres 时须先调用 init_checkpointer_async()，否则会抛错。"""
    global _default_saver
    if _default_saver is not None:
        return _default_saver
    if _use_postgres():
        raise RuntimeError(
            "CHECKPOINTER=postgres 时须在应用启动时先调用 init_checkpointer_async()，并安装 langgraph-checkpoint-postgres"
        )
    _default_saver = InMemorySaver()
    return _default_saver


async def init_checkpointer_async() -> None:
    """使用 Postgres 时在应用启动时调用一次：建表并建立连接。非 Postgres 时初始化 InMemorySaver。"""
    global _default_saver, _postgres_cm
    async with _init_lock:
        await _init_checkpointer_nolock()


async def _init_checkpointer_nolock() -> None:
    global _default_saver, _postgres_cm
    if _default_saver is not None:
        return
    if _use_postgres():
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            dsn = _postgres_dsn()
            _postgres_cm = AsyncPostgresSaver.from_conn_string(dsn)
            _default_saver = await _postgres_cm.__aenter__()
            await _default_saver.setup()
        except ImportError as e:
            raise RuntimeError(
                "CHECKPOINTER=postgres 需要安装: pip install langgraph-checkpoint-postgres"
            ) from e
    else:
        _default_saver = InMemorySaver()


async def reconnect_checkpointer_async() -> None:
    """重建 checkpointer 连接（用于数据库重启后连接失效场景）。"""
    global _default_saver, _postgres_cm
    async with _init_lock:
        old_cm = _postgres_cm
        _default_saver = None
        _postgres_cm = None
        if old_cm is not None:
            try:
                await old_cm.__aexit__(None, None, None)
            except Exception:
                # 旧连接可能已损坏，忽略关闭异常并继续重建
                pass
        await _init_checkpointer_nolock()
