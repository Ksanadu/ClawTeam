"""通过工厂按配置创建 Agent（checkpointer、tools、prompt、middleware）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from graph.guardian import build_guardian_middleware
from graph.checkpointer import get_checkpointer
from service.prompt_builder import build_system_prompt
from graph.llm import build_llm_config_from_settings, get_llm

# 兼容 LangGraph 的 CompiledStateGraph 类型
AgentGraph = Any

# 默认：消息数达到 50 触发压缩，保留最近 20 条
DEFAULT_SUMMARIZATION_TRIGGER_MESSAGES = 50
DEFAULT_SUMMARIZATION_KEEP_MESSAGES = 20


def _summarization_trigger_messages() -> int:
    v = os.getenv("SUMMARIZATION_TRIGGER_MESSAGES", "").strip()
    if not v:
        return DEFAULT_SUMMARIZATION_TRIGGER_MESSAGES
    try:
        return max(1, int(v))
    except ValueError:
        return DEFAULT_SUMMARIZATION_TRIGGER_MESSAGES


def _summarization_keep_messages() -> int:
    v = os.getenv("SUMMARIZATION_KEEP_MESSAGES", "").strip()
    if not v:
        return DEFAULT_SUMMARIZATION_KEEP_MESSAGES
    try:
        return max(1, int(v))
    except ValueError:
        return DEFAULT_SUMMARIZATION_KEEP_MESSAGES


@dataclass
class AgentConfig:
    """Agent 构建所需配置。"""

    llm: BaseChatModel
    tools: list[BaseTool]
    system_prompt: str
    checkpointer: Any | None = None
    guardian_enabled: bool = True
    use_summarization: bool = False
    summarization_trigger_messages: int = DEFAULT_SUMMARIZATION_TRIGGER_MESSAGES
    summarization_keep_messages: int = DEFAULT_SUMMARIZATION_KEEP_MESSAGES


def build_agent_config(
    base_dir: Path,
    tools: list[BaseTool],
    *,
    use_checkpointer: bool = True,
    use_summarization: bool | None = None,
) -> AgentConfig:
    """从当前运行配置与 base_dir、tools 构建 AgentConfig。"""
    from config import get_settings

    settings = get_settings()
    prompt = build_system_prompt(base_dir) if base_dir else ""
    llm = get_llm(build_llm_config_from_settings(settings, temperature=0.0, streaming=True))
    checkpointer = get_checkpointer() if use_checkpointer else None
    if use_summarization is None:
        use_summarization = os.getenv("SUMMARIZATION_ENABLED", "false").strip().lower() in ("true", "1", "yes")
    return AgentConfig(
        llm=llm,
        tools=tools,
        system_prompt=prompt,
        checkpointer=checkpointer,
        guardian_enabled=settings.guardian_enabled,
        use_summarization=use_summarization,
        summarization_trigger_messages=_summarization_trigger_messages(),
        summarization_keep_messages=_summarization_keep_messages(),
    )


def create_agent_from_config(config: AgentConfig) -> AgentGraph:
    """根据 AgentConfig 创建带 checkpointer、可选 Guardian / Summarization 的 agent graph。"""
    middleware: list[Any] = []
    if config.guardian_enabled:
        middleware.append(build_guardian_middleware())
    if config.use_summarization:
        middleware.append(
            SummarizationMiddleware(
                model=config.llm,
                trigger=("messages", config.summarization_trigger_messages),
                keep=("messages", config.summarization_keep_messages),
            )
        )
    return create_agent(
        model=config.llm,
        tools=config.tools,
        system_prompt=config.system_prompt,
        checkpointer=config.checkpointer,
        middleware=middleware if middleware else (),
    )

