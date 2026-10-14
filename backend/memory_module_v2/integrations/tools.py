"""LangChain tools for memory_module_v2: search_memory, distill_session."""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.tools import tool

from ..domain.enums import SearchMode
from ..domain.models import MemorySearchFilters
from ..service.api import distill_session as _distill_session
from ..service.api import get_exchange as _get_exchange
from ..service.api import search_memory as _search_memory


@tool
def search_memory(
    query: str,
    mode: str = "hybrid_cross",
    top_k: int = 5,
    session_ids: Optional[str] = None,
) -> str:
    """搜索跨会话的长期记忆，查找与当前问题相关的历史对话片段。
    默认搜索所有历史 session（跨 session 长期记忆）。
    当用户提到过去讨论过的话题、之前解决过的问题、或需要回忆历史上下文时，调用此工具。

    Args:
        query: 搜索查询，描述你要查找的内容。
        mode: 检索模式 - "hybrid_cross"（默认推荐）、"dense_distilled" 或 "keyword_verbatim"。
        top_k: 返回结果数量（默认 5）。
        session_ids: 按 session ID 过滤（逗号分隔，可选；不传则搜索所有 session）。
    """
    filters = None
    if session_ids:
        filters = MemorySearchFilters(session_ids=session_ids.split(","))

    try:
        search_mode = SearchMode(mode)
    except ValueError:
        search_mode = SearchMode.HYBRID_CROSS

    response = _search_memory(
        query=query,
        mode=search_mode,
        top_k=top_k,
        filters=filters,
    )

    if not response.hits:
        return "No relevant memory found."

    parts: list[str] = []
    for hit in response.hits:
        part = (
            f"[#{hit.rank}] session={hit.session_id[:8]}… "
            f"ply={hit.ply_start}-{hit.ply_end} "
            f"score={hit.scores.get('fused', 0):.3f}\n"
            f"{hit.verbatim_snippet[:600]}"
        )
        if hit.files_touched:
            part += f"\nFiles: {', '.join(hit.files_touched[:5])}"
        parts.append(part)

    return "\n\n---\n\n".join(parts)


@tool
def distill_session_tool(
    session_id: str,
    force: bool = False,
) -> str:
    """Distill a conversation session into structured memory objects.

    Args:
        session_id: The session ID to distill.
        force: Force re-distillation even if already processed.
    """
    result = _distill_session(session_id, force=force)

    output = {
        "session_id": result.session_id,
        "exchanges_total": result.exchanges_total,
        "exchanges_new": result.exchanges_new,
        "objects_created": result.objects_created,
        "errors": len(result.errors),
    }
    return json.dumps(output, ensure_ascii=False)


def get_memory_tools() -> list:
    """Return the list of memory v2 tools for agent registration."""
    return [search_memory, distill_session_tool]
