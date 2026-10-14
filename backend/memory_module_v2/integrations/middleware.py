"""Memory v2 context builder for forced injection mode (MEMORY_V2_INJECT=always).

This is NOT a LangChain middleware — it's a helper called by AgentManager.astream()
when inject_mode="always". When inject_mode="tool", the agent calls search_memory
autonomously via the registered tool, and this module is not involved.
"""

from __future__ import annotations

import logging

from ..domain.enums import SearchMode
from ..domain.models import MemorySearchFilters
from ..service.api import search_memory
from ..service.config import get_memory_v2_config

logger = logging.getLogger(__name__)


def build_memory_context(
    user_message: str,
    *,
    top_k: int | None = None,
    session_ids: list[str] | None = None,
) -> str | None:
    """Retrieve memory context for forced injection.

    Searches across ALL sessions by default (cross-session long-term memory).
    Pass session_ids only to narrow scope intentionally.

    Returns formatted context string or None if no relevant hits.
    """
    config = get_memory_v2_config()
    if not config.enabled:
        return None

    k = top_k or config.inject_top_k

    filters = None
    if session_ids:
        filters = MemorySearchFilters(session_ids=session_ids)

    try:
        response = search_memory(
            user_message,
            mode=SearchMode.HYBRID_CROSS,
            top_k=k,
            filters=filters,
        )
    except Exception as exc:
        logger.warning("Memory v2 retrieval failed: %s", exc)
        return None

    if not response.hits:
        return None

    parts: list[str] = ["[长期记忆检索结果 / Memory v2 – cross-session retrieval]"]
    for hit in response.hits:
        entry = (
            f"来源: session={hit.session_id[:12]}… "
            f"ply={hit.ply_start}–{hit.ply_end} "
            f"score={hit.scores.get('fused', 0):.3f}\n"
            f"{hit.verbatim_snippet[:800]}"
        )
        parts.append(entry)

    return "\n\n---\n\n".join(parts)
