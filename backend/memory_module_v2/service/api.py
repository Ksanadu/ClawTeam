"""Service API: distill_session, search_memory, get_exchange."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from config import get_settings
from graph.llm import build_embedding_config_from_settings, get_embedding_model

from ..domain.enums import SearchMode
from ..domain.models import (
    DistillSessionResult,
    ExchangeEvidence,
    MemorySearchFilters,
    MemorySearchResponse,
)
from ..distill.distiller import distill_exchange
from ..ingest.exchange_segmenter import _is_substantive_assistant, segment_exchanges
from ..ingest.text_cleaner import clean_text
from ..ingest.session_reader import load_session_raw, read_session
from ..retrieval.service import retrieval_search
from ..storage.repos import ExchangesRepo, ObjectsRepo
from .config import get_memory_v2_config

logger = logging.getLogger(__name__)

exchanges_repo = ExchangesRepo()
objects_repo = ObjectsRepo()


def distill_session(session_id: str, *, force: bool = False) -> DistillSessionResult:
    """Distill a session into exchanges and structured objects (idempotent)."""
    config = get_memory_v2_config()
    result = DistillSessionResult(
        session_id=session_id,
        started_at=datetime.now(timezone.utc),
    )

    messages = read_session(session_id)
    if not messages:
        logger.warning("No messages found for session %s", session_id)
        result.finished_at = datetime.now(timezone.utc)
        return result

    exchanges = segment_exchanges(
        session_id,
        messages,
        min_exchange_chars=config.min_exchange_chars,
        max_ply_len=config.max_ply_len,
        min_assistant_chars=config.min_assistant_chars,
    )
    result.exchanges_total = len(exchanges)

    existing_ids = exchanges_repo.get_exchange_ids_for_session(session_id) if not force else set()

    new_exchanges = [ex for ex in exchanges if ex.exchange_id not in existing_ids]
    result.exchanges_new = len(new_exchanges)

    exchanges_repo.upsert_batch(new_exchanges if not force else exchanges)

    settings = get_settings()
    emb_config = build_embedding_config_from_settings(settings)
    embedding_model = get_embedding_model(emb_config)

    to_distill = new_exchanges if not force else exchanges
    for exchange in to_distill:
        try:
            obj = distill_exchange(exchange)

            try:
                emb = embedding_model.embed_query(obj.distill_text)
                obj.embedding = emb
            except Exception as emb_exc:
                logger.warning("Embedding failed for %s: %s", obj.object_id, emb_exc)

            objects_repo.upsert(obj)
            result.objects_created += 1
        except Exception as exc:
            logger.exception("Distillation failed for exchange %s", exchange.exchange_id)
            result.errors.append({
                "exchange_id": exchange.exchange_id,
                "error": str(exc),
            })

    result.finished_at = datetime.now(timezone.utc)
    return result


def search_memory(
    query: str,
    *,
    mode: SearchMode = SearchMode.HYBRID_CROSS,
    top_k: int = 10,
    filters: MemorySearchFilters | None = None,
    debug: bool = False,
) -> MemorySearchResponse:
    """Search memory using the two-layer architecture."""
    return retrieval_search(
        query=query,
        mode=mode,
        top_k=top_k,
        filters=filters,
        debug=debug,
    )


def get_exchange(
    session_id: str,
    ply_start: int,
    ply_end: int,
) -> ExchangeEvidence:
    """Evidence drilldown: fetch verbatim messages for an exchange."""
    config = get_memory_v2_config()
    data = load_session_raw(session_id)
    if data is None:
        return ExchangeEvidence(
            session_id=session_id,
            ply_start=ply_start,
            ply_end=ply_end,
        )

    all_messages = data.get("messages", [])
    sliced = all_messages[ply_start: ply_end + 1]

    messages_out: list[dict[str, Any]] = []
    snippet_parts: list[str] = []
    for i, msg in enumerate(sliced):
        msg_index = ply_start + i
        entry: dict[str, Any] = {
            "msg_index": msg_index,
            "role": msg.get("role", "unknown"),
            "content": msg.get("content", ""),
        }
        if msg.get("tool_calls"):
            entry["tool_calls"] = msg["tool_calls"]
        messages_out.append(entry)

        role = msg.get("role", "").upper()
        content = msg.get("content", "")
        if not content:
            continue
        if role == "USER":
            snippet_parts.append(f"{role}: {content}")
        elif role == "ASSISTANT" and _is_substantive_assistant(
            content,
            min_chars=config.min_assistant_chars,
        ):
            snippet_parts.append(f"{role}: {clean_text(content)}")

    return ExchangeEvidence(
        session_id=session_id,
        ply_start=ply_start,
        ply_end=ply_end,
        messages=messages_out,
        verbatim_snippet="\n\n".join(snippet_parts),
    )
