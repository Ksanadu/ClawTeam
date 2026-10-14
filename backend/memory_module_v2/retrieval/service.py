"""Retrieval service: orchestrates dense + keyword + fusion into search_memory."""

from __future__ import annotations

import logging
from typing import Any

from ..domain.enums import SearchMode
from ..domain.models import (
    MemoryHit,
    MemorySearchDebug,
    MemorySearchFilters,
    MemorySearchResponse,
    RoomAssignment,
)
from ..service.config import get_memory_v2_config
from ..storage.repos import ExchangesRepo
from .dense import dense_search
from .fusion import rrf_fusion, weighted_sum_fusion
from .keyword import keyword_search

logger = logging.getLogger(__name__)


def retrieval_search(
    query: str,
    *,
    mode: SearchMode = SearchMode.HYBRID_CROSS,
    top_k: int = 10,
    filters: MemorySearchFilters | None = None,
    debug: bool = False,
) -> MemorySearchResponse:
    """Main retrieval entry point."""
    config = get_memory_v2_config()
    session_ids = filters.session_ids if filters else None

    dense_candidates: list[dict[str, Any]] = []
    keyword_candidates: list[dict[str, Any]] = []

    if mode in (SearchMode.DENSE_DISTILLED, SearchMode.HYBRID_CROSS):
        dense_candidates = dense_search(
            query,
            top_k=config.dense_top_k,
            session_ids=session_ids,
        )

    if mode in (SearchMode.KEYWORD_VERBATIM, SearchMode.HYBRID_CROSS):
        keyword_candidates = keyword_search(
            query,
            top_k=config.keyword_top_k,
            session_ids=session_ids,
        )

    if mode == SearchMode.DENSE_DISTILLED:
        fused = [
            {**c, "fused_score": c.get("dense_score", 0.0)}
            for c in dense_candidates[:top_k]
        ]
    elif mode == SearchMode.KEYWORD_VERBATIM:
        fused = [
            {**c, "fused_score": c.get("keyword_score", 0.0)}
            for c in keyword_candidates[:top_k]
        ]
    else:
        fusion_method = (getattr(config, "fusion_method", "weighted_sum") or "weighted_sum").lower()
        if fusion_method == "weighted_sum":
            fused = weighted_sum_fusion(
                dense_candidates,
                keyword_candidates,
                dense_weight=getattr(config, "dense_weight", 0.3),
                keyword_weight=getattr(config, "keyword_weight", 0.7),
                top_k=top_k,
            )
        else:
            # Default fallback: original RRF fusion
            fused = rrf_fusion(
                dense_candidates,
                keyword_candidates,
                k=getattr(config, "rrf_k", 60),
                top_k=top_k,
            )

    if filters and filters.min_fused_score:
        fused = [f for f in fused if f.get("fused_score", 0) >= filters.min_fused_score]

    exchanges_repo = ExchangesRepo()
    hits: list[MemoryHit] = []

    for rank, item in enumerate(fused, 1):
        exchange_id = item["exchange_id"]
        session_id = item.get("session_id", "")
        ply_start = item.get("ply_start", 0)
        ply_end = item.get("ply_end", 0)

        verbatim_snippet = ""
        if not session_id or not ply_start:
            ex_row = exchanges_repo.get_by_backref(session_id, ply_start, ply_end) if session_id else None
            if ex_row is None:
                for field_name in ("exchange_id",):
                    lookup = exchanges_repo.get_by_backref(session_id, ply_start, ply_end)
                    if lookup:
                        ex_row = lookup
                        break
            if ex_row:
                session_id = ex_row.get("session_id", session_id)
                ply_start = ex_row.get("ply_start", ply_start)
                ply_end = ex_row.get("ply_end", ply_end)
                verbatim_snippet = ex_row.get("verbatim_snippet", "")
        else:
            ex_row = exchanges_repo.get_by_backref(session_id, ply_start, ply_end)
            if ex_row:
                verbatim_snippet = ex_row.get("verbatim_snippet", "")

        rooms = None
        raw_rooms = item.get("room_assignments")
        if raw_rooms and isinstance(raw_rooms, list):
            rooms = [
                RoomAssignment.from_dict(r) if isinstance(r, dict) else r
                for r in raw_rooms
            ]

        files_touched = item.get("files_touched")
        if isinstance(files_touched, str):
            import json
            files_touched = json.loads(files_touched)

        hits.append(MemoryHit(
            rank=rank,
            session_id=session_id,
            exchange_id=exchange_id,
            ply_start=ply_start,
            ply_end=ply_end,
            verbatim_snippet=verbatim_snippet,
            object_id=item.get("object_id"),
            rooms=rooms,
            files_touched=files_touched,
            scores={
                "dense": item.get("dense_score", 0.0),
                "keyword": item.get("keyword_score", 0.0),
                "fused": item.get("fused_score", 0.0),
            },
        ))

    debug_info = None
    if debug:
        fusion_method = (getattr(config, "fusion_method", "weighted_sum") or "weighted_sum").lower()
        debug_info = MemorySearchDebug(
            dense_candidates=[
                {"object_id": c.get("object_id"), "exchange_id": c["exchange_id"], "score": c.get("dense_score", 0)}
                for c in dense_candidates[:20]
            ],
            keyword_candidates=[
                {"exchange_id": c["exchange_id"], "score": c.get("keyword_score", 0)}
                for c in keyword_candidates[:20]
            ],
            fusion=(
                {"method": fusion_method, "dense_weight": getattr(config, "dense_weight", None), "keyword_weight": getattr(config, "keyword_weight", None)}
                if mode == SearchMode.HYBRID_CROSS
                else {"method": mode.value}
            ),
        )

    return MemorySearchResponse(
        query=query,
        mode=mode,
        top_k=top_k,
        hits=hits,
        debug=debug_info,
    )
