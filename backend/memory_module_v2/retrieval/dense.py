"""Dense retrieval via pgvector on memory_objects."""

from __future__ import annotations

import logging
from typing import Any

from config import get_settings
from graph.llm import build_embedding_config_from_settings, get_embedding_model

from ..storage.repos import ObjectsRepo

logger = logging.getLogger(__name__)


def dense_search(
    query: str,
    top_k: int = 200,
    *,
    session_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Dense retrieval: embed query, then pgvector cosine similarity search."""
    settings = get_settings()
    emb_config = build_embedding_config_from_settings(settings)
    embedding_model = get_embedding_model(emb_config)

    query_embedding = embedding_model.embed_query(query)

    repo = ObjectsRepo()
    results = repo.dense_search(
        query_embedding,
        top_k=top_k,
        session_ids=session_ids,
    )

    return [
        {
            "object_id": r["object_id"],
            "exchange_id": r["exchange_id"],
            "session_id": r["session_id"],
            "ply_start": r["ply_start"],
            "ply_end": r["ply_end"],
            "dense_score": r["dense_score"],
            "distill_text": r.get("distill_text", ""),
            "room_assignments": r.get("room_assignments", []),
            "files_touched": r.get("files_touched", []),
        }
        for r in results
    ]
