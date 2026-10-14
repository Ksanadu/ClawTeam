"""Operational utilities: incremental detection, background rebuild, health checks."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from ..ingest.session_reader import get_session_updated_at, list_session_ids
from ..retrieval.keyword import (
    get_or_build_shard,
    get_shard_stats,
    mark_shard_dirty,
    should_rebuild,
)
from ..storage.repos import ExchangesRepo, ObjectsRepo
from .config import get_memory_v2_config

logger = logging.getLogger(__name__)

_session_timestamps: dict[str, float] = {}


def detect_dirty_sessions() -> list[str]:
    """Detect sessions that have been updated since last distillation."""
    dirty: list[str] = []
    for sid in list_session_ids():
        updated_at = get_session_updated_at(sid)
        if updated_at is None:
            continue
        last_known = _session_timestamps.get(sid, 0)
        if updated_at > last_known:
            dirty.append(sid)
    return dirty


def mark_session_processed(session_id: str) -> None:
    updated_at = get_session_updated_at(session_id)
    if updated_at:
        _session_timestamps[session_id] = updated_at


def trigger_incremental_rebuild(*, force: bool = False) -> dict[str, Any]:
    """Check dirty shards and rebuild BM25 if thresholds met."""
    config = get_memory_v2_config()
    result: dict[str, Any] = {"rebuilt": False, "reason": ""}

    if force or should_rebuild("default"):
        shard = get_or_build_shard("default", force_rebuild=True)
        result["rebuilt"] = True
        result["reason"] = "force" if force else "threshold_met"
        result["corpus_size"] = shard.corpus_size
    else:
        result["reason"] = "no_rebuild_needed"

    return result


def get_health_stats() -> dict[str, Any]:
    """Return health check and statistics."""
    exchanges_repo = ExchangesRepo()
    objects_repo = ObjectsRepo()

    try:
        exchanges_count = exchanges_repo.count()
        objects_count = objects_repo.count()
        db_status = "ok"
    except Exception as exc:
        exchanges_count = -1
        objects_count = -1
        db_status = f"error: {exc}"

    bm25_stats = get_shard_stats("default")
    dirty_sessions = detect_dirty_sessions()

    return {
        "db_status": db_status,
        "exchanges_count": exchanges_count,
        "objects_count": objects_count,
        "bm25": bm25_stats,
        "dirty_sessions_count": len(dirty_sessions),
        "known_sessions": len(_session_timestamps),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
