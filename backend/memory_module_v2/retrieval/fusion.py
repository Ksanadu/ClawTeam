"""Fusion strategies: RRF and weighted_sum for combining dense + keyword results."""

from __future__ import annotations

from typing import Any


def rrf_fusion(
    dense_candidates: list[dict[str, Any]],
    keyword_candidates: list[dict[str, Any]],
    *,
    k: int = 60,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion (RRF) combining dense and keyword results."""
    scores: dict[str, float] = {}
    meta: dict[str, dict[str, Any]] = {}

    for rank, item in enumerate(dense_candidates):
        eid = item["exchange_id"]
        scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank + 1)
        if eid not in meta:
            meta[eid] = dict(item)
        meta[eid]["dense_score"] = item.get("dense_score", 0.0)

    for rank, item in enumerate(keyword_candidates):
        eid = item["exchange_id"]
        scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank + 1)
        if eid not in meta:
            meta[eid] = dict(item)
        meta[eid]["keyword_score"] = item.get("keyword_score", 0.0)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for eid, fused_score in ranked:
        entry = meta.get(eid, {})
        entry["exchange_id"] = eid
        entry["fused_score"] = fused_score
        entry.setdefault("dense_score", 0.0)
        entry.setdefault("keyword_score", 0.0)
        results.append(entry)

    return results


def weighted_sum_fusion(
    dense_candidates: list[dict[str, Any]],
    keyword_candidates: list[dict[str, Any]],
    *,
    dense_weight: float = 0.6,
    keyword_weight: float = 0.4,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Weighted sum fusion with normalized scores."""
    dense_max = max((c.get("dense_score", 0) for c in dense_candidates), default=1.0) or 1.0
    keyword_max = max((c.get("keyword_score", 0) for c in keyword_candidates), default=1.0) or 1.0

    scores: dict[str, float] = {}
    meta: dict[str, dict[str, Any]] = {}

    for item in dense_candidates:
        eid = item["exchange_id"]
        normalized = item.get("dense_score", 0.0) / dense_max
        scores[eid] = scores.get(eid, 0.0) + dense_weight * normalized
        if eid not in meta:
            meta[eid] = dict(item)
        meta[eid]["dense_score"] = item.get("dense_score", 0.0)

    for item in keyword_candidates:
        eid = item["exchange_id"]
        normalized = item.get("keyword_score", 0.0) / keyword_max
        scores[eid] = scores.get(eid, 0.0) + keyword_weight * normalized
        if eid not in meta:
            meta[eid] = dict(item)
        meta[eid]["keyword_score"] = item.get("keyword_score", 0.0)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for eid, fused_score in ranked:
        entry = meta.get(eid, {})
        entry["exchange_id"] = eid
        entry["fused_score"] = fused_score
        entry.setdefault("dense_score", 0.0)
        entry.setdefault("keyword_score", 0.0)
        results.append(entry)

    return results
