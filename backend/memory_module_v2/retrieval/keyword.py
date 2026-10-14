"""BM25 keyword retrieval on verbatim exchanges (rank-bm25, windowed + sharded)."""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from ..service.config import get_memory_v2_config
from ..storage.repos import ExchangesRepo
from .tokenizer import tokenize

logger = logging.getLogger(__name__)

_bm25_cache: dict[str, _BM25Shard] = {}


class _BM25Shard:
    """Holds a BM25 index for a set of exchanges."""

    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.exchange_ids: list[str] = []
        self.docs: list[list[str]] = []
        self.built_at: float = 0
        self.corpus_size: int = 0
        self.dirty_count: int = 0

    def build(self, corpus: list[dict[str, Any]], *, use_facets: bool = True) -> None:
        self.exchange_ids = []
        self.docs = []

        for item in corpus:
            exchange_id = item["exchange_id"]
            bm25_text = item.get("verbatim_snippet", "") or item.get("verbatim_text", "")

            if use_facets:
                obj = _get_object_for_exchange(exchange_id)
                if obj:
                    files = obj.get("files_touched", [])
                    if isinstance(files, str):
                        files = json.loads(files)
                    rooms = obj.get("room_assignments", [])
                    if isinstance(rooms, str):
                        rooms = json.loads(rooms)

                    if files:
                        bm25_text += "\nFILES: " + " ".join(files)
                    if rooms:
                        room_tokens = []
                        for r in rooms:
                            room_tokens.append(r.get("room_key", ""))
                            room_tokens.append(r.get("room_label", ""))
                        bm25_text += "\nROOMS: " + " ".join(room_tokens)

            tokens = tokenize(bm25_text)
            if tokens:
                self.exchange_ids.append(exchange_id)
                self.docs.append(tokens)

        if self.docs:
            self.bm25 = BM25Okapi(self.docs)
        else:
            self.bm25 = None

        self.built_at = time.time()
        self.corpus_size = len(self.docs)

    def search(self, query: str, top_k: int = 200) -> list[dict[str, Any]]:
        # `self.docs` is only populated during `build()`.
        # When loading from cache (`load()`), we intentionally don't store full tokenized docs.
        # bm25 + exchange_ids are sufficient to compute scores and map to exchange_id.
        if self.bm25 is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        scored = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = []
        for idx, score in scored:
            if score <= 0:
                continue
            results.append({
                "exchange_id": self.exchange_ids[idx],
                "keyword_score": float(score),
            })
        return results

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        meta = {
            "built_at": self.built_at,
            "corpus_size": self.corpus_size,
            "exchange_ids": self.exchange_ids,
        }
        (path / "index_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )

    def load(self, path: Path) -> bool:
        pkl_path = path / "bm25.pkl"
        meta_path = path / "index_meta.json"
        if not pkl_path.exists() or not meta_path.exists():
            return False
        try:
            with open(pkl_path, "rb") as f:
                self.bm25 = pickle.load(f)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.built_at = meta.get("built_at", 0)
            self.corpus_size = meta.get("corpus_size", 0)
            self.exchange_ids = meta.get("exchange_ids", [])
            return True
        except Exception as exc:
            logger.warning("Failed to load BM25 cache: %s", exc)
            return False


def _get_object_for_exchange(exchange_id: str) -> dict[str, Any] | None:
    from ..storage.repos import ObjectsRepo
    return ObjectsRepo().get_by_exchange_id(exchange_id)


def get_or_build_shard(
    shard_key: str = "default",
    *,
    force_rebuild: bool = False,
    session_ids: list[str] | None = None,
) -> _BM25Shard:
    """Get or build a BM25 shard."""
    config = get_memory_v2_config()

    if shard_key in _bm25_cache and not force_rebuild:
        shard = _bm25_cache[shard_key]
        if shard.bm25 is not None:
            return shard

    shard = _BM25Shard()

    cache_path = config.bm25_index_path / shard_key
    if not force_rebuild and shard.load(cache_path):
        _bm25_cache[shard_key] = shard
        return shard

    repo = ExchangesRepo()
    corpus = repo.fetch_bm25_corpus(
        window_days=config.bm25_window_days,
        max_docs=config.bm25_max_docs,
        session_ids=session_ids,
    )

    shard.build(corpus, use_facets=config.bm25_use_facets)

    try:
        shard.save(cache_path)
    except Exception as exc:
        logger.warning("Failed to save BM25 cache: %s", exc)

    _bm25_cache[shard_key] = shard
    return shard


def keyword_search(
    query: str,
    top_k: int = 200,
    *,
    session_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """BM25 keyword search on verbatim exchanges."""
    config = get_memory_v2_config()

    if config.bm25_sharding == "session" and session_ids:
        all_results: list[dict[str, Any]] = []
        for sid in session_ids:
            shard = get_or_build_shard(f"session_{sid}", session_ids=[sid])
            all_results.extend(shard.search(query, top_k))
        all_results.sort(key=lambda x: x["keyword_score"], reverse=True)
        return all_results[:top_k]

    shard = get_or_build_shard("default", session_ids=session_ids)
    return shard.search(query, top_k)


def mark_shard_dirty(shard_key: str = "default") -> None:
    if shard_key in _bm25_cache:
        _bm25_cache[shard_key].dirty_count += 1


def should_rebuild(shard_key: str = "default") -> bool:
    config = get_memory_v2_config()
    if shard_key not in _bm25_cache:
        return True
    shard = _bm25_cache[shard_key]
    if shard.dirty_count >= config.bm25_rebuild_min_new_docs:
        return True
    if shard.built_at and (time.time() - shard.built_at) > config.bm25_rebuild_min_seconds:
        if shard.dirty_count > 0:
            return True
    return False


def get_shard_stats(shard_key: str = "default") -> dict[str, Any]:
    if shard_key not in _bm25_cache:
        return {"status": "not_loaded", "shard_key": shard_key}
    shard = _bm25_cache[shard_key]
    return {
        "shard_key": shard_key,
        "status": "loaded" if shard.bm25 is not None else "empty",
        "corpus_size": shard.corpus_size,
        "built_at": datetime.fromtimestamp(shard.built_at, tz=timezone.utc).isoformat() if shard.built_at else None,
        "dirty_count": shard.dirty_count,
    }
