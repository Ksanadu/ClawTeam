"""memory_module_v2 configuration – reads from environment with sane defaults.

Unified memory backend selection via MEMORY_BACKEND env var:
  - "off"  : no long-term memory (default)
  - "v1"   : legacy Chroma / MEMORY.md RAG
  - "v2"   : structured distillation + hybrid retrieval (this module)

When MEMORY_BACKEND=v2, injection strategy is controlled by MEMORY_V2_INJECT:
  - "tool"   : register search_memory as an agent tool (default, agent decides)
  - "always" : force-inject retrieval context every turn
  - "off"    : v2 enabled for API but no auto-injection
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip() or default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


MemoryBackend = Literal["off", "v1", "v2"]
MemoryV2Inject = Literal["tool", "always", "off"]


def get_memory_backend() -> MemoryBackend:
    """Unified switch: which memory backend is active."""
    raw = os.getenv("MEMORY_BACKEND", "").strip().lower()
    if raw in ("v1", "v2", "off"):
        return raw  # type: ignore[return-value]
    # Legacy compat: check old env vars
    if _env_bool("MEMORY_V2_ENABLED", False):
        return "v2"
    return "off"


def get_memory_v2_inject_mode() -> MemoryV2Inject:
    raw = os.getenv("MEMORY_V2_INJECT", "").strip().lower()
    if raw in ("tool", "always", "off"):
        return raw  # type: ignore[return-value]
    # Legacy compat
    if _env_bool("MEMORY_V2_AUTO_INJECT", False):
        return "always"
    return "tool"


@dataclass(frozen=True)
class MemoryV2Config:
    # BM25 index storage
    bm25_index_dir: str = field(default_factory=lambda: _env("BM25_INDEX_DIR", "./storage/memory_v2/bm25"))
    bm25_rebuild_on_start: bool = field(default_factory=lambda: _env_bool("BM25_REBUILD_ON_START", False))
    bm25_sharding: str = field(default_factory=lambda: _env("BM25_SHARDING", "session"))
    bm25_max_docs: int = field(default_factory=lambda: _env_int("BM25_MAX_DOCS", 50_000))
    bm25_window_days: int = field(default_factory=lambda: _env_int("BM25_WINDOW_DAYS", 30))
    bm25_rebuild_min_new_docs: int = field(default_factory=lambda: _env_int("BM25_REBUILD_MIN_NEW_DOCS", 500))
    bm25_rebuild_min_seconds: int = field(default_factory=lambda: _env_int("BM25_REBUILD_MIN_SECONDS", 600))
    bm25_use_facets: bool = field(default_factory=lambda: _env_bool("BM25_USE_FACETS_IN_CORPUS", True))

    # exchange segmenter
    min_exchange_chars: int = 100
    max_ply_len: int = 20
    min_assistant_chars: int = 80

    # retrieval defaults
    dense_top_k: int = 200
    keyword_top_k: int = 200
    final_top_k: int = 10
    rrf_k: int = 60

    # hybrid fusion
    # - "rrf" keeps the original behavior (RRF(k=config.rrf_k))
    # - "weighted_sum" uses score normalization + user-defined weights
    fusion_method: str = field(default_factory=lambda: _env("MEMORY_V2_FUSION_METHOD", "weighted_sum").lower())
    dense_weight: float = field(default_factory=lambda: _env_float("MEMORY_V2_DENSE_WEIGHT", 0.3))
    keyword_weight: float = field(default_factory=lambda: _env_float("MEMORY_V2_KEYWORD_WEIGHT", 0.7))

    # injection
    inject_top_k: int = field(default_factory=lambda: _env_int("MEMORY_V2_INJECT_TOP_K", 3))

    @property
    def enabled(self) -> bool:
        return get_memory_backend() == "v2"

    @property
    def inject_mode(self) -> MemoryV2Inject:
        return get_memory_v2_inject_mode()

    @property
    def bm25_index_path(self) -> Path:
        return Path(self.bm25_index_dir)


def get_memory_v2_config() -> MemoryV2Config:
    return MemoryV2Config()
