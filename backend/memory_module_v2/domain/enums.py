from __future__ import annotations

from enum import Enum


class SearchMode(str, Enum):
    DENSE_DISTILLED = "dense_distilled"
    KEYWORD_VERBATIM = "keyword_verbatim"
    HYBRID_CROSS = "hybrid_cross"


class RoomType(str, Enum):
    FILE = "file"
    CONCEPT = "concept"
    WORKFLOW = "workflow"
