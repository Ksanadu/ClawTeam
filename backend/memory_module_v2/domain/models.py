from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .enums import RoomType, SearchMode


@dataclass
class NormalizedMessage:
    msg_index: int
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class Exchange:
    exchange_id: str
    session_id: str
    ply_start: int
    ply_end: int
    messages: list[NormalizedMessage] = field(default_factory=list)
    verbatim_text: str = ""
    verbatim_snippet: str = ""
    message_count: int = 0
    has_substantive_assistant: bool = False


@dataclass
class RoomAssignment:
    room_type: RoomType
    room_key: str
    room_label: str
    relevance: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_type": self.room_type.value,
            "room_key": self.room_key,
            "room_label": self.room_label,
            "relevance": self.relevance,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RoomAssignment:
        return cls(
            room_type=RoomType(d["room_type"]),
            room_key=d["room_key"],
            room_label=d["room_label"],
            relevance=d.get("relevance", 1.0),
        )


@dataclass
class DistilledObject:
    object_id: str
    exchange_id: str
    session_id: str
    ply_start: int
    ply_end: int
    exchange_core: str
    specific_context: str
    room_assignments: list[RoomAssignment] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    distill_text: str = ""
    distilled_at: datetime | None = None
    distill_provider: str = ""
    distill_model: str = ""
    embedding: list[float] | None = None


@dataclass
class MemoryHit:
    rank: int
    session_id: str
    exchange_id: str
    ply_start: int
    ply_end: int
    verbatim_snippet: str
    object_id: str | None = None
    rooms: list[RoomAssignment] | None = None
    files_touched: list[str] | None = None
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class MemorySearchFilters:
    session_ids: list[str] | None = None
    time_range: dict[str, str] | None = None
    room_keys: list[str] | None = None
    files: list[str] | None = None
    min_fused_score: float | None = None


@dataclass
class MemorySearchDebug:
    dense_candidates: list[dict[str, Any]] = field(default_factory=list)
    keyword_candidates: list[dict[str, Any]] = field(default_factory=list)
    fusion: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemorySearchResponse:
    query: str
    mode: SearchMode
    top_k: int
    hits: list[MemoryHit] = field(default_factory=list)
    debug: MemorySearchDebug | None = None


@dataclass
class ExchangeEvidence:
    session_id: str
    ply_start: int
    ply_end: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    verbatim_snippet: str = ""


@dataclass
class DistillSessionResult:
    session_id: str
    exchanges_total: int = 0
    exchanges_new: int = 0
    objects_created: int = 0
    objects_updated: int = 0
    objects_skipped: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)
