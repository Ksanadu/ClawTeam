"""Repositories for memory_exchanges and memory_objects."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg  # type: ignore[import-untyped]

from ..domain.models import DistilledObject, Exchange, RoomAssignment
from .pg import get_connection

logger = logging.getLogger(__name__)


class ExchangesRepo:
    """CRUD for memory_v2.memory_exchanges."""

    def upsert(self, exchange: Exchange) -> None:
        sql = """
            INSERT INTO memory_v2.memory_exchanges
                (exchange_id, session_id, ply_start, ply_end,
                 verbatim_text, verbatim_snippet, message_count,
                 has_substantive_assistant, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (exchange_id) DO UPDATE SET
                verbatim_text = EXCLUDED.verbatim_text,
                verbatim_snippet = EXCLUDED.verbatim_snippet,
                message_count = EXCLUDED.message_count,
                has_substantive_assistant = EXCLUDED.has_substantive_assistant,
                updated_at = now()
        """
        with get_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    exchange.exchange_id,
                    exchange.session_id,
                    exchange.ply_start,
                    exchange.ply_end,
                    exchange.verbatim_text,
                    exchange.verbatim_snippet,
                    exchange.message_count,
                    exchange.has_substantive_assistant,
                ))

    def upsert_batch(self, exchanges: list[Exchange]) -> None:
        if not exchanges:
            return
        sql = """
            INSERT INTO memory_v2.memory_exchanges
                (exchange_id, session_id, ply_start, ply_end,
                 verbatim_text, verbatim_snippet, message_count,
                 has_substantive_assistant, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (exchange_id) DO UPDATE SET
                verbatim_text = EXCLUDED.verbatim_text,
                verbatim_snippet = EXCLUDED.verbatim_snippet,
                message_count = EXCLUDED.message_count,
                has_substantive_assistant = EXCLUDED.has_substantive_assistant,
                updated_at = now()
        """
        with get_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                for ex in exchanges:
                    cur.execute(sql, (
                        ex.exchange_id,
                        ex.session_id,
                        ex.ply_start,
                        ex.ply_end,
                        ex.verbatim_text,
                        ex.verbatim_snippet,
                        ex.message_count,
                        ex.has_substantive_assistant,
                    ))

    def exists(self, exchange_id: str) -> bool:
        sql = "SELECT 1 FROM memory_v2.memory_exchanges WHERE exchange_id = %s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (exchange_id,))
                return cur.fetchone() is not None

    def get_by_session(self, session_id: str) -> list[dict[str, Any]]:
        sql = """
            SELECT exchange_id, session_id, ply_start, ply_end,
                   verbatim_text, verbatim_snippet, message_count,
                   has_substantive_assistant, created_at
            FROM memory_v2.memory_exchanges
            WHERE session_id = %s
            ORDER BY ply_start
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (session_id,))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_exchange_ids_for_session(self, session_id: str) -> set[str]:
        sql = "SELECT exchange_id FROM memory_v2.memory_exchanges WHERE session_id = %s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (session_id,))
                return {row[0] for row in cur.fetchall()}

    def fetch_bm25_corpus(
        self,
        *,
        window_days: int | None = None,
        max_docs: int | None = None,
        session_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch exchanges for BM25 index construction."""
        conditions: list[str] = []
        params: list[Any] = []

        if window_days:
            cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
            conditions.append("created_at >= %s")
            params.append(cutoff)

        if session_ids:
            conditions.append("session_id = ANY(%s)")
            params.append(session_ids)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        limit = ""
        if max_docs:
            limit = f"LIMIT {max_docs}"

        sql = f"""
            SELECT exchange_id, session_id, ply_start, ply_end,
                   verbatim_text, verbatim_snippet, created_at
            FROM memory_v2.memory_exchanges
            {where}
            ORDER BY created_at DESC
            {limit}
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_by_backref(self, session_id: str, ply_start: int, ply_end: int) -> dict[str, Any] | None:
        sql = """
            SELECT exchange_id, session_id, ply_start, ply_end,
                   verbatim_text, verbatim_snippet
            FROM memory_v2.memory_exchanges
            WHERE session_id = %s AND ply_start = %s AND ply_end = %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (session_id, ply_start, ply_end))
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [d[0] for d in cur.description]
                return dict(zip(cols, row))

    def count(self) -> int:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM memory_v2.memory_exchanges")
                return cur.fetchone()[0]


class ObjectsRepo:
    """CRUD for memory_v2.memory_objects."""

    def upsert(self, obj: DistilledObject) -> None:
        sql = """
            INSERT INTO memory_v2.memory_objects
                (object_id, exchange_id, session_id, ply_start, ply_end,
                 exchange_core, specific_context, distill_text,
                 room_assignments, files_touched,
                 distill_provider, distill_model, distilled_at,
                 embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (object_id) DO UPDATE SET
                exchange_core = EXCLUDED.exchange_core,
                specific_context = EXCLUDED.specific_context,
                distill_text = EXCLUDED.distill_text,
                room_assignments = EXCLUDED.room_assignments,
                files_touched = EXCLUDED.files_touched,
                distill_provider = EXCLUDED.distill_provider,
                distill_model = EXCLUDED.distill_model,
                distilled_at = EXCLUDED.distilled_at,
                embedding = EXCLUDED.embedding
        """
        rooms_json = json.dumps([r.to_dict() for r in obj.room_assignments])
        files_json = json.dumps(obj.files_touched)
        embedding_str = _format_vector(obj.embedding) if obj.embedding else None

        with get_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    obj.object_id,
                    obj.exchange_id,
                    obj.session_id,
                    obj.ply_start,
                    obj.ply_end,
                    obj.exchange_core,
                    obj.specific_context,
                    obj.distill_text,
                    rooms_json,
                    files_json,
                    obj.distill_provider,
                    obj.distill_model,
                    obj.distilled_at or datetime.now(timezone.utc),
                    embedding_str,
                ))

    def update_embedding(self, object_id: str, embedding: list[float]) -> None:
        sql = "UPDATE memory_v2.memory_objects SET embedding = %s WHERE object_id = %s"
        with get_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (_format_vector(embedding), object_id))

    def dense_search(
        self,
        query_embedding: list[float],
        top_k: int = 200,
        *,
        session_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Cosine-distance similarity search against memory_objects."""
        conditions = ["embedding IS NOT NULL"]
        params: list[Any] = []

        if session_ids:
            conditions.append("session_id = ANY(%s)")
            params.append(session_ids)

        where = "WHERE " + " AND ".join(conditions)
        emb_str = _format_vector(query_embedding)

        sql = f"""
            SELECT object_id, exchange_id, session_id, ply_start, ply_end,
                   exchange_core, specific_context, distill_text,
                   room_assignments, files_touched,
                   1.0 / (1.0 + (embedding <=> %s)) AS dense_score
            FROM memory_v2.memory_objects
            {where}
            ORDER BY embedding <=> %s
            LIMIT %s
        """
        params_full = [emb_str] + params
        params_order = [emb_str, top_k]

        combined_sql = f"""
            SELECT object_id, exchange_id, session_id, ply_start, ply_end,
                   exchange_core, specific_context, distill_text,
                   room_assignments, files_touched,
                   1.0 / (1.0 + (embedding <=> %s)) AS dense_score
            FROM memory_v2.memory_objects
            {where}
            ORDER BY embedding <=> %s
            LIMIT %s
        """
        all_params = [emb_str] + params + [emb_str, top_k]

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(combined_sql, all_params)
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, row)) for row in cur.fetchall()]
                for row in rows:
                    if isinstance(row.get("room_assignments"), str):
                        row["room_assignments"] = json.loads(row["room_assignments"])
                    if isinstance(row.get("files_touched"), str):
                        row["files_touched"] = json.loads(row["files_touched"])
                return rows

    def get_by_exchange_id(self, exchange_id: str) -> dict[str, Any] | None:
        sql = """
            SELECT object_id, exchange_id, session_id, ply_start, ply_end,
                   exchange_core, specific_context, distill_text,
                   room_assignments, files_touched
            FROM memory_v2.memory_objects
            WHERE exchange_id = %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (exchange_id,))
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [d[0] for d in cur.description]
                return dict(zip(cols, row))

    def count(self) -> int:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM memory_v2.memory_objects")
                return cur.fetchone()[0]

    def objects_without_embedding(self, limit: int = 100) -> list[dict[str, Any]]:
        sql = """
            SELECT object_id, distill_text
            FROM memory_v2.memory_objects
            WHERE embedding IS NULL
            LIMIT %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (limit,))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]


def _format_vector(vec: list[float]) -> str:
    """Format a Python list as a pgvector literal: '[0.1,0.2,...]'"""
    return "[" + ",".join(str(v) for v in vec) + "]"
