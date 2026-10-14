"""Postgres connection & schema initialization for memory_module_v2."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import psycopg  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_SCHEMA_SQL = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")


def _get_dsn() -> str:
    dsn = os.getenv("POSTGRES_DSN", "").strip()
    if dsn:
        return dsn
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    db = os.getenv("POSTGRES_DB", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def get_connection(*, autocommit: bool = False) -> psycopg.Connection[Any]:
    dsn = _get_dsn()
    return psycopg.connect(dsn, autocommit=autocommit)


def ensure_schema() -> None:
    """Create the memory_v2 schema and tables if they don't exist."""
    try:
        with get_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                for statement in _split_statements(_SCHEMA_SQL):
                    cur.execute(statement)
        logger.info("memory_v2 schema initialized successfully")
    except Exception as exc:
        logger.error(
            "Failed to initialize memory_v2 schema. "
            "Make sure PostgreSQL is running and pgvector extension is available. "
            "Error: %s",
            exc,
        )
        raise


def _split_statements(sql: str) -> list[str]:
    """Split a SQL file into individual statements."""
    statements: list[str] = []
    current: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        current.append(line)
        if stripped.endswith(";"):
            statements.append("\n".join(current))
            current = []
    if current:
        statements.append("\n".join(current))
    return [s for s in statements if s.strip()]
