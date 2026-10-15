"""Offline regressions for durable distillation completion and advisory locking."""
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from backend.memory_module_v2.distill import distiller
from backend.memory_module_v2.domain.models import NormalizedMessage
from backend.memory_module_v2.ingest import session_reader
from backend.memory_module_v2.service import api
from backend.memory_module_v2.service.config import MemoryV2Config
from backend.memory_module_v2.storage import pg, repos


class FakeAdvisoryDatabase:
    """Connection-owned locks, not a replacement implementation of the service."""

    def __init__(self):
        self.connections = []
        self.locks = {}
        self.guard = threading.Lock()
        self.waiting = threading.Event()
        self.fail_acquire = False
        self.fail_unlock = False

    def connect(self, *, autocommit=False):
        assert autocommit is True, "Model time must not hold a transaction open"
        conn = FakeLockConnection(self)
        with self.guard:
            self.connections.append(conn)
        return conn

    def owns_lock(self):
        return any(
            conn.key is not None and conn.thread_id == threading.get_ident()
            for conn in self.connections
        )

    def assert_released(self):
        assert self.connections
        assert all(conn.closed and conn.key is None for conn in self.connections)
        assert all(not lock.locked() for lock in self.locks.values())


class FakeLockConnection:
    def __init__(self, db):
        self.db = db
        self.closed = False
        self.key = None
        self.thread_id = threading.get_ident()
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        # A dedicated connection must close even when acquisition/unlock fails.
        if self.key is not None:
            self.db.locks[self.key].release()
            self.key = None
        self.closed = True

    def cursor(self):
        return FakeLockCursor(self)


class FakeLockCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def execute(self, sql, params):
        conn = self.conn
        conn.executed.append((sql, params))
        key, = params
        if sql == "SELECT pg_advisory_lock(%s)":
            if conn.db.fail_acquire:
                raise RuntimeError("lock acquisition failed")
            with conn.db.guard:
                lock = conn.db.locks.setdefault(key, threading.Lock())
            if not lock.acquire(blocking=False):
                conn.db.waiting.set()
                assert lock.acquire(timeout=5), "Advisory lock was not released"
            conn.key = key
        elif sql == "SELECT pg_advisory_unlock(%s)":
            assert conn.key == key
            if conn.db.fail_unlock:
                raise RuntimeError("unlock failed")
            conn.db.locks[key].release()
            conn.key = None
        else:
            pytest.fail(f"Unexpected SQL in lock fake: {sql}")


class FakeExchangesRepo:
    def __init__(self):
        self.exchanges = {}

    def get_exchange_ids_for_session(self, session_id):
        return {
            key for key, exchange in self.exchanges.items()
            if exchange.session_id == session_id
        }

    def upsert_batch(self, exchanges):
        self.exchanges.update((ex.exchange_id, ex) for ex in exchanges)


class FakeObjectsRepo(repos.ObjectsRepo):
    """Use the real advisory-lock method, with in-memory object persistence."""

    def __init__(self, db):
        self.db = db
        self.objects = {}
        self.fail_next_save = False

    def get_exchange_ids_for_session(self, session_id):
        assert self.db.owns_lock(), "Completion must be checked after claiming"
        return {
            obj.exchange_id for obj in self.objects.values()
            if obj.session_id == session_id
        }

    def upsert(self, obj):
        assert self.db.owns_lock()
        if self.fail_next_save:
            self.fail_next_save = False
            raise RuntimeError("temporary object save failure")
        self.objects[obj.object_id] = obj


class FakeModel:
    def __init__(self):
        self.calls = []
        self.outcomes = []
        self.on_invoke = None

    def invoke(self, messages):
        self.calls.append(messages)
        if self.on_invoke is not None:
            self.on_invoke()
        if self.outcomes:
            error = self.outcomes.pop(0)
            if error is not None:
                raise error
        return SimpleNamespace(content=json.dumps({
            "exchange_core": f"distilled exchange {len(self.calls)}",
            "specific_context": "regression test context",
            "rooms": [],
        }))


@pytest.fixture(autouse=True)
def forbid_real_database(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Regression tests must never connect to a real database")

    monkeypatch.setattr(pg.psycopg, "connect", forbidden)


@pytest.fixture()
def harness(monkeypatch):
    db = FakeAdvisoryDatabase()
    exchanges = FakeExchangesRepo()
    objects = FakeObjectsRepo(db)
    model = FakeModel()
    messages = [
        NormalizedMessage(0, "user", "Question about a retry"),
        NormalizedMessage(1, "assistant", "A substantive answer to retain"),
    ]
    monkeypatch.setattr(repos, "get_connection", db.connect)
    monkeypatch.setattr(api, "exchanges_repo", exchanges)
    monkeypatch.setattr(api, "objects_repo", objects)
    monkeypatch.setattr(api, "read_session", lambda session_id: messages)
    monkeypatch.setattr(api, "get_memory_v2_config", lambda: MemoryV2Config(
        min_exchange_chars=1, min_assistant_chars=1,
    ))
    monkeypatch.setattr(api, "get_settings", lambda: object())
    monkeypatch.setattr(api, "build_embedding_config_from_settings", lambda settings: object())
    monkeypatch.setattr(api, "get_embedding_model", lambda config: SimpleNamespace(
        embed_query=lambda text: [0.25, 0.75],
    ))
    # Exercise the real distiller and its inference-error wrapping, not an
    # extracted AST function or a fabricated module in sys.modules.
    monkeypatch.setattr(distiller, "_get_distill_llm", lambda: (model, "fake", "test-model"))
    return SimpleNamespace(
        db=db, exchanges=exchanges, objects=objects, model=model, messages=messages,
    )


@pytest.mark.parametrize("failure", ["model", "object_save", "worker_crash"])
def test_failure_before_object_completion_is_retryable(harness, failure):
    if failure == "object_save":
        harness.objects.fail_next_save = True
    else:
        error = RuntimeError("temporary model failure") if failure == "model" else SystemExit("worker stopped")
        harness.model.outcomes = [error]

    if failure == "worker_crash":
        with pytest.raises(SystemExit, match="worker stopped"):
            api.distill_session("session-a")
    else:
        first = api.distill_session("session-a")
        assert first.exchanges_new == 1
        assert first.objects_created == 0
        assert len(first.errors) == 1
        assert first.finished_at is not None

    assert len(harness.exchanges.exchanges) == 1
    assert not harness.objects.objects
    harness.db.assert_released()

    retry = api.distill_session("session-a")
    assert retry.exchanges_new == 0
    assert retry.objects_created == 1
    assert not retry.errors
    assert len(harness.model.calls) == 2
    assert len(harness.objects.objects) == 1
    harness.db.assert_released()


def test_success_is_idempotent_and_force_reprocesses(harness, monkeypatch):
    first = api.distill_session("session-a")
    original = next(iter(harness.objects.objects.values()))
    assert first.objects_created == 1

    def should_not_build(config):
        pytest.fail("A completed exchange must not initialize an embedding model")

    with monkeypatch.context() as patch:
        patch.setattr(api, "get_embedding_model", should_not_build)
        repeated = api.distill_session("session-a")
    assert repeated.objects_created == repeated.objects_updated == 0
    assert repeated.objects_skipped == 1
    assert len(harness.model.calls) == 1

    forced = api.distill_session("session-a", force=True)
    replacement = next(iter(harness.objects.objects.values()))
    assert forced.exchanges_new == 0
    assert forced.objects_updated == 1
    assert forced.objects_created == forced.objects_skipped == 0
    assert replacement.object_id == original.object_id
    assert replacement.distill_text != original.distill_text
    assert len(harness.objects.objects) == 1
    assert len(harness.model.calls) == 2
    harness.db.assert_released()


def test_one_exchange_failure_does_not_block_others(harness):
    harness.messages.extend([
        NormalizedMessage(2, "user", "Another question"),
        NormalizedMessage(3, "assistant", "Another substantive answer"),
    ])
    harness.model.outcomes = [RuntimeError("temporary failure"), None]
    first = api.distill_session("session-a")
    assert first.exchanges_total == first.exchanges_new == 2
    assert first.objects_created == 1
    assert len(first.errors) == 1

    retry = api.distill_session("session-a")
    assert retry.exchanges_new == 0
    assert retry.objects_created == retry.objects_skipped == 1
    assert not retry.errors
    assert len(harness.model.calls) == 3
    assert len(harness.objects.objects) == 2
    harness.db.assert_released()


@pytest.mark.parametrize("failure", ["initialization", "embedding_query"])
def test_optional_embedding_failure_preserves_null_vector_for_backfill(harness, monkeypatch, caplog, failure):
    def fail(*args):
        raise RuntimeError("embedding temporarily unavailable")

    monkeypatch.setattr(api, "get_embedding_model", fail if failure == "initialization" else (
        lambda config: SimpleNamespace(embed_query=fail)
    ))
    result = api.distill_session("session-a")
    assert result.objects_created == 1
    assert not result.errors
    assert next(iter(harness.objects.objects.values())).embedding is None
    assert "objects_without_embedding" in caplog.text
    repeated = api.distill_session("session-a")
    assert repeated.objects_skipped == 1
    assert len(harness.model.calls) == 1
    harness.db.assert_released()


def test_concurrent_invocations_recheck_completion_under_lock(harness):
    inference_started = threading.Event()
    release_inference = threading.Event()

    def block_inference():
        inference_started.set()
        assert release_inference.wait(timeout=5), "Test did not release inference"

    harness.model.on_invoke = block_inference
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(api.distill_session, "session-a")
        try:
            assert inference_started.wait(timeout=5)
            second = pool.submit(api.distill_session, "session-a")
            assert harness.db.waiting.wait(timeout=5), "Second invocation never waited for the claim"
        finally:
            release_inference.set()
        results = [first.result(timeout=5), second.result(timeout=5)]

    assert sum(result.objects_created for result in results) == 1
    assert sum(result.objects_skipped for result in results) == 1
    assert all(not result.errors for result in results)
    assert len(harness.model.calls) == len(harness.objects.objects) == 1
    harness.db.assert_released()


@pytest.mark.parametrize("phase", ["acquire", "unlock"])
def test_lock_connection_closes_on_lock_sql_failure(monkeypatch, phase):
    db = FakeAdvisoryDatabase()
    setattr(db, f"fail_{phase}", True)
    monkeypatch.setattr(repos, "get_connection", db.connect)
    with pytest.raises(RuntimeError, match="lock acquisition failed|unlock failed"):
        with repos.ObjectsRepo().session_distillation_lock("session-a"):
            assert db.owns_lock()
    db.assert_released()


def test_lock_keys_are_stable_and_session_scoped(monkeypatch):
    db = FakeAdvisoryDatabase()
    monkeypatch.setattr(repos, "get_connection", db.connect)
    for session_id in ("session-a", "session-a", "session-b"):
        with repos.ObjectsRepo().session_distillation_lock(session_id):
            assert db.owns_lock()
    keys = [conn.executed[0][1][0] for conn in db.connections]
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]
    assert all(-(2 ** 63) <= key < 2 ** 63 for key in keys)
    db.assert_released()


class FakeQueryConnection:
    def __init__(self, rows):
        self.rows = rows
        self.description = [("object_id",), ("distill_text",)]
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def cursor(self):
        return self

    def execute(self, sql, params):
        self.executed.append((sql, params))

    def fetchall(self):
        return self.rows


def test_completion_comes_from_objects_and_null_embeddings_remain_queryable(monkeypatch):
    conn = FakeQueryConnection([("exchange-a",)])
    monkeypatch.setattr(repos, "get_connection", lambda **kwargs: conn)
    objects = repos.ObjectsRepo()
    assert objects.get_exchange_ids_for_session("session-a") == {"exchange-a"}
    sql, params = conn.executed[-1]
    assert "FROM memory_v2.memory_objects" in sql
    assert "embedding" not in sql
    assert params == ("session-a",)

    conn.rows = [("exchange-a", "distilled text")]
    assert objects.objects_without_embedding(limit=7) == [{
        "object_id": "exchange-a", "distill_text": "distilled text",
    }]
    sql, params = conn.executed[-1]
    assert "WHERE embedding IS NULL" in sql
    assert params == (7,)


def test_schema_initialization_is_additive_and_preserves_existing_data(monkeypatch):
    statements = pg._split_statements(pg._SCHEMA_SQL)
    assert statements
    assert all(re.match(
        r"CREATE\s+(?:(?:UNIQUE\s+)?INDEX|TABLE|SCHEMA|EXTENSION)\s+IF\s+NOT\s+EXISTS\b",
        statement, re.IGNORECASE,
    ) for statement in statements)
    assert not any(re.match(r"\s*(?:DROP|TRUNCATE|DELETE)\b", statement, re.IGNORECASE) for statement in statements)
    conn = FakeQueryConnection([])
    monkeypatch.setattr(pg, "get_connection", lambda **kwargs: conn)
    # Exercise the real initializer twice, but only against an inert SQL recorder.
    monkeypatch.setattr(conn, "execute", lambda sql: conn.executed.append(sql))
    pg.ensure_schema()
    pg.ensure_schema()
    assert conn.executed == statements + statements
    assert "explicit migration" in pg._SCHEMA_SQL


SESSION_ID = "c1e1f1d1c1a14b1a8b1a1b1a1b1a1b1a"


def test_memory_reader_uses_validated_session_ids_and_safe_listing(tmp_path, monkeypatch):
    monkeypatch.setattr(session_reader, "_SESSIONS_DIR", tmp_path)
    payload = {"messages": [{"role": "user", "content": "safe session"}]}
    (tmp_path / f"{SESSION_ID}.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "arbitrary-name.json").write_text("{}", encoding="utf-8")
    assert session_reader.load_session_raw(SESSION_ID) == payload
    assert session_reader.list_session_ids() == [SESSION_ID]


@pytest.mark.parametrize("session_id", [
    "../outside", "..\\outside", "C:\\outside", "/outside", "invalid", "0" * 32,
    SESSION_ID.upper(), SESSION_ID + ".json", SESSION_ID + "\n",
])
def test_memory_reader_rejects_unsafe_or_non_uuid4_ids(tmp_path, monkeypatch, session_id):
    monkeypatch.setattr(session_reader, "_SESSIONS_DIR", tmp_path)
    with pytest.raises(ValueError, match="Session ID"):
        session_reader.load_session_raw(session_id)


@pytest.mark.parametrize("link_directory", [False, True])
def test_memory_reader_rejects_symlinked_storage(tmp_path, monkeypatch, link_directory):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    target = real_dir / f"{SESSION_ID}.json"
    target.write_text('{"messages": []}', encoding="utf-8")
    sessions_dir = tmp_path / "sessions"
    try:
        if link_directory:
            sessions_dir.symlink_to(real_dir, target_is_directory=True)
        else:
            sessions_dir.mkdir()
            (sessions_dir / f"{SESSION_ID}.json").symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("This platform does not permit test symlink creation")
    monkeypatch.setattr(session_reader, "_SESSIONS_DIR", sessions_dir)
    with pytest.raises(ValueError, match="storage path|Symlinked"):
        session_reader.load_session_raw(SESSION_ID)
    if not link_directory:
        assert session_reader.list_session_ids() == []
