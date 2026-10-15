from __future__ import annotations

import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api import compress, sessions
from service import session_manager as session_module
from service.session_manager import (
    MAX_COMPRESSED_CONTEXT_CHARS,
    MESSAGE_METADATA_FIELDS,
    SessionConflictError,
    SessionDataError,
    SessionManager,
)
from service.session_runtime import SessionBusyError, claim_session


@pytest.fixture
def manager(tmp_path):
    return SessionManager(tmp_path)


@pytest.fixture
def session_id(manager):
    return manager.create_session()["id"]


def _symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")


@pytest.mark.parametrize("invalid", [
    "", "../outside", "a/b", r"..\outside", "/tmp/session", "C:/session",
    "0" * 32, "f" * 32, "5d49c91b1de24585b98742bb18b3049a.json",
    "5d49c91b-1de2-4585-b987-42bb18b3049a", "5D49C91B1DE24585B98742BB18B3049A",
    "5d49c91b1de21585b98742bb18b3049a", None,
])
def test_every_session_operation_rejects_non_uuid4_ids(manager, invalid):
    operations = [
        manager.load_session, manager.load_session_record, manager.get_history,
        manager.get_compressed_context, manager.load_session_for_agent, manager.delete_session,
        lambda value: manager.rename_session(value, "title"),
        lambda value: manager.save_message(value, "user", "text"),
        lambda value: manager.compress_history(value, "summary", 1),
    ]
    for operation in operations:
        with pytest.raises(ValueError):
            operation(invalid)
    assert list(manager.sessions_dir.glob("*.json")) == []
    assert list(manager.archive_dir.iterdir()) == []


def test_missing_session_is_never_implicitly_created(manager):
    missing = "5d49c91b1de24585b98742bb18b3049a"
    for operation in [
        manager.load_session, manager.load_session_record, manager.get_history,
        manager.load_session_for_agent, manager.get_compressed_context,
        lambda value: manager.save_message(value, "user", "hello"),
        lambda value: manager.set_title(value, "hello"),
        lambda value: manager.compress_history(value, "summary", 1),
    ]:
        with pytest.raises(FileNotFoundError):
            operation(missing)
    manager.delete_session(missing)  # existing idempotent public API
    assert not manager._session_path(missing).exists()


def test_creation_uses_uuid4_and_preserves_default_api(manager):
    record = manager.create_session(title="title")
    assert len(record["id"]) == 32
    assert uuid.UUID(hex=record["id"]).version == 4
    assert record["messages"] == []
    manager.set_title(record["id"], " renamed ")
    assert manager.get_history(record["id"])["title"] == "renamed"
    assert manager.list_sessions()[0]["id"] == record["id"]


def test_tool_calls_and_all_metadata_roundtrip_with_stable_message_id(manager, session_id):
    metadata = {field: [] for field in MESSAGE_METADATA_FIELDS}
    metadata.update({"id": "stream-message-1", "guardian_blocked": False, "run_id": "run-1", "tool_call_id": "tool-1"})
    calls = [{"id": "tool-1", "name": "example", "args": {"value": 1}}]
    message = manager.save_message(session_id, "assistant", "result", calls, metadata=metadata)
    metadata["role_opinions"].append("mutated")
    calls[0]["name"] = "mutated"
    stored = manager.load_session(session_id)[0]
    assert stored == message
    assert stored["id"] == "stream-message-1"
    assert stored["role_opinions"] == []
    assert stored["guardian_blocked"] is False
    assert stored["tool_call_id"] == "tool-1"
    assert stored["tool_calls"][0]["name"] == "example"
    assert manager.get_history(session_id)["messages"][0]["id"] == stored["id"]
    with pytest.raises(ValueError):
        manager.save_message(session_id, "assistant", "duplicate", metadata={"id": stored["id"]})
    with pytest.raises(ValueError):
        manager.save_message(session_id, "assistant", "bad", metadata={"content": "override"})
    assert len(manager.load_session(session_id)) == 1


def test_new_message_ids_are_stable_unique_and_generated_by_default(manager, session_id):
    first = manager.save_message(session_id, "user", "hello")
    second = manager.save_message(session_id, "assistant", "answer")
    assert first["id"] != second["id"]
    assert uuid.UUID(hex=first["id"]).version == 4
    assert manager.load_session(session_id) == [first, second]


@pytest.mark.parametrize("legacy_list", [False, True])
def test_legacy_migration_assigns_persisted_stable_ids_without_losing_fields(manager, session_id, legacy_list):
    messages = [
        {"role": "user", "content": "question", "attachments": [{"name": "a.txt"}]},
        {"role": "assistant", "content": "answer", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "content": "tool result", "tool_call_id": "a"},
    ]
    record = messages if legacy_list else {"id": session_id, "messages": messages}
    path = manager._session_path(session_id)
    path.write_text(json.dumps(record), encoding="utf-8")
    first = manager.load_session(session_id)
    second = SessionManager(manager.base_dir).load_session(session_id)
    assert first == second
    assert len({item["id"] for item in first}) == 3
    assert first[0]["attachments"] == [{"name": "a.txt"}]
    assert first[2]["tool_call_id"] == "a"
    assert json.loads(path.read_text(encoding="utf-8"))["messages"] == first


@pytest.mark.parametrize("stored", [
    "not json", "null", "42", '{"messages": "bad"}',
    '{"messages": ["bad"]}', '{"compressed_context": null}',
    '{"id": "5d49c91b1de24585b98742bb18b3049a", "messages": []}',
])
def test_corrupt_sessions_fail_closed_without_overwriting(manager, session_id, stored):
    path = manager._session_path(session_id)
    path.write_text(stored, encoding="utf-8")
    with pytest.raises(SessionDataError):
        manager.get_history(session_id)
    assert path.read_text(encoding="utf-8") == stored
    assert manager.list_sessions() == []


def test_atomic_session_write_keeps_old_json_and_cleans_temp_on_replace_failure(manager, session_id, monkeypatch):
    manager.save_message(session_id, "user", "original")
    path = manager._session_path(session_id)
    before = path.read_bytes()

    def fail_replace(source, target):
        # Complete JSON exists in the same directory before replacing the old one.
        assert Path(source).parent == path.parent
        assert json.loads(Path(source).read_text(encoding="utf-8"))["messages"][-1]["content"] == "new"
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(session_module.os, "replace", fail_replace)
    with pytest.raises(OSError):
        manager.save_message(session_id, "assistant", "new")
    assert path.read_bytes() == before
    assert list(manager.sessions_dir.glob("*.tmp")) == []
    assert list(manager.sessions_dir.glob(".*.tmp")) == []


def test_concurrent_local_managers_do_not_lose_message_updates(manager, session_id):
    second = SessionManager(manager.base_dir)

    def append(index):
        selected = manager if index % 2 else second
        selected.save_message(session_id, "user", str(index))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append, range(40)))
    messages = manager.load_session(session_id)
    assert len(messages) == 40
    assert {item["content"] for item in messages} == {str(index) for index in range(40)}
    assert len({item["id"] for item in messages}) == 40


def test_session_file_symlink_cannot_read_write_or_delete_other_backend_file(manager):
    sid = "5d49c91b1de24585b98742bb18b3049a"
    target = manager.base_dir / "config.json"
    target.write_text('{"secret": "untouched"}', encoding="utf-8")
    _symlink(manager.sessions_dir / f"{sid}.json", target)
    for operation in [manager.get_history, manager.delete_session, lambda value: manager.save_message(value, "user", "bad")]:
        with pytest.raises(ValueError):
            operation(sid)
    assert target.read_text() == '{"secret": "untouched"}'
    assert manager.list_sessions() == []


def test_constructor_rejects_symlinked_storage(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink(base / "memory_module_v1", outside, directory=True)
    with pytest.raises(ValueError):
        SessionManager(base)
    assert list(outside.iterdir()) == []


def test_archive_directory_symlink_rejected_without_removing_messages(manager, session_id):
    manager.save_message(session_id, "user", "original")
    manager.archive_dir.rmdir()
    target = manager.base_dir / "other"
    target.mkdir()
    _symlink(manager.archive_dir, target, directory=True)
    with pytest.raises(ValueError):
        manager.compress_history(session_id, "summary", 1)
    assert len(manager.load_session(session_id)) == 1
    assert list(target.iterdir()) == []


def test_compression_archive_unique_atomic_metadata_preserved_and_context_bounded(manager, session_id, monkeypatch):
    monkeypatch.setattr(session_module.time, "time", lambda: 1000)
    first = manager.save_message(session_id, "user", "first", metadata={"attachments": [{"name": "a.txt"}]})
    manager.save_message(session_id, "assistant", "answer", metadata={"role_opinions": [{"role": "surgeon"}]})
    manager.save_message(session_id, "user", "remaining")
    result = manager.compress_history(session_id, "A" * 20000, 1)
    assert result == {"archived_count": 1, "remaining_count": 2}
    manager.compress_history(session_id, "B" * 20000, 1)
    archives = sorted(manager.archive_dir.glob("*.json"))
    assert len(archives) == 2
    archived_messages = [json.loads(path.read_text())["messages"][0] for path in archives]
    assert first in archived_messages
    assert len(manager.get_compressed_context(session_id)) <= MAX_COMPRESSED_CONTEXT_CHARS
    agent_history = manager.load_session_for_agent(session_id)
    assert agent_history[0]["role"] == "assistant"
    assert "B" in agent_history[0]["content"]
    assert agent_history[-1] == {"role": "user", "content": "remaining"}


@pytest.mark.parametrize("count,summary", [(0, "ok"), (-1, "ok"), (2, "ok"), (True, "ok"), (1, " "), (1, None)])
def test_invalid_compression_keeps_original_history(manager, session_id, count, summary):
    original = manager.save_message(session_id, "user", "original")
    with pytest.raises(ValueError):
        manager.compress_history(session_id, summary, count)
    assert manager.load_session(session_id) == [original]
    assert list(manager.archive_dir.iterdir()) == []


def test_compression_cas_rejects_changed_prefix_or_context_before_archive(manager, session_id):
    manager.save_message(session_id, "user", "original")
    with pytest.raises(SessionConflictError):
        manager.compress_history(session_id, "summary", 1, expected_message_ids=["not-current"])
    with pytest.raises(SessionConflictError):
        manager.compress_history(session_id, "summary", 1, expected_compressed_context="changed")
    assert len(manager.load_session(session_id)) == 1
    assert list(manager.archive_dir.iterdir()) == []


@pytest.fixture
def route_manager(manager, monkeypatch):
    invalidations = []
    summary_inputs = []

    async def summarize(messages):
        summary_inputs.append(messages)
        return "usable summary"

    async def invalidate(sid):
        invalidations.append(sid)

    value = SimpleNamespace(
        session_manager=manager, base_dir=manager.base_dir,
        summarize_history=summarize, invalidate_session_checkpoint=invalidate,
        invalidations=invalidations, summary_inputs=summary_inputs,
    )
    monkeypatch.setattr(sessions, "_get_agent_manager", lambda: value)
    monkeypatch.setattr(compress, "_get_agent_manager", lambda: value)
    return value


@pytest.mark.parametrize("sid,status", [("../escape", 400), ("5d49c91b1de24585b98742bb18b3049a", 404)])
def test_session_http_routes_cleanly_report_invalid_or_missing_ids(route_manager, sid, status):
    calls = [
        lambda: sessions.get_session_history(sid),
        lambda: sessions.rename_session(sid, sessions.RenameSessionRequest(title="title")),
        lambda: sessions.generate_title(sid, sessions.GenerateTitleRequest(message="seed")),
        lambda: compress.compress_session(sid),
    ]
    for call in calls:
        with pytest.raises(HTTPException) as error:
            asyncio.run(call())
        assert error.value.status_code == status


def test_compress_route_bounds_model_input_and_invalidates_checkpoint(route_manager, manager, session_id):
    for index in range(8):
        manager.save_message(session_id, "user" if index % 2 == 0 else "assistant", str(index) * 40000,
                             metadata={"events": [{"sensitive": "not in summarizer"}]})
    result = asyncio.run(compress.compress_session(session_id))
    assert result == {"archived_count": 4, "remaining_count": 4}
    model_input = route_manager.summary_inputs[0]
    assert len(model_input) == 4
    assert sum(len(item["content"]) for item in model_input) <= compress.MAX_SUMMARIZATION_CHARS
    assert all(set(item) == {"role", "content"} for item in model_input)
    assert route_manager.invalidations == [session_id]
    assert manager.get_compressed_context(session_id) == "usable summary"


def test_compression_and_delete_reject_active_generation(route_manager, manager, session_id):
    for _ in range(4):
        manager.save_message(session_id, "user", "text")
    with claim_session(session_id):
        for operation in [compress.compress_session, sessions.delete_session]:
            with pytest.raises(HTTPException) as error:
                asyncio.run(operation(session_id))
            assert error.value.status_code == 409
    assert len(manager.load_session(session_id)) == 4
    assert route_manager.invalidations == []


def test_delete_route_invalidates_checkpoint_and_releases_claim(route_manager, manager, session_id):
    assert asyncio.run(sessions.delete_session(session_id)) == {"ok": True}
    assert route_manager.invalidations == [session_id]
    with pytest.raises(FileNotFoundError):
        manager.get_history(session_id)
    with claim_session(session_id):
        pass


def test_compression_failed_cas_never_invalidates_checkpoint(route_manager, manager, session_id):
    for _ in range(4):
        manager.save_message(session_id, "user", "text")

    async def conflict(messages):
        manager.compress_history(session_id, "other summary", 1)
        return "stale summary"

    route_manager.summarize_history = conflict
    with pytest.raises(HTTPException) as error:
        asyncio.run(compress.compress_session(session_id))
    assert error.value.status_code == 409
    assert route_manager.invalidations == []
    assert len(list(manager.archive_dir.glob("*.json"))) == 1
    with claim_session(session_id):
        pass


def test_mocked_storage_symlink_resolution_rejects_access_without_privileges(manager, session_id, monkeypatch):
    path = manager._session_path(session_id)
    target = manager.base_dir / "config.json"
    original_resolve = Path.resolve

    def redirected_resolve(value, *args, **kwargs):
        if value == path:
            return target
        return original_resolve(value, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", redirected_resolve)
    with pytest.raises(ValueError):
        manager.load_session(session_id)
    with pytest.raises(ValueError):
        manager.delete_session(session_id)


def test_failed_checkpoint_invalidation_releases_operation_claim(route_manager, manager, session_id):
    for _ in range(4):
        manager.save_message(session_id, "user", "text")

    async def fail(sid):
        raise RuntimeError("checkpoint unavailable")

    route_manager.invalidate_session_checkpoint = fail
    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        asyncio.run(compress.compress_session(session_id))
    assert manager.get_compressed_context(session_id) == "usable summary"
    with claim_session(session_id):
        pass
