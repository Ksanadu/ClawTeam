from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any


# A process-wide lock also serializes multiple managers for the same directory.
# This is local-process protection, not a multi-worker/distributed transaction.
_MUTATION_LOCK = threading.RLock()
_SESSION_ID = re.compile(r"[0-9a-f]{32}\Z")
MAX_COMPRESSED_CONTEXT_CHARS = 8000
MAX_SUMMARY_CHARS = 4000
MESSAGE_METADATA_FIELDS = frozenset({
    "role_opinions", "routing", "retrievals", "progress", "attachments",
    "guardian_blocked", "collaboration_metric", "run_id", "tool_call_id",
    "experiment_mode", "status", "failed_roles", "events",
})


class SessionConflictError(RuntimeError):
    """The history changed while a compression summary was being generated."""


class SessionDataError(ValueError):
    """A stored session is corrupt; never replace it with a new empty session."""


class SessionManager:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.sessions_dir = self.base_dir / "memory_module_v1" / "sessions"
        self.archive_dir = self.sessions_dir / "archive"
        with _MUTATION_LOCK:
            self._checked_storage_path(self.sessions_dir)
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            self._checked_storage_path(self.archive_dir)
            self.archive_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_session_id(session_id: str) -> str:
        if not isinstance(session_id, str) or _SESSION_ID.fullmatch(session_id) is None:
            raise ValueError("Session ID must be a lowercase UUID4 hex string")
        parsed = uuid.UUID(hex=session_id)
        if parsed.version != 4 or parsed.variant != uuid.RFC_4122:
            raise ValueError("Session ID must be a lowercase UUID4 hex string")
        return session_id

    def _checked_storage_path(self, path: Path) -> Path:
        try:
            resolved = path.resolve()
            resolved.relative_to(self.base_dir)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("Unsafe session storage path") from exc
        # Storage names have no aliases. Reject links to other sessions or to
        # other backend files, even if the resolved target remains under base.
        if resolved != path:
            raise ValueError("Symlinked session storage is not allowed")
        return resolved

    def _session_path(self, session_id: str) -> Path:
        self.validate_session_id(session_id)
        return self._checked_storage_path(self.sessions_dir / f"{session_id}.json")

    def _default_record(self, session_id: str, title: str = "新会话") -> dict[str, Any]:
        now = time.time()
        return {
            "id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "compressed_context": "",
            "messages": [],
        }

    def _read_session_file(self, session_id: str) -> dict[str, Any]:
        path = self._session_path(session_id)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError("Session not found") from None
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SessionDataError("Stored session is not valid JSON") from exc
        changed = False
        if isinstance(raw, list):
            record = self._default_record(session_id)
            record["messages"] = raw
            changed = True
        elif isinstance(raw, dict):
            record = raw
            if record.get("id", session_id) != session_id:
                raise SessionDataError("Stored session ID does not match its filename")
            defaults = self._default_record(session_id)
            defaults["updated_at"] = record.get("created_at", defaults["created_at"])
            for key, value in defaults.items():
                if key not in record:
                    record[key] = value
                    changed = True
        else:
            raise SessionDataError("Stored session must be an object or legacy message list")
        messages = record["messages"]
        if not isinstance(messages, list) or any(not isinstance(item, dict) for item in messages):
            raise SessionDataError("Stored messages must be a list of objects")
        if not isinstance(record["compressed_context"], str):
            raise SessionDataError("Stored compressed context must be text")
        ids: set[str] = set()
        for index, message in enumerate(messages):
            message_id = message.get("id")
            if not isinstance(message_id, str) or not message_id or message_id in ids:
                # Deterministic even if migration has to be retried after a crash.
                message_id = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"clawteam:{session_id}:{index}:"
                    + json.dumps(message, ensure_ascii=False, sort_keys=True),
                ).hex
                message["id"] = message_id
                changed = True
            ids.add(message_id)
        bounded_context = self._bound_context(record["compressed_context"])
        if bounded_context != record["compressed_context"]:
            record["compressed_context"] = bounded_context
            changed = True
        if changed:
            self._write_session(record)
        return record

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        self._checked_storage_path(path)
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.stem}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self._checked_storage_path(path)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _write_session(self, record: dict[str, Any]) -> None:
        path = self._session_path(record["id"])
        record["updated_at"] = time.time()
        self._atomic_write_json(path, record)

    def create_session(self, title: str = "新会话") -> dict[str, Any]:
        with _MUTATION_LOCK:
            session_id = uuid.uuid4().hex
            record = self._default_record(session_id, title=title)
            self._write_session(record)
            return record

    def list_sessions(self) -> list[dict[str, Any]]:
        with _MUTATION_LOCK:
            self._checked_storage_path(self.sessions_dir)
            records: list[dict[str, Any]] = []
            for path in self.sessions_dir.glob("*.json"):
                try:
                    record = self._read_session_file(path.stem)
                except (OSError, ValueError):
                    continue
                records.append({
                    "id": record["id"],
                    "title": record["title"],
                    "created_at": record["created_at"],
                    "updated_at": record["updated_at"],
                    "message_count": len(record["messages"]),
                })
            return sorted(
                records,
                key=lambda item: item["updated_at"]
                if isinstance(item["updated_at"], (int, float)) else 0,
                reverse=True,
            )

    def load_session_record(self, session_id: str) -> dict[str, Any]:
        with _MUTATION_LOCK:
            return self._read_session_file(session_id)

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        return self.load_session_record(session_id)["messages"]

    def load_session_for_agent(self, session_id: str) -> list[dict[str, str]]:
        record = self.load_session_record(session_id)
        merged: list[dict[str, str]] = []
        compressed_context = record.get("compressed_context", "").strip()
        if compressed_context:
            merged.append({
                "role": "assistant",
                "content": f"[以下是之前对话的摘要]\n{compressed_context}",
            })
        for message in record.get("messages", []):
            role = message.get("role", "")
            content = str(message.get("content", "") or "")
            if role == "assistant" and merged and merged[-1]["role"] == "assistant":
                if content:
                    if merged[-1]["content"]:
                        merged[-1]["content"] += "\n\n" + content
                    else:
                        merged[-1]["content"] = content
                continue
            merged.append({"role": role, "content": content})
        return [item for item in merged if item["role"] in {"user", "assistant"}]

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with _MUTATION_LOCK:
            record = self._read_session_file(session_id)
            metadata = copy.deepcopy(metadata) if metadata is not None else {}
            if not isinstance(metadata, dict):
                raise ValueError("Message metadata must be an object")
            unknown = set(metadata) - MESSAGE_METADATA_FIELDS - {"id"}
            if unknown:
                raise ValueError("Unsupported message metadata fields: " + ", ".join(sorted(unknown)))
            message_id = metadata.pop("id", None)
            if message_id is None:
                message_id = uuid.uuid4().hex
            if (
                not isinstance(message_id, str) or not message_id
                or len(message_id) > 200 or any(ord(char) < 32 for char in message_id)
                or any(item["id"] == message_id for item in record["messages"])
            ):
                raise ValueError("Message ID must be a unique nonempty string")
            message: dict[str, Any] = {"id": message_id, "role": role, "content": content}
            if tool_calls:
                message["tool_calls"] = copy.deepcopy(tool_calls)
            message.update(metadata)
            record["messages"].append(message)
            self._write_session(record)
            return message

    def get_history(self, session_id: str) -> dict[str, Any]:
        return self.load_session_record(session_id)

    def rename_session(self, session_id: str, title: str) -> dict[str, Any]:
        with _MUTATION_LOCK:
            record = self._read_session_file(session_id)
            record["title"] = title.strip() or "新会话"
            self._write_session(record)
            return record

    def set_title(self, session_id: str, title: str) -> dict[str, Any]:
        return self.rename_session(session_id, title)

    def delete_session(self, session_id: str) -> None:
        with _MUTATION_LOCK:
            # Preserve idempotent deletion, but never bypass ID/path validation.
            self._session_path(session_id).unlink(missing_ok=True)

    @staticmethod
    def _bound_context(context: str) -> str:
        context = context.strip()
        if len(context) <= MAX_COMPRESSED_CONTEXT_CHARS:
            return context
        marker = "[较早摘要已截断]\n"
        return marker + context[-(MAX_COMPRESSED_CONTEXT_CHARS - len(marker)):]

    def compress_history(
        self, session_id: str, summary: str, n_messages: int, *,
        expected_message_ids: list[str] | None = None,
        expected_compressed_context: str | None = None,
    ) -> dict[str, int]:
        with _MUTATION_LOCK:
            record = self._read_session_file(session_id)
            messages = record["messages"]
            if type(n_messages) is not int or n_messages < 1:
                raise ValueError("Compression count must be within the existing message history")
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("A nonempty compression summary is required")
            if expected_message_ids is not None and (
                [item["id"] for item in messages[:n_messages]] != expected_message_ids
            ):
                raise SessionConflictError("Session history changed; retry compression")
            if expected_compressed_context is not None and (
                record["compressed_context"] != expected_compressed_context
            ):
                raise SessionConflictError("Session summary changed; retry compression")
            if n_messages > len(messages):
                raise ValueError("Compression count must be within the existing message history")
            archived = messages[:n_messages]
            existing_summary = record["compressed_context"].strip()
            archive_payload = {
                "session_id": session_id,
                "archived_at": time.time(),
                "compressed_context": existing_summary,
                "messages": archived,
            }
            # Random component prevents two compressions in one second from
            # overwriting each other's archive. Check archive symlinks as well.
            archive_path = self.archive_dir / f"{session_id}_{uuid.uuid4().hex}.json"
            self._atomic_write_json(archive_path, archive_payload)
            current_summary = summary.strip()[:MAX_SUMMARY_CHARS]
            record["compressed_context"] = self._bound_context(
                f"{existing_summary}\n---\n{current_summary}" if existing_summary else current_summary
            )
            record["messages"] = messages[n_messages:]
            self._write_session(record)
            return {"archived_count": len(archived), "remaining_count": len(record["messages"])}

    def get_compressed_context(self, session_id: str) -> str:
        return self.load_session_record(session_id).get("compressed_context", "")
