from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api import files
from service.path_safety import normalize_editable_path, resolve_editable_path


@pytest.fixture
def document_root(tmp_path, monkeypatch):
    manager = SimpleNamespace(base_dir=tmp_path)
    monkeypatch.setattr(files, "_get_agent_manager", lambda: manager)
    return tmp_path


def _symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")


@pytest.mark.parametrize("path", [
    "workspace/AGENTS.md", "workspace/IDENTITY.md", "workspace/SOUL.md",
    "workspace/USER.md", "workspace/roles/SURGEON.md", "MEMORY.md",
    "memory_module_v1/long_term_memory/MEMORY.md", "SKILLS_SNAPSHOT.md",
    "skills/retry-lesson-capture/SKILL.md", "skills/rag-skill/SKILL.md",
])
def test_ui_markdown_documents_and_aliases_remain_editable(tmp_path, path):
    actual = resolve_editable_path(tmp_path, path)
    assert actual == tmp_path / normalize_editable_path(path)


@pytest.mark.parametrize("path", [
    "workspace/../config/config.json", "skills/../graph/agent.py",
    "knowledge/../config/.env", "workspace/../../outside.md",
    "/workspace/SOUL.md", "C:/workspace/SOUL.md", "//server/workspace/SOUL.md",
    "workspace/./SOUL.md", "workspace//SOUL.md", "workspace/SOUL.md/",
    "workspace/SOUL.md:secret", "workspace/SOUL.md.", "workspace/SOUL.md\x00",
    "skills/rag-skill/scripts/retrieve.py", "skills/rag-skill/config.json",
    "skills/rag-skill/.env", "skills/.secrets/SKILL.md", "skills/config/SKILL.md/..",
    "workspace/secret.md", "knowledge/secrets.md", "memory/passwords.md",
])
def test_reject_traversal_absolute_code_configuration_and_secret_paths(tmp_path, path):
    with pytest.raises(ValueError):
        resolve_editable_path(tmp_path, path)


def test_windows_separators_keep_legacy_paths_without_permitting_traversal(tmp_path):
    assert resolve_editable_path(tmp_path, r"workspace\SOUL.md") == tmp_path / "workspace/SOUL.md"
    with pytest.raises(ValueError):
        resolve_editable_path(tmp_path, r"workspace\..\config\config.json")


@pytest.mark.parametrize("outside", [False, True])
def test_allowed_document_symlink_cannot_escape_its_resolved_allowlist(tmp_path, outside):
    base = tmp_path / "backend"
    (base / "workspace").mkdir(parents=True)
    target = tmp_path / "outside.md" if outside else base / "config/config.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("private", encoding="utf-8")
    _symlink(base / "workspace/SOUL.md", target)
    with pytest.raises(ValueError):
        resolve_editable_path(base, "workspace/SOUL.md")


def test_allowed_directory_symlink_cannot_reach_backend_code(tmp_path):
    (tmp_path / "graph").mkdir()
    (tmp_path / "graph/SKILL.md").write_text("secret", encoding="utf-8")
    (tmp_path / "skills").mkdir()
    _symlink(tmp_path / "skills/escape", tmp_path / "graph", directory=True)
    with pytest.raises(ValueError):
        resolve_editable_path(tmp_path, "skills/escape/SKILL.md")


def test_default_template_and_memory_alias_read_write(document_root, monkeypatch):
    rebuilds = []
    monkeypatch.setattr(files, "_rebuild_memory_index", lambda: rebuilds.append(True))
    original = asyncio.run(files.read_file(path="MEMORY.md"))
    assert "Long-term Memory" in original["content"]
    result = asyncio.run(files.save_file(files.SaveFileRequest(path="MEMORY.md", content="updated")))
    assert result == {"ok": True, "path": "MEMORY.md"}
    assert rebuilds == [True]
    assert (document_root / "memory_module_v1/long_term_memory/MEMORY.md").read_text() == "updated"


def test_snapshot_can_be_edited_without_automatic_overwrite(document_root):
    asyncio.run(files.save_file(files.SaveFileRequest(path="SKILLS_SNAPSHOT.md", content="custom")))
    assert asyncio.run(files.read_file(path="SKILLS_SNAPSHOT.md"))["content"] == "custom"


@pytest.mark.parametrize("path,status", [("workspace/code.py", 400), ("skills/missing/SKILL.md", 404)])
def test_document_route_reports_expected_http_errors(document_root, path, status):
    with pytest.raises(HTTPException) as error:
        asyncio.run(files.read_file(path=path))
    assert error.value.status_code == status


def test_directory_instead_of_document_is_clean_http_error(document_root):
    (document_root / "workspace/SOUL.md").mkdir(parents=True)
    with pytest.raises(HTTPException) as error:
        asyncio.run(files.read_file(path="workspace/SOUL.md"))
    # Windows reports PermissionError when opening directories.
    assert error.value.status_code in {400, 500}


def test_list_skills_rejects_symlink_before_scanner_can_read(document_root):
    (document_root / "skills/escape").mkdir(parents=True)
    target = document_root / "private.md"
    target.write_text("must not read", encoding="utf-8")
    _symlink(document_root / "skills/escape/SKILL.md", target)
    with pytest.raises(HTTPException) as error:
        asyncio.run(files.list_skills())
    assert error.value.status_code == 400


@pytest.mark.parametrize("target", ["config/config.json", "../outside.md"])
def test_canonical_allowlist_rejects_mocked_resolution_without_symlink_privileges(tmp_path, monkeypatch, target):
    original_resolve = Path.resolve
    allowed = tmp_path / "workspace/SOUL.md"
    resolved_target = original_resolve(tmp_path / target)

    def redirected_resolve(path, *args, **kwargs):
        if path == allowed:
            return resolved_target
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", redirected_resolve)
    with pytest.raises(ValueError):
        resolve_editable_path(tmp_path, "workspace/SOUL.md")
