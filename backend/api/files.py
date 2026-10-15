from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from service.path_safety import normalize_editable_path, resolve_editable_path

router = APIRouter()


def _get_agent_manager():
    # Document policy and its tests must not need to initialize the LLM stack.
    from graph.agent import agent_manager
    return agent_manager


class SaveFileRequest(BaseModel):
    path: str = Field(..., min_length=1)
    content: str


_DEFAULT_TEMPLATES: dict[str, str] = {
    "workspace/AGENTS.md": "# Agents Guide\n\n请在此描述工作区中各个 Agent 的职责、风格与使用约定。\n",
    "workspace/IDENTITY.md": "# Workspace Identity\n\n在这里定义此工作区的项目愿景、价值观与整体风格。\n",
    "workspace/SOUL.md": "# Soul\n\n用于描述此 AI 的“灵魂”、长期目标与行为准则。\n",
    "workspace/USER.md": "# User Profile\n\n记录典型用户画像、需求与偏好。\n",
    "memory_module_v1/long_term_memory/MEMORY.md": "# Long-term Memory\n\n在这里追加重要的长期记忆片段。\n",
}


def _resolve_path(relative_path: str) -> Path:
    base_dir = _get_agent_manager().base_dir
    if base_dir is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")
    try:
        return resolve_editable_path(base_dir, relative_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _safe_scan_skills(base_dir: Path):
    from tools.skills_scanner import scan_skills

    # scan_skills reads instruction files; validate every target *before* it reads.
    # Checking only its returned list would already have exposed symlink targets.
    _resolve_path("skills/SKILLS_SNAPSHOT.md")
    for skill_file in (base_dir / "skills").glob("*/SKILL.md"):
        _resolve_path(skill_file.relative_to(base_dir).as_posix())
    return scan_skills(base_dir)


def _rebuild_memory_index() -> None:
    from service.memory_indexer import memory_indexer
    memory_indexer.rebuild_index()


@router.get("/files")
async def read_file(path: str = Query(..., min_length=1)) -> dict[str, str]:
    file_path = _resolve_path(path)
    normalized = normalize_editable_path(path)
    try:
        if not file_path.exists():
            default_content = _DEFAULT_TEMPLATES.get(normalized)
            if default_content is None:
                raise HTTPException(status_code=404, detail="File not found")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path = _resolve_path(path)
            file_path.write_text(default_content, encoding="utf-8")
        content = file_path.read_text(encoding="utf-8")
    except (IsADirectoryError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Path is not a UTF-8 document") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Unable to read document") from exc
    return {"path": path.replace("\\", "/"), "content": content}


@router.post("/files")
async def save_file(payload: SaveFileRequest) -> dict[str, Any]:
    file_path = _resolve_path(payload.path)
    normalized = normalize_editable_path(payload.path)
    base_dir = _get_agent_manager().base_dir
    assert base_dir is not None
    # Refreshing a skill also reads other skills and writes the snapshot, so those
    # secondary accesses must satisfy the same canonical allowlist.
    if normalized.startswith("skills/") and normalized.endswith("/SKILL.md"):
        _safe_scan_skills(base_dir)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path = _resolve_path(payload.path)
        file_path.write_text(payload.content, encoding="utf-8")
        if normalized == "memory_module_v1/long_term_memory/MEMORY.md":
            _rebuild_memory_index()
        if normalized.startswith("skills/") and normalized.endswith("/SKILL.md"):
            from tools.skills_scanner import refresh_snapshot
            refresh_snapshot(base_dir)
    except (IsADirectoryError, UnicodeError) as exc:
        raise HTTPException(status_code=400, detail="Path is not a UTF-8 document") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Unable to save document") from exc
    return {"ok": True, "path": payload.path.replace("\\", "/")}


@router.get("/skills")
async def list_skills() -> list[dict[str, str]]:
    base_dir = _get_agent_manager().base_dir
    if base_dir is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")
    return [skill.__dict__ for skill in _safe_scan_skills(base_dir)]
