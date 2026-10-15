from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from api import files, tokens
from service.session_manager import SessionManager


@pytest.fixture
def client_app(tmp_path, monkeypatch):
    manager = SessionManager(tmp_path)
    agent = SimpleNamespace(base_dir=tmp_path, session_manager=manager)
    monkeypatch.setattr(tokens, "_get_agent_manager", lambda: agent)
    monkeypatch.setattr(files, "_get_agent_manager", lambda: agent)
    monkeypatch.setattr(tokens, "_count_tokens", lambda text: len(text))
    monkeypatch.setattr(tokens, "build_system_prompt", lambda base: "system")
    app = FastAPI()
    app.include_router(tokens.router, prefix="/api")
    return app, manager


@pytest.mark.parametrize("path", ["workspace/../config/.env", "../README.md", "C:/private.txt", "workspace/test.py"])
async def test_token_files_reuses_canonical_editable_policy(client_app, path):
    app, _ = client_app
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/tokens/files", json={"paths": [path]})
    assert response.status_code == 400


async def test_allowed_document_token_count(client_app, tmp_path):
    app, _ = client_app
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "SOUL.md").write_text("test", encoding="utf-8")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/tokens/files", json={"paths": ["workspace/SOUL.md"]})
    assert response.status_code == 200
    assert response.json()["total_tokens"] == 4


async def test_invalid_and_missing_token_session_errors(client_app):
    app, manager = client_app
    session = manager.create_session()
    manager.delete_session(session["id"])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        invalid = await client.get("/api/tokens/session/not-an-id")
        missing = await client.get(f"/api/tokens/session/{session['id']}")
    assert invalid.status_code == 400
    assert missing.status_code == 404
