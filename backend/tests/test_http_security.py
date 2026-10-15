from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from service.http_security import configure_http_security


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv("CLAWTEAM_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("CLAWTEAM_ALLOWED_HOSTS", raising=False)
    instance = FastAPI()
    configure_http_security(instance)

    @instance.post("/api/write")
    async def write():
        return {"ok": True}

    return instance


async def test_local_request_allowed(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        response = await client.post("/api/write", headers={"Origin": "http://localhost:7788"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:7788"
    assert "access-control-allow-credentials" not in response.headers


async def test_untrusted_origin_is_rejected_before_handler(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        response = await client.post("/api/write", headers={"Origin": "https://untrusted.example"})
    assert response.status_code == 403


async def test_untrusted_host_is_rejected(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://rebind.example") as client:
        response = await client.post("/api/write")
    assert response.status_code == 400


async def test_explicit_reverse_proxy_configuration(monkeypatch):
    monkeypatch.setenv("CLAWTEAM_ALLOWED_HOSTS", "mdt.example")
    monkeypatch.setenv("CLAWTEAM_ALLOWED_ORIGINS", "https://mdt.example")
    instance = FastAPI()
    configure_http_security(instance)

    @instance.get("/health")
    async def health():
        return {"status": "ok"}

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=instance), base_url="https://mdt.example") as client:
        response = await client.get("/health", headers={"Origin": "https://mdt.example"})
    assert response.status_code == 200


@pytest.mark.parametrize("name", ["CLAWTEAM_ALLOWED_HOSTS", "CLAWTEAM_ALLOWED_ORIGINS"])
def test_wildcard_configuration_is_rejected(monkeypatch, name):
    monkeypatch.setenv(name, "*")
    with pytest.raises(ValueError, match="explicit"):
        configure_http_security(FastAPI())
