from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

_DEFAULT_ORIGINS = (
    "http://localhost:7788",
    "http://127.0.0.1:7788",
    "http://[::1]:7788",
)
_DEFAULT_HOSTS = ("localhost", "127.0.0.1", "[::1]")


def _values(name: str, defaults: tuple[str, ...]) -> list[str]:
    raw = os.getenv(name)
    values = [part.strip().rstrip("/") for part in raw.split(",") if part.strip()] if raw else list(defaults)
    if "*" in values:
        raise ValueError(f"{name} must list explicit trusted values, not '*'")
    return values


def configure_http_security(app: FastAPI) -> None:
    origins = _values("CLAWTEAM_ALLOWED_ORIGINS", _DEFAULT_ORIGINS)
    hosts = _values("CLAWTEAM_ALLOWED_HOSTS", _DEFAULT_HOSTS)

    @app.middleware("http")
    async def enforce_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin is not None and origin.rstrip("/") not in origins:
            return JSONResponse({"detail": "Origin is not allowed"}, status_code=403)
        return await call_next(request)

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=hosts,
        www_redirect=False,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )
