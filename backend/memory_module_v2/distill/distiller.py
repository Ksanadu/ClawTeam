"""Distiller: Exchange -> DistilledObject via LLM.

Supports a dedicated distillation model via DISTILL_PROVIDER / DISTILL_MODEL env vars.
When not configured, falls back to the main LLM (LLM_PROVIDER / LLM_MODEL).
This allows using a smaller, cheaper model for distillation to save tokens.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from json import JSONDecodeError

from langchain_core.messages import HumanMessage, SystemMessage

from config import get_settings, LLM_PROVIDER_DEFAULTS, PROVIDER_ALIASES
from graph.llm import ResolvedLLMConfig, get_llm, build_llm_config_from_settings

from ..domain.enums import RoomType
from ..domain.models import DistilledObject, Exchange, RoomAssignment
from ..ingest.file_path_extractor import extract_file_paths
from .prompts import build_distill_prompt

logger = logging.getLogger(__name__)


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _get_distill_llm():
    """Build the LLM for distillation.

    Priority: DISTILL_* env vars → fall back to main LLM_* settings.
    """
    distill_provider_raw = _first_env("DISTILL_PROVIDER")

    if not distill_provider_raw:
        settings = get_settings()
        config = build_llm_config_from_settings(settings, temperature=0.0, streaming=False)
        return get_llm(config), settings.llm_provider, settings.llm_model

    provider = PROVIDER_ALIASES.get(distill_provider_raw.lower(), distill_provider_raw.lower())
    if provider not in LLM_PROVIDER_DEFAULTS:
        provider = "openai"

    defaults = LLM_PROVIDER_DEFAULTS[provider]
    model = _first_env("DISTILL_MODEL") or defaults["model"]
    base_url = _first_env("DISTILL_BASE_URL") or defaults["base_url"]

    api_key = _first_env("DISTILL_API_KEY")
    if not api_key:
        settings = get_settings()
        if provider == settings.llm_provider:
            api_key = settings.llm_api_key
        else:
            api_key = _first_env("LLM_API_KEY")

    config = ResolvedLLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        streaming=False,
    )
    return get_llm(config), provider, model


def _parse_distill_response(text: str) -> dict:
    """Parse LLM JSON response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except JSONDecodeError as exc:
        snippet = text[:1200].replace("\r\n", "\n")
        raise ValueError(
            "LLM distillation output is not valid JSON. "
            "Ensure the model follows the required JSON-only format. "
            f"Raw snippet:\n{snippet}"
        ) from exc


def distill_exchange(exchange: Exchange) -> DistilledObject:
    """Distill a single exchange into a DistilledObject."""
    llm, provider, model = _get_distill_llm()

    prompt_messages = build_distill_prompt(
        session_id=exchange.session_id,
        ply_start=exchange.ply_start,
        ply_end=exchange.ply_end,
        exchange_text=exchange.verbatim_snippet or exchange.verbatim_text,
    )

    lc_messages = [
        SystemMessage(content=prompt_messages[0]["content"]),
        HumanMessage(content=prompt_messages[1]["content"]),
    ]

    try:
        response = llm.invoke(lc_messages)
    except Exception as exc:
        raise RuntimeError(
            "LLM invoke failed during distillation. "
            "Check provider/model/base_url/api_key configuration."
        ) from exc

    raw = _parse_distill_response(str(getattr(response, "content", "") or ""))

    rooms: list[RoomAssignment] = []
    for r in raw.get("rooms", []):
        try:
            rooms.append(RoomAssignment(
                room_type=RoomType(r.get("room_type", "concept")),
                room_key=r.get("room_key", "unknown"),
                room_label=r.get("room_label", ""),
                relevance=float(r.get("relevance", 1.0)),
            ))
        except (ValueError, KeyError):
            continue

    files_touched = extract_file_paths(exchange.verbatim_text)

    exchange_core = raw.get("exchange_core", "")
    specific_context = raw.get("specific_context", "")
    distill_text = f"{exchange_core}\n{specific_context}"

    return DistilledObject(
        object_id=exchange.exchange_id,
        exchange_id=exchange.exchange_id,
        session_id=exchange.session_id,
        ply_start=exchange.ply_start,
        ply_end=exchange.ply_end,
        exchange_core=exchange_core,
        specific_context=specific_context,
        room_assignments=rooms,
        files_touched=files_touched,
        distill_text=distill_text,
        distilled_at=datetime.now(timezone.utc),
        distill_provider=provider,
        distill_model=model,
    )
