from __future__ import annotations

from pathlib import Path

from config import get_settings
from memory_module_v2.service.config import get_memory_backend, get_memory_v2_inject_mode

SYSTEM_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("Skills Snapshot", "skills/SKILLS_SNAPSHOT.md"),
    ("Soul", "workspace/SOUL.md"),
    ("Identity", "workspace/IDENTITY.md"),
    ("User Profile", "workspace/USER.md"),
    ("Agents Guide", "workspace/AGENTS.md"),
)

_MEMORY_V1_PATH = "memory_module_v1/long_term_memory/MEMORY.md"

_MEMORY_HINTS = {
    "off": None,
    "v1_rag": (
        "<!-- Long-term Memory -->\n"
        "长期记忆将通过检索动态注入。你应优先使用当次检索到的 MEMORY 片段，"
        "不要假设未检索到的记忆仍然有效。"
    ),
    "v2_tool": (
        "<!-- Long-term Memory (v2) -->\n"
        "你可以使用 `search_memory` 工具检索跨会话的长期记忆。\n"
        "当用户提到过去讨论过的话题、之前解决过的问题、或需要回忆历史上下文时，主动调用此工具。\n"
        "检索结果包含历史对话的原始片段（verbatim evidence），优先引用这些证据来回答。"
    ),
    "v2_always": (
        "<!-- Long-term Memory (v2) -->\n"
        "系统会在每轮对话前自动检索长期记忆并注入相关的历史对话片段。\n"
        "优先使用注入的证据片段来回答与历史相关的问题，不要假设未注入的记忆仍然有效。"
    ),
}


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _read_component(base_dir: Path, relative_path: str, limit: int) -> str:
    path = base_dir / relative_path
    if not path.exists():
        return f"[missing component: {relative_path}]"
    return _truncate(path.read_text(encoding="utf-8"), limit)


def _get_memory_hint_key() -> str:
    backend = get_memory_backend()
    if backend == "v1":
        return "v1_rag"
    if backend == "v2":
        mode = get_memory_v2_inject_mode()
        return "v2_tool" if mode == "tool" else "v2_always"
    return "off"


def build_system_prompt(base_dir: Path) -> str:
    settings = get_settings()
    parts: list[str] = []

    for label, relative_path in SYSTEM_COMPONENTS:
        content = _read_component(base_dir, relative_path, settings.component_char_limit)
        parts.append(f"<!-- {label} -->\n{content}")

    hint_key = _get_memory_hint_key()

    if hint_key == "off" or hint_key == "v1_rag":
        v1_content = _read_component(base_dir, _MEMORY_V1_PATH, settings.component_char_limit)
        if hint_key == "v1_rag":
            parts.append(_MEMORY_HINTS["v1_rag"])
        else:
            parts.append(f"<!-- Long-term Memory -->\n{v1_content}")

    memory_hint = _MEMORY_HINTS.get(hint_key)
    if memory_hint and hint_key.startswith("v2_"):
        parts.append(memory_hint)

    return "\n\n".join(parts)
