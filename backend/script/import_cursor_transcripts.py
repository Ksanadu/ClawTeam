import json
import os
from pathlib import Path
from typing import List, Dict, Tuple


# 根目录：Cursor 在本机保存所有项目的地方
CURSOR_PROJECTS_ROOT = Path(r"C:\Users\Jimmy\.cursor\projects")

# 目标：当前项目的 sessions 目录
TARGET_SESSIONS_DIR = Path(
    r"d:\LLM Project\langchain-miniopenclaw-main\backend\memory_module_v1\sessions"
)
TARGET_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def file_times_to_timestamps(path: Path) -> Tuple[float, float]:
    """
    把文件的创建时间 / 修改时间转换成 Unix 时间戳（秒）。
    Windows 上 st_ctime 是创建时间，st_mtime 是最后修改时间。
    """
    stat = path.stat()
    created_at = stat.st_ctime
    updated_at = stat.st_mtime
    return created_at, updated_at


def load_messages_from_jsonl(jsonl_path: Path) -> List[Dict]:
    """
    从 Cursor 的 jsonl 会话文件中提取 user / assistant 的纯文本消息。
    只保留 message.content 里 type == "text" 的片段。
    """
    messages: List[Dict] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            role = row.get("role")
            if role not in {"user", "assistant"}:
                continue

            msg = row.get("message") or {}
            content_items = msg.get("content") or []

            texts: List[str] = []
            for item in content_items:
                if item.get("type") == "text" and "text" in item:
                    texts.append(str(item["text"]))

            if not texts:
                continue

            text = "\n\n".join(texts).strip()
            if not text:
                continue

            messages.append({"role": role, "content": text})

    return messages


def guess_title(messages: List[Dict], session_id: str) -> str:
    """
    用第一条 user 消息的前若干个字符作为标题；否则用一个兜底标题。
    """
    for m in messages:
        if m.get("role") == "user":
            text = str(m.get("content", "")).strip()
            if text:
                return text[:20]

    return f"导入自 Cursor 会话 {session_id}"


def import_one_jsonl(jsonl_path: Path) -> None:
    """
    把单个 uuid.jsonl 文件转换成一份 session json 写入当前项目。
    """
    session_id = jsonl_path.stem  # uuid.jsonl -> uuid

    messages = load_messages_from_jsonl(jsonl_path)
    if not messages:
        return

    created_at, updated_at = file_times_to_timestamps(jsonl_path)
    title = guess_title(messages, session_id)

    session_record: Dict = {
        "id": session_id,
        "title": title,
        "created_at": created_at,
        "updated_at": updated_at,
        "compressed_context": "",
        "messages": messages,
    }

    out_path = TARGET_SESSIONS_DIR / f"{session_id}.json"

    # 避免重复导入：如果已经存在同名 session 文件就跳过
    if out_path.exists():
        print(f"Skip existing session: {out_path}")
        return

    out_path.write_text(
        json.dumps(session_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Imported {jsonl_path} -> {out_path}")


def main() -> None:
    """
    遍历所有 Cursor 项目，批量导入每个项目的主会话 jsonl。
    规则：
      - 只看 agent-transcripts/<uuid>/<uuid>.jsonl
      - 忽略 subagents 里的子 agent 记录
    """
    if not CURSOR_PROJECTS_ROOT.exists():
        print(f"Cursor projects root not found: {CURSOR_PROJECTS_ROOT}")
        return

    for project_dir in CURSOR_PROJECTS_ROOT.iterdir():
        if not project_dir.is_dir():
            continue

        agent_dir = project_dir / "agent-transcripts"
        if not agent_dir.exists():
            continue

        print(f"Scanning project: {project_dir}")

        for sub in agent_dir.iterdir():
            if not sub.is_dir():
                continue

            # 形如 agent-transcripts/<uuid>/<uuid>.jsonl
            main_jsonl = sub / f"{sub.name}.jsonl"
            if main_jsonl.exists():
                import_one_jsonl(main_jsonl)


if __name__ == "__main__":
    main()

