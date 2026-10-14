from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

def _ensure_import_path() -> None:
    """
    Ensure `backend/` is in sys.path so `python backend/.../generate_ground_truth.py` works.
    """
    repo_backend = Path(__file__).resolve().parents[2]
    if str(repo_backend) not in sys.path:
        sys.path.insert(0, str(repo_backend))


_ensure_import_path()

from memory_module_v2.ingest.exchange_segmenter import segment_exchanges
from memory_module_v2.ingest.session_reader import get_sessions_dir, list_session_ids, read_session
from memory_module_v2.service.config import get_memory_v2_config


@dataclass
class GroundTruthMeta:
    generated_at: str
    sessions_dir: str
    segmenter_params: dict[str, Any]
    labels_written: int
    labels_skipped: dict[str, int]
    sessions_processed: int


_USER_QUERY_TAG_RE = re.compile(r"</?user_query>", flags=re.IGNORECASE)
_STANDALONE_XML_TAG_LINE_RE = re.compile(r"^\s*<[^>]+>\s*$")


def _clean_query_text(query: str) -> str:
    """
    Clean wrapper-like tags in v1 sessions so retrieval input focuses on user question content.

    Current target: remove <user_query>...</user_query> wrappers and standalone XML-like tag lines.
    """
    if not query:
        return query

    text = query.strip()
    # Remove explicit <user_query> tags anywhere in the string.
    text = _USER_QUERY_TAG_RE.sub("", text)

    # Drop lines that are only XML-like tags: "<attached_files>", "</cursor_commands>", etc.
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        if _STANDALONE_XML_TAG_LINE_RE.match(line):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines).strip()

    return text


def _label_one_exchange(
    *,
    session_id: str,
    exchange: Any,
    messages: list[Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Create a single label for one exchange (proposal: strict Scheme A).

    - query = messages[exchange.ply_start].content (must be role=user)
    - relevant_exchange_ids = [exchange.exchange_id]
    - only substantive=true exchanges are used (caller filters)
    """
    ply_start = int(exchange.ply_start)
    if ply_start < 0 or ply_start >= len(messages):
        return None, "out_of_range_ply_start"

    query_msg = messages[ply_start]
    if getattr(query_msg, "role", "") != "user":
        # Segmenter contract generally starts exchanges at user, but we guard anyway.
        return None, "ply_start_not_user"

    query = getattr(query_msg, "content", "") or ""
    if not query.strip():
        return None, "empty_query"

    query = _clean_query_text(query)
    if not query.strip():
        return None, "empty_query_after_clean"

    return {
        "session_id": session_id,
        "ply_start": ply_start,
        "ply_end": int(exchange.ply_end),
        "query": query,
        "relevant_exchange_ids": [exchange.exchange_id],
        "substantive": True,
    }, None


def generate_ground_truth(
    *,
    output_jsonl: Path,
    max_sessions: int | None = None,
    limit_exchanges: int | None = None,
    segmenter_min_exchange_chars: int | None = None,
    segmenter_max_ply_len: int | None = None,
    segmenter_min_assistant_chars: int | None = None,
) -> None:
    config = get_memory_v2_config()

    sessions_dir = get_sessions_dir()
    session_ids = list_session_ids()
    if max_sessions is not None:
        session_ids = session_ids[:max_sessions]

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    labels_written = 0
    labels_skipped: dict[str, int] = {}
    sessions_processed = 0
    exchanges_emitted = 0

    segmenter_params = {
        "min_exchange_chars": segmenter_min_exchange_chars or config.min_exchange_chars,
        "max_ply_len": segmenter_max_ply_len or config.max_ply_len,
        "min_assistant_chars": segmenter_min_assistant_chars or config.min_assistant_chars,
    }

    # Import-time update: path correctness for direct script execution.
    # (kept here intentionally so it happens after argument parsing in typical flow)
    # no-op for `python -m ...`
    # _ensure_import_path()

    with output_jsonl.open("w", encoding="utf-8") as f:
        for sid in session_ids:
            sessions_processed += 1
            messages = read_session(sid)
            if not messages:
                continue

            exchanges = segment_exchanges(
                sid,
                messages,
                min_exchange_chars=segmenter_params["min_exchange_chars"],
                max_ply_len=segmenter_params["max_ply_len"],
                min_assistant_chars=segmenter_params["min_assistant_chars"],
            )

            for ex in exchanges:
                if not getattr(ex, "has_substantive_assistant", False):
                    continue

                if limit_exchanges is not None and exchanges_emitted >= limit_exchanges:
                    break

                label, skip_reason = _label_one_exchange(
                    session_id=sid,
                    exchange=ex,
                    messages=messages,
                )
                if label is None:
                    if skip_reason:
                        labels_skipped[skip_reason] = labels_skipped.get(skip_reason, 0) + 1
                    continue

                f.write(json.dumps(label, ensure_ascii=False) + "\n")
                labels_written += 1
                exchanges_emitted += 1

            if limit_exchanges is not None and exchanges_emitted >= limit_exchanges:
                break

    meta_path = output_jsonl.with_suffix(".meta.json")
    meta = GroundTruthMeta(
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sessions_dir=str(sessions_dir),
        segmenter_params=segmenter_params,
        labels_written=labels_written,
        labels_skipped=labels_skipped,
        sessions_processed=sessions_processed,
    )
    meta_path.write_text(json.dumps(meta.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ground-truth] written={labels_written} skipped={labels_skipped} sessions={sessions_processed}")
    print(f"[ground-truth] output={output_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ground truth labels for memory_module_v2 (MRR, Scheme A).")
    parser.add_argument("--output", type=str, default="backend/memory_module_v2/eval/ground_truth.jsonl")
    parser.add_argument("--max-sessions", type=int, default=0, help="0 means no limit")
    parser.add_argument("--limit-exchanges", type=int, default=0, help="0 means no limit")
    args = parser.parse_args()

    max_sessions = None if args.max_sessions <= 0 else args.max_sessions
    limit_exchanges = None if args.limit_exchanges <= 0 else args.limit_exchanges

    generate_ground_truth(
        output_jsonl=Path(args.output),
        max_sessions=max_sessions,
        limit_exchanges=limit_exchanges,
    )


if __name__ == "__main__":
    main()

