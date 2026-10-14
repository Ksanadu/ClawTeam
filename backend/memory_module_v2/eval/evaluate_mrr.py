from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _ensure_import_path() -> None:
    repo_backend = Path(__file__).resolve().parents[2]  # .../backend
    if str(repo_backend) not in sys.path:
        sys.path.insert(0, str(repo_backend))


_ensure_import_path()

from memory_module_v2.domain.enums import SearchMode
from memory_module_v2.domain.models import MemorySearchFilters
from memory_module_v2.service.api import search_memory


@dataclass
class MrrStats:
    mode: str
    top_k: int
    label_count: int
    evaluated_count: int
    mrr_sum: float
    hit_count: int
    failure_count: int

    def to_dict(self) -> dict[str, Any]:
        label_count = int(self.label_count)
        evaluated_count = int(self.evaluated_count)
        # We define MRR over all labels (including failures as RR=0).
        denom = label_count if label_count > 0 else 1
        mrr = self.mrr_sum / denom
        hit_rate = self.hit_count / denom
        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "label_count": label_count,
            "evaluated_count": evaluated_count,
            "mrr": mrr,
            "hit_rate": hit_rate,
            "failure_count": self.failure_count,
        }


def _iter_ground_truth(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line), line_no
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc


def evaluate_mrr(
    *,
    ground_truth_jsonl: Path,
    output_json: Path,
    mode: SearchMode,
    top_k: int,
    restrict_session_ids: list[str] | None = None,
    debug_first_n: int = 0,
) -> None:
    if not ground_truth_jsonl.exists():
        raise FileNotFoundError(str(ground_truth_jsonl))

    filters = None
    if restrict_session_ids:
        filters = MemorySearchFilters(session_ids=restrict_session_ids)

    stats = MrrStats(
        mode=str(mode),
        top_k=top_k,
        label_count=0,
        evaluated_count=0,
        mrr_sum=0.0,
        hit_count=0,
        failure_count=0,
    )

    per_query_debug: list[dict[str, Any]] = []

    for item, line_no in _iter_ground_truth(ground_truth_jsonl):
        stats.label_count += 1
        query = str(item.get("query", "") or "")
        relevant = item.get("relevant_exchange_ids") or []
        if not query.strip() or not relevant:
            continue

        try:
            response = search_memory(
                query=query,
                mode=mode,
                top_k=top_k,
                filters=filters,
            )
        except Exception:
            stats.failure_count += 1
            continue

        hit_rr = 0.0
        hit_rank = None
        # response.hits are already in rank order; MemoryHit.rank is 1-based.
        for hit in response.hits:
            if hit.exchange_id in relevant:
                hit_rank = int(hit.rank)
                hit_rr = 1.0 / hit_rank
                break

        stats.evaluated_count += 1
        stats.mrr_sum += hit_rr
        if hit_rr > 0:
            stats.hit_count += 1

        if debug_first_n and stats.evaluated_count <= debug_first_n:
            per_query_debug.append(
                {
                    "line_no": line_no,
                    "query_len": len(query),
                    "relevant_exchange_ids": relevant,
                    "hit_rank": hit_rank,
                    "rr": hit_rr,
                    "top_hits": [
                        {"rank": int(h.rank), "exchange_id": h.exchange_id, "score": h.scores.get("fused", 0.0)}
                        for h in response.hits[: min(10, len(response.hits))]
                    ],
                }
            )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "summary": stats.to_dict(),
        "debug_first_queries": per_query_debug if per_query_debug else None,
    }
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[mrr-eval] saved: {output_json}")
    print("[mrr-eval] summary:", result["summary"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MRR for memory_module_v2 retrieval.")
    parser.add_argument("--labels", type=str, default="backend/memory_module_v2/eval/ground_truth.jsonl")
    parser.add_argument("--output", type=str, default="backend/memory_module_v2/eval/mrr_results.json")
    parser.add_argument("--mode", type=str, default="hybrid_cross", choices=["dense_distilled", "keyword_verbatim", "hybrid_cross"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--restrict-session-ids", type=str, default="", help="comma-separated session_ids (optional)")
    parser.add_argument("--debug-first-n", type=int, default=0)
    args = parser.parse_args()

    restrict_session_ids = None
    if args.restrict_session_ids.strip():
        restrict_session_ids = [s.strip() for s in args.restrict_session_ids.split(",") if s.strip()]

    evaluate_mrr(
        ground_truth_jsonl=Path(args.labels),
        output_json=Path(args.output),
        mode=SearchMode(args.mode),
        top_k=args.top_k,
        restrict_session_ids=restrict_session_ids,
        debug_first_n=args.debug_first_n,
    )


if __name__ == "__main__":
    main()

