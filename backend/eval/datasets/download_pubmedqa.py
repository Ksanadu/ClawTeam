"""下载 PubMedQA，筛选肿瘤相关问答。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.datasets._oncology_keywords import is_oncology_en

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
ONCOLOGY_DIR = DATA_DIR / "oncology"


def main():
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    from datasets import load_dataset

    print("[PubMedQA] 下载 qiaojin/PubMedQA...")
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    except Exception as exc:
        print(f"[PubMedQA] 失败: {exc}")
        return

    for split_name in ds:
        split = ds[split_name]
        records = list(split)
        raw_path = RAW_DIR / f"pubmedqa_{split_name}.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[PubMedQA/{split_name}] {len(records)} 条 → {raw_path}")

        oncology_items = [
            r for r in records
            if is_oncology_en(str(r.get("question", "")) + " " + str(r.get("long_answer", "")))
        ]
        onc_path = ONCOLOGY_DIR / f"pubmedqa_{split_name}_oncology.jsonl"
        with open(onc_path, "w", encoding="utf-8") as f:
            for r in oncology_items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[PubMedQA/{split_name}] 肿瘤子集 {len(oncology_items)} 条 → {onc_path}")


if __name__ == "__main__":
    main()
