"""下载 CMExam 中文医学考试数据集，筛选肿瘤子集。

数据来源：HuggingFace `fzkuji/CMExam` 或类似仓库
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.datasets._oncology_keywords import is_oncology_zh

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
ONCOLOGY_DIR = DATA_DIR / "oncology"

CANDIDATE_REPOS = [
    "fzkuji/CMExam",
    "BAAI/CMExam",
]


def main():
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    from datasets import load_dataset

    print("[CMExam] 尝试下载...")
    ds = None
    for repo in CANDIDATE_REPOS:
        try:
            ds = load_dataset(repo)
            print(f"[CMExam] 成功从 {repo} 加载")
            break
        except Exception as exc:
            print(f"  {repo} 失败: {exc}")

    if ds is None:
        print("[CMExam] ⚠️ 所有源失败。如需使用，请手动从 https://github.com/williamliujl/CMExam 下载")
        return

    for split_name in ds:
        split = ds[split_name]
        records = list(split)
        raw_path = RAW_DIR / f"cmexam_{split_name}.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[CMExam/{split_name}] {len(records)} 条 → {raw_path}")

        oncology_items = [r for r in records if is_oncology_zh(json.dumps(r, ensure_ascii=False))]
        onc_path = ONCOLOGY_DIR / f"cmexam_{split_name}_oncology.jsonl"
        with open(onc_path, "w", encoding="utf-8") as f:
            for r in oncology_items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[CMExam/{split_name}] 肿瘤子集 {len(oncology_items)} 条 → {onc_path}")


if __name__ == "__main__":
    main()
