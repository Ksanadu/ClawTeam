"""下载 MedBullets 复杂临床病例数据集。

数据来源：HuggingFace `XuhuiZhou/MedBullets`（也可能在其他仓库，运行时探测）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.datasets._oncology_keywords import is_oncology

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
ONCOLOGY_DIR = DATA_DIR / "oncology"

CANDIDATE_REPOS = [
    "HiTZ/Medical-mT5-test",  # MedBullets 的镜像之一
    "ucinlp/medbullets",
]


def main():
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    from datasets import load_dataset

    print("[MedBullets] 尝试下载...")
    ds = None
    last_err = None
    for repo in CANDIDATE_REPOS:
        try:
            ds = load_dataset(repo)
            print(f"[MedBullets] 成功从 {repo} 加载")
            break
        except Exception as exc:
            last_err = exc
            print(f"  {repo} 失败: {exc}")

    if ds is None:
        print(f"[MedBullets] ⚠️ 所有源都失败：{last_err}")
        print("[MedBullets] 请手动检查 HF Hub 上的可用仓库，或跳过本数据集")
        print("[MedBullets] 替代方案：用 MedQA + CMExam 已经够 benchmark 用")
        return

    for split_name in ds:
        split = ds[split_name]
        records = []
        for item in split:
            text = json.dumps(item, ensure_ascii=False)
            records.append(item)

        raw_path = RAW_DIR / f"medbullets_{split_name}.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[MedBullets/{split_name}] {len(records)} 条 → {raw_path}")

        # 筛选肿瘤
        oncology_items = [r for r in records if is_oncology(json.dumps(r, ensure_ascii=False))]
        onc_path = ONCOLOGY_DIR / f"medbullets_{split_name}_oncology.jsonl"
        with open(onc_path, "w", encoding="utf-8") as f:
            for r in oncology_items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[MedBullets/{split_name}] 肿瘤子集 {len(oncology_items)} 条 → {onc_path}")


if __name__ == "__main__":
    main()
