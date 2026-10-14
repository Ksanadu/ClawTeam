"""下载 MedQA-USMLE 数据集，并筛选肿瘤子集。

数据来源：HuggingFace `bigbio/med_qa` 或 `GBaker/MedQA-USMLE-4-options`
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
RAW_DIR.mkdir(parents=True, exist_ok=True)
ONCOLOGY_DIR.mkdir(parents=True, exist_ok=True)


def download_medqa():
    """下载 MedQA-USMLE。"""
    from datasets import load_dataset
    import os

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    print("[MedQA] 下载 GBaker/MedQA-USMLE-4-options（约 100MB）...")
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options")
    except Exception as exc:
        print(f"主源失败 ({exc})，尝试备源...")
        ds = load_dataset("bigbio/med_qa")

    return ds


def to_unified_format(item: dict) -> dict:
    """统一格式：question / options / answer / explanation"""
    # MedQA-USMLE-4-options 字段
    question = item.get("question") or item.get("Question") or ""
    options = item.get("options")  # 兼容多种格式
    if isinstance(options, dict):
        if all(key in options for key in ["A", "B", "C", "D"]):
            opt_list = [options[key] for key in ["A", "B", "C", "D"]]
        elif all(key in options for key in ["opa", "opb", "opc", "opd"]):
            opt_list = [options[key] for key in ["opa", "opb", "opc", "opd"]]
        elif "value" in options and isinstance(options["value"], list):
            opt_list = options["value"]
        else:
            opt_list = [value for _, value in sorted(options.items())]
    elif isinstance(options, list):
        opt_list = options
    else:
        opt_list = [item.get(f"op{c}", "") for c in "abcd"]
    opt_list = [str(option).strip() for option in opt_list if str(option).strip()]

    answer_idx = item.get("answer_idx")
    if answer_idx in (None, ""):
        answer_idx = item.get("answer")
    if answer_idx in (None, ""):
        answer_idx = item.get("cop", 0)
    if isinstance(answer_idx, str):
        answer_idx = answer_idx.strip()
        if answer_idx.upper() in "ABCD":
            answer_idx = "ABCD".index(answer_idx.upper())
        elif answer_idx.isdigit():
            answer_idx = int(answer_idx)

    return {
        "question": question,
        "options": opt_list,
        "answer_idx": answer_idx,
        "answer_text": opt_list[answer_idx] if isinstance(answer_idx, int) and 0 <= answer_idx < len(opt_list) else "",
        "source": "MedQA-USMLE",
    }


def main():
    ds = download_medqa()

    # 处理所有 split
    for split_name in ds:
        split = ds[split_name]
        unified = [to_unified_format(item) for item in split]

        # 保存全集
        raw_path = RAW_DIR / f"medqa_{split_name}.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for item in unified:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[MedQA/{split_name}] 全集 {len(unified)} 条 → {raw_path}")

        # 筛选肿瘤子集
        oncology_items = [
            item for item in unified
            if is_oncology(item["question"] + " ".join(str(o) for o in item["options"]))
        ]
        onc_path = ONCOLOGY_DIR / f"medqa_{split_name}_oncology.jsonl"
        with open(onc_path, "w", encoding="utf-8") as f:
            for item in oncology_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[MedQA/{split_name}] 肿瘤子集 {len(oncology_items)} 条 → {onc_path}")


if __name__ == "__main__":
    main()
