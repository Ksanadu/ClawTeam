"""准备肿瘤内科训练数据：真实 Benchmark 筛选 + LLM 改写扩充。

与 surgeon 同结构，但筛选关键词换成内科相关（化疗/靶向/免疫等）。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from _common import (
    BACKEND_DIR,
    DATA_DIR,
    append_jsonl,
    call_llm,
    load_jsonl,
    parse_json_safely,
)
from _quality import filter_records
from eval.datasets._oncology_keywords import (
    is_oncologist_topic,
    MEDICAL_ONCOLOGIST_KEYWORDS,
)

ONCOLOGY_DIR = DATA_DIR / "oncology"
TRAINING_DIR = DATA_DIR / "training"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = TRAINING_DIR / "oncologist_train.jsonl"
# v3.2: 多视角扩充 + 质量过滤
TARGET_AUGMENT_COUNT = 2000
ROLE_PROMPT_PATH = BACKEND_DIR / "workspace" / "roles" / "MEDICAL_ONCOLOGIST.md"
TRAINING_SPLITS = ["train", "validation"]

AUGMENT_VIEWS = [
    {
        "name": "regimen_selection",
        "instruction": "改写为方案选择视角：'XX 患者，分子标志物 YY，应该选什么化疗/靶向方案？剂量周期？'",
    },
    {
        "name": "adverse_management",
        "instruction": "改写为不良反应管理视角：'患者用 XX 药出现 YY 反应，怎么处理？是否减量？'",
    },
    {
        "name": "treatment_sequencing",
        "instruction": "改写为治疗时序视角：'XX 患者，新辅助 vs 辅助？同步 vs 序贯？时机怎么定？'",
    },
]


def load_role_prompt() -> str:
    if ROLE_PROMPT_PATH.exists():
        return ROLE_PROMPT_PATH.read_text(encoding="utf-8")
    return "你是肿瘤内科医生，参与 Tumor Board 会诊。"


def collect_real_data() -> list[dict]:
    real = []

    # MedQA
    for split in TRAINING_SPLITS:
        path = ONCOLOGY_DIR / f"medqa_{split}_oncology.jsonl"
        if not path.exists():
            continue
        for item in load_jsonl(path):
            text = (item.get("question", "")
                    + " " + " ".join(str(o) for o in item.get("options", []))
                    + " " + str(item.get("answer_text", "")))
            if is_oncologist_topic(text):
                real.append({
                    "source": "MedQA",
                    "question": item["question"],
                    "options": item.get("options", []),
                    "answer": item.get("answer_text", ""),
                    "raw": item,
                })

    # CMExam
    for split in TRAINING_SPLITS:
        path = ONCOLOGY_DIR / f"cmexam_{split}_oncology.jsonl"
        if not path.exists():
            continue
        for item in load_jsonl(path):
            text = json.dumps(item, ensure_ascii=False)
            if is_oncologist_topic(text):
                real.append({
                    "source": "CMExam",
                    "question": item.get("Question", item.get("question", "")),
                    "options": item.get("Options", item.get("options", [])),
                    "answer": item.get("Answer", item.get("answer", "")),
                    "explanation": item.get("Explanation", ""),
                    "raw": item,
                })

    # PubMedQA 肿瘤子集（QA 形式，作为补充）
    for split in TRAINING_SPLITS:
        path = ONCOLOGY_DIR / f"pubmedqa_{split}_oncology.jsonl"
        if not path.exists():
            continue
        for item in load_jsonl(path):
            real.append({
                "source": "PubMedQA",
                "question": item.get("question", ""),
                "answer": item.get("long_answer", item.get("final_decision", "")),
                "raw": item,
            })

    print(f"[oncologist] 真实数据：{len(real)} 条")
    return real


def to_chatml(record: dict, role_prompt: str) -> dict | None:
    question = (record.get("question") or "").strip()
    if not question:
        return None

    options = record.get("options") or []
    answer = (record.get("answer") or "").strip()
    explanation = (record.get("explanation") or "").strip()

    if isinstance(options, list) and options:
        opt_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))
        user_text = f"{question}\n\n{opt_text}"
    else:
        user_text = question

    if explanation:
        assistant_text = f"答案：{answer}\n\n解析：{explanation}"
    else:
        assistant_text = f"答案：{answer}" if answer else ""

    if not assistant_text:
        return None

    return {
        "messages": [
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "source": record["source"],
    }


def build_augment_prompt(seed_records: list[dict], view: dict) -> tuple[str, str]:
    seeds_text = "\n\n".join(
        f"原题：{r['question']}\n标准答案：{r.get('answer', '')}"
        for r in seed_records
    )

    system = (
        "你是医学训练数据生成员。把医学考试题改写成 Tumor Board 肿瘤内科医生回答风格。"
        "严格基于原题医学事实，不编造剂量。回答按内科结构化输出。输出 JSON。"
    )

    user = f"""基于下面 3 道真实考题，按【{view['name']}】视角改写 3 对训练对话：

{seeds_text}

【视角说明】
{view['instruction']}

要求：
1. 改写为口语化 MDT 场景，保留原题医学事实
2. 答案按"治疗目标 → 推荐方案 → 剂量周期 → 不良反应 → 与其他治疗协调"结构
3. 长度 200-450 字，必须含具体剂量/药物名（不能只说"建议化疗"）
4. 不同视角的输出必须有明显差异

输出 JSON：
{{"items": [{{"user": "...", "assistant": "..."}}, ...]}}
"""
    return system, user


def augment_with_llm(real: list[dict], role_prompt: str, target: int) -> int:
    """多视角扩充。"""
    if len(real) < 3:
        return 0

    existing = load_jsonl(OUTPUT)
    augmented_count = sum(1 for r in existing if r.get("source") == "augmented")
    seen_users = {r["messages"][1]["content"] for r in existing if "messages" in r}

    view_idx = 0
    while augmented_count < target:
        seed = random.sample(real, 3)
        view = AUGMENT_VIEWS[view_idx % len(AUGMENT_VIEWS)]
        view_idx += 1
        system, user = build_augment_prompt(seed, view)

        try:
            response = call_llm(system, user, temperature=0.85, response_format_json=True)
            payload = parse_json_safely(response)
            if not payload or "items" not in payload:
                continue

            for item in payload["items"]:
                user_text = (item.get("user") or "").strip()
                asst_text = (item.get("assistant") or "").strip()
                if not user_text or not asst_text or user_text in seen_users:
                    continue

                record = {
                    "messages": [
                        {"role": "system", "content": role_prompt},
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": asst_text},
                    ],
                    "source": "augmented",
                    "view": view["name"],
                }
                append_jsonl(record, OUTPUT)
                seen_users.add(user_text)
                augmented_count += 1
                if augmented_count >= target:
                    break

            if augmented_count % 30 == 0:
                print(f"[oncologist-aug] 进度 {augmented_count}/{target} (视角: {view['name']})")
        except Exception as exc:
            print(f"[oncologist-aug] {exc}")
            continue

    return augmented_count


def main():
    role_prompt = load_role_prompt()

    real = collect_real_data()
    if not real:
        print("[oncologist] ⚠️ 真实数据为空，请先下载 benchmark")
        return

    existing_augmented = []
    if OUTPUT.exists():
        for r in load_jsonl(OUTPUT):
            if r.get("source") == "augmented":
                existing_augmented.append(r)
        print(f"[oncologist] 已有扩充: {len(existing_augmented)} 条")

    real_chatml = []
    for r in real:
        cm = to_chatml(r, role_prompt)
        if cm is not None:
            real_chatml.append(cm)
    print(f"[oncologist] 真实 ChatML: {len(real_chatml)} 条")

    all_records = real_chatml + existing_augmented

    print("\n[oncologist] === 应用质量过滤（LIMA 思路）===")
    result = filter_records(
        all_records,
        role_keywords=MEDICAL_ONCOLOGIST_KEYWORDS,
        min_len=30,
        max_len=3000,
    )
    filtered = result["kept"]

    OUTPUT.write_text("")
    for r in filtered:
        append_jsonl(r, OUTPUT)
    print(f"[oncologist] 过滤后写入: {len(filtered)} 条")

    real = collect_real_data()
    augment_with_llm(real, role_prompt, TARGET_AUGMENT_COUNT)

    final_raw = load_jsonl(OUTPUT)
    print("\n[oncologist] === 最终质量门 ===")
    final_result = filter_records(
        final_raw,
        role_keywords=MEDICAL_ONCOLOGIST_KEYWORDS,
        min_len=30,
        max_len=3000,
    )
    final = final_result["kept"]
    OUTPUT.write_text("")
    for r in final:
        append_jsonl(r, OUTPUT)

    print(f"\n[oncologist] ✅ 完成，共 {len(final)} 条（已应用 LIMA 质量过滤 + 去重）")
    print(f"  真实: {sum(1 for r in final if r.get('source') != 'augmented')}")
    print(f"  扩充: {sum(1 for r in final if r.get('source') == 'augmented')}")
    from collections import Counter
    view_dist = Counter(r.get("view", "real") for r in final)
    print(f"  视角分布: {dict(view_dist)}")


if __name__ == "__main__":
    random.seed(42)
    main()
