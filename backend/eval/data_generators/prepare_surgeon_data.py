"""准备肿瘤外科训练数据：真实 Benchmark 筛选 + LLM 改写扩充。

来源：
- MedQA-USMLE 肿瘤外科题（~600 条）
- MedBullets 肿瘤外科病例（~200 条，可选）
- CMExam 中文肿瘤外科题（~400 条）
- DeepSeek 改写扩充（用真实题做 seed，~400 条）

输出格式：ChatML 格式（含 system/user/assistant），可直接用于 Qwen3 LoRA 训练。
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
from eval.datasets._oncology_keywords import is_surgeon_topic, SURGEON_KEYWORDS

ONCOLOGY_DIR = DATA_DIR / "oncology"
TRAINING_DIR = DATA_DIR / "training"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = TRAINING_DIR / "surgeon_train.jsonl"
# v3.2: 升级为多视角扩充（每条 seed 生成 3 视角），目标 ~3000 总数据
TARGET_AUGMENT_COUNT = 2000  # 从 400 升到 2000
ROLE_PROMPT_PATH = BACKEND_DIR / "workspace" / "roles" / "SURGEON.md"
TRAINING_SPLITS = ["train", "validation"]

# 多视角扩充：每条 seed 生成 3 个不同视角的对话
AUGMENT_VIEWS = [
    {
        "name": "clinical_decision",
        "instruction": "改写为临床决策视角：'58 岁男性，肺癌 T2N0M0...是否手术？怎么切？'",
    },
    {
        "name": "perioperative",
        "instruction": "改写为围手术期管理视角：'XX 患者拟行 XX 手术，围手术期评估和注意事项？'",
    },
    {
        "name": "differential",
        "instruction": "改写为外科鉴别视角：'XX 病灶，外科应该考虑哪些诊断？术中冰冻还是术后病理？'",
    },
]


def load_role_prompt() -> str:
    if ROLE_PROMPT_PATH.exists():
        return ROLE_PROMPT_PATH.read_text(encoding="utf-8")
    return "你是肿瘤外科医生，参与 Tumor Board 会诊。"


# ───────────── Step 1: 从 Benchmark 筛选真实数据 ─────────────

def collect_real_data() -> list[dict]:
    """从已下载的肿瘤子集中筛选外科相关题。"""
    real_records = []

    # MedQA 肿瘤子集
    for split in TRAINING_SPLITS:
        path = ONCOLOGY_DIR / f"medqa_{split}_oncology.jsonl"
        if not path.exists():
            continue
        for item in load_jsonl(path):
            text = (item.get("question", "")
                    + " " + " ".join(str(o) for o in item.get("options", []))
                    + " " + str(item.get("answer_text", "")))
            if is_surgeon_topic(text):
                real_records.append({
                    "source": "MedQA",
                    "question": item["question"],
                    "options": item.get("options", []),
                    "answer": item.get("answer_text", ""),
                    "raw": item,
                })

    # CMExam 肿瘤子集
    for split in TRAINING_SPLITS:
        path = ONCOLOGY_DIR / f"cmexam_{split}_oncology.jsonl"
        if not path.exists():
            continue
        for item in load_jsonl(path):
            text = json.dumps(item, ensure_ascii=False)
            if is_surgeon_topic(text):
                real_records.append({
                    "source": "CMExam",
                    "question": item.get("Question", item.get("question", "")),
                    "options": item.get("Options", item.get("options", [])),
                    "answer": item.get("Answer", item.get("answer", "")),
                    "explanation": item.get("Explanation", ""),
                    "raw": item,
                })

    # MedBullets 肿瘤子集（如有）
    for split in TRAINING_SPLITS:
        path = ONCOLOGY_DIR / f"medbullets_{split}_oncology.jsonl"
        if not path.exists():
            continue
        for item in load_jsonl(path):
            text = json.dumps(item, ensure_ascii=False)
            if is_surgeon_topic(text):
                real_records.append({
                    "source": "MedBullets",
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "raw": item,
                })

    print(f"[surgeon] 真实数据筛选：{len(real_records)} 条")
    return real_records


# ───────────── Step 2: 转换为 ChatML 格式 ─────────────

def to_chatml(record: dict, role_prompt: str) -> dict | None:
    """将真实题转为 ChatML 格式：system + user(病例/问题) + assistant(标准答+解析)"""
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


# ───────────── Step 3: LLM 改写扩充（用真实题做 seed）─────────────

def build_augment_prompt(seed_records: list[dict], view: dict) -> tuple[str, str]:
    """让 DeepSeek 把真实题按指定视角改写成 MDT 讨论格式。"""
    seeds_text = "\n\n".join(
        f"原题：{r['question']}\n标准答案：{r.get('answer', '')}"
        for r in seed_records
    )

    system = (
        "你是医学训练数据生成员。把医学考试题改写成 Tumor Board "
        "肿瘤外科医生回答风格的高质量训练样本。"
        "严格基于原题医学事实，不编造。回答按外科结构化输出。输出 JSON。"
    )

    user = f"""基于下面 3 道真实考题，按【{view['name']}】视角改写 3 对训练对话：

{seeds_text}

【视角说明】
{view['instruction']}

要求：
1. 改写为口语化的 MDT 场景，保留原题医学事实
2. 答案按"可切除性 → 术式 → 淋巴清扫 → 围手术期"结构
3. 长度 200-450 字，必须含具体决策依据（不能只说"建议手术"）
4. 不同视角的输出必须有明显差异

输出 JSON：
{{"items": [{{"user": "...", "assistant": "..."}}, ...]}}
"""
    return system, user


def augment_with_llm(real_records: list[dict], role_prompt: str, target: int) -> int:
    """多视角扩充：每条 seed 按 3 个视角生成不同对话。"""
    if len(real_records) < 3:
        print("[surgeon] 真实数据不足 3 条，无法做 seed 扩充")
        return 0

    existing = load_jsonl(OUTPUT)
    augmented_count = sum(1 for r in existing if r.get("source") == "augmented")
    seen_users = {r["messages"][1]["content"] for r in existing if "messages" in r}

    view_idx = 0  # 轮换视角
    while augmented_count < target:
        seed = random.sample(real_records, 3)
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
                print(f"[surgeon-aug] 进度 {augmented_count}/{target} (视角: {view['name']})")
        except Exception as exc:
            print(f"[surgeon-aug] {exc}")
            continue

    return augmented_count


# ───────────── Main ─────────────

def main():
    role_prompt = load_role_prompt()

    # Step 1+2: 真实数据 → ChatML（重新生成保证完整性）
    real = collect_real_data()
    if not real:
        print("[surgeon] ⚠️ 真实数据为空，请先下载 benchmark")
        return

    # 加载已有的扩充数据（保留，避免重复花 API 钱）
    existing_augmented = []
    if OUTPUT.exists():
        for r in load_jsonl(OUTPUT):
            if r.get("source") == "augmented":
                existing_augmented.append(r)
        print(f"[surgeon] 已有扩充: {len(existing_augmented)} 条")

    # 重新生成所有真实 ChatML
    real_chatml = []
    for r in real:
        cm = to_chatml(r, role_prompt)
        if cm is not None:
            real_chatml.append(cm)
    print(f"[surgeon] 真实 ChatML 生成: {len(real_chatml)} 条")

    # 合并所有数据
    all_records = real_chatml + existing_augmented

    # Step 3: 质量过滤 + 去重（LIMA 思路）
    print("\n[surgeon] === 应用质量过滤（LIMA 思路）===")
    result = filter_records(
        all_records,
        role_keywords=SURGEON_KEYWORDS,
        min_len=30,
        max_len=3000,
    )
    filtered = result["kept"]

    # 写回过滤后的数据
    OUTPUT.write_text("")
    for r in filtered:
        append_jsonl(r, OUTPUT)
    print(f"[surgeon] 过滤后写入: {len(filtered)} 条")

    # Step 4: LLM 多视角扩充（补足到目标）
    real = collect_real_data()  # 用真实数据做 seed
    augment_with_llm(real, role_prompt, TARGET_AUGMENT_COUNT)

    # Step 5: 最后再过一遍质量+去重
    final_raw = load_jsonl(OUTPUT)
    print("\n[surgeon] === 最终质量门 ===")
    final_result = filter_records(
        final_raw,
        role_keywords=SURGEON_KEYWORDS,
        min_len=30,
        max_len=3000,
    )
    final = final_result["kept"]
    OUTPUT.write_text("")
    for r in final:
        append_jsonl(r, OUTPUT)

    print(f"\n[surgeon] ✅ 完成，共 {len(final)} 条训练数据（已应用 LIMA 质量过滤 + 去重）")
    print(f"  真实: {sum(1 for r in final if r.get('source') != 'augmented')}")
    print(f"  扩充: {sum(1 for r in final if r.get('source') == 'augmented')}")
    # 视角分布
    from collections import Counter
    view_dist = Counter(r.get("view", "real") for r in final)
    print(f"  视角分布: {dict(view_dist)}")


if __name__ == "__main__":
    random.seed(42)
    main()
