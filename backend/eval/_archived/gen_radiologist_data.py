"""生成影像科 LoRA 训练数据。

⚠️ 文本版本（不含图片）：训练影像科 Agent 在文本场景下的检查建议能力。
真正的多模态训练（图文对）请用 SLAKE / VQA-RAD / IU X-Ray 公开数据集，
将在 05_train_radiologist_vl.ipynb 中通过 HuggingFace datasets 加载。

本脚本生成的数据用于：
- 文本场景下"应该做什么影像检查"的咨询回答
- 影像报告解读咨询（用户描述报告，询问意义）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    BACKEND_DIR,
    DATA_DIR,
    append_jsonl,
    call_llm,
    load_jsonl,
    parse_json_safely,
)

OUTPUT = DATA_DIR / "radiologist_text_train.jsonl"
TARGET_COUNT = 600

ROLE_PROMPT_PATH = BACKEND_DIR / "workspace" / "roles" / "RADIOLOGIST.md"

SCENARIOS = [
    "胸部检查选择（咳嗽/胸痛 → X线/CT/低剂量CT）",
    "腹部检查选择（腹痛/肝胆问题 → 超声/CT/MRI）",
    "颅脑检查（头痛/头晕/卒中筛查 → CT/MRI/MRA）",
    "脊柱检查（颈/胸/腰椎症状 → X线/CT/MRI）",
    "关节检查（膝/肩/髋/腕 → 超声/MRI/CT）",
    "甲状腺/乳腺检查（结节 → 超声/钼靶/MRI）",
    "妇科检查（盆腔超声/MRI/HSG）",
    "心血管检查（冠脉CTA/心脏超声/MRI）",
    "影像报告解读咨询（结节大小、BI-RADS分级等）",
    "造影检查的适应证与风险",
    "辐射剂量与孕妇/儿童检查选择",
    "复查频率建议（结节随访等）",
]


def load_role_prompt() -> str:
    if ROLE_PROMPT_PATH.exists():
        return ROLE_PROMPT_PATH.read_text(encoding="utf-8")
    return "你是影像科医生，请专业回答影像学检查问题。"


def build_prompt(scenario: str, batch_size: int = 3) -> tuple[str, str]:
    role_prompt = load_role_prompt()

    system = (
        "你是医疗 NLP 训练数据生成员，负责生成影像科医生的高质量训练样本。"
        "回答必须符合影像科 prompt 中定义的结构化要求。"
        "严格按 JSON 输出。"
    )

    user = f"""请生成 {batch_size} 对中文「影像咨询 + 影像科回答」训练数据。

场景：{scenario}

影像科回答规范：
{role_prompt}

要求：
1. 患者提问 30-150 字
2. 回答 200-500 字，按"推荐检查 → 检查理由 → 注意事项 → 预期所见 → 后续建议"结构
3. 说明为何选这个检查（敏感性/特异性/辐射/费用对比）
4. 涉及辐射时给出剂量评估
5. 使用国际分级（BI-RADS / Lung-RADS / LI-RADS 等）
6. 必须提醒"影像学诊断需结合临床，请以实际报告为准"

输出 JSON：
{{
  "items": [
    {{"user": "...", "assistant": "..."}}
  ]
}}
"""
    return system, user


def main() -> None:
    existing = load_jsonl(OUTPUT)
    seen_users = {
        item.get("messages", [{}, {}, {}])[1].get("content", "") for item in existing
    }
    print(f"[INFO] 已有 {len(existing)} 条，目标 {TARGET_COUNT} 条")

    role_prompt = load_role_prompt()

    while len(existing) < TARGET_COUNT:
        scenario = SCENARIOS[len(existing) % len(SCENARIOS)]
        system, user = build_prompt(scenario, batch_size=3)

        try:
            response = call_llm(system, user, temperature=0.85, response_format_json=True)
            payload = parse_json_safely(response)
            if not payload or "items" not in payload:
                continue

            new_count = 0
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
                    "scenario": scenario,
                }
                append_jsonl(record, OUTPUT)
                seen_users.add(user_text)
                existing.append(record)
                new_count += 1

            print(f"[+{new_count}] 进度 {len(existing)}/{TARGET_COUNT} ({scenario})")

        except Exception as exc:
            print(f"[ERR] {exc}")
            continue

    print(f"[DONE] 共生成 {len(existing)} 条到 {OUTPUT}")


if __name__ == "__main__":
    main()
