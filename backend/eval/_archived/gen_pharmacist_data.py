"""生成药师 LoRA 训练数据。"""

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

OUTPUT = DATA_DIR / "pharmacist_train.jsonl"
TARGET_COUNT = 800

ROLE_PROMPT_PATH = BACKEND_DIR / "workspace" / "roles" / "PHARMACIST.md"

SCENARIOS = [
    "药物相互作用（DDI）：常见联合用药风险",
    "特殊人群剂量调整（老年、儿童、孕妇、肝肾功能不全）",
    "抗生素合理使用（适应证、疗程、耐药）",
    "镇痛药选择与禁忌（NSAIDs、阿片类）",
    "心血管药物（降压药、抗凝药、他汀）使用",
    "糖尿病用药（口服降糖药、胰岛素）",
    "精神科药物（抗抑郁、抗焦虑、安眠药）",
    "中成药与西药联用注意事项",
    "OTC 药物的合理选择",
    "药物过敏与不良反应识别",
    "药物剂型与给药途径选择",
    "药物经济学（同类替代、性价比）",
]


def load_role_prompt() -> str:
    if ROLE_PROMPT_PATH.exists():
        return ROLE_PROMPT_PATH.read_text(encoding="utf-8")
    return "你是临床药师，请专业回答用药安全问题。"


def build_prompt(scenario: str, batch_size: int = 3) -> tuple[str, str]:
    role_prompt = load_role_prompt()

    system = (
        "你是医疗 NLP 训练数据生成员，负责生成临床药师的高质量训练样本。"
        "回答必须符合临床药师 prompt 中定义的结构化要求。"
        "严格按 JSON 输出。"
    )

    user = f"""请生成 {batch_size} 对中文「用药咨询 + 药师回答」训练数据。

场景：{scenario}

药师回答规范：
{role_prompt}

要求：
1. 患者/家属提问 30-150 字
2. 药师回答 200-500 字，按"用药建议 → 剂量途径 → 相互作用风险 → 禁忌注意事项 → 替代方案"结构
3. 包含具体剂量、给药频率、疗程参考
4. 标注严重程度（X/D/C/B 级或文字描述）
5. 涉及特殊人群时主动调整剂量
6. 必须以"具体用药请遵医嘱"结尾

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
