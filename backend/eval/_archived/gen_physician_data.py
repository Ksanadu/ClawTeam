"""生成主治医生 LoRA 训练数据。

格式 (JSONL，符合 ChatML / Qwen 格式)：
{
  "messages": [
    {"role": "system", "content": "你是一名主治医生..."},
    {"role": "user", "content": "我最近头痛..."},
    {"role": "assistant", "content": "[结构化诊断回复]"}
  ]
}
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

OUTPUT = DATA_DIR / "physician_train.jsonl"
TARGET_COUNT = 800

# 复用现有的角色 prompt
ROLE_PROMPT_PATH = BACKEND_DIR / "workspace" / "roles" / "PHYSICIAN.md"

SCENARIOS = [
    "呼吸系统症状（咳嗽、哮喘、胸闷、呼吸困难等）",
    "消化系统症状（腹痛、腹泻、便秘、恶心呕吐等）",
    "心血管症状（胸痛、心悸、晕厥、高血压等）",
    "神经系统症状（头痛、头晕、失眠、麻木等）",
    "骨骼肌肉症状（关节痛、腰背痛、扭伤等）",
    "皮肤症状（皮疹、瘙痒、红肿等）",
    "全身症状（发热、乏力、体重变化等）",
    "妇科症状（月经异常、白带异常、孕期问题等）",
    "儿科症状（小儿发烧、咳嗽、腹泻、湿疹等）",
    "老年常见问题（认知下降、跌倒、多病共存等）",
    "慢病管理（糖尿病、高血压、高血脂、痛风等随访）",
    "急症识别（胸痛/卒中/急腹症等需要急诊就医的判断）",
]


def load_role_prompt() -> str:
    if ROLE_PROMPT_PATH.exists():
        return ROLE_PROMPT_PATH.read_text(encoding="utf-8")
    return "你是主治医生，请专业回答医疗问题。"


def build_prompt(scenario: str, batch_size: int = 3) -> tuple[str, str]:
    role_prompt = load_role_prompt()

    system = (
        "你是医疗 NLP 训练数据生成员。你需要生成主治医生的训练样本，"
        "包括用户提问和高质量的主治回答。"
        "回答要符合主治医生 prompt 中定义的结构化要求。"
        "请严格按 JSON 输出。"
    )

    user = f"""请生成 {batch_size} 对中文「患者提问 + 主治医生回答」训练数据。

场景：{scenario}

主治医生回答规范（严格遵循）：
{role_prompt}

要求：
1. 患者提问 30-150 字，模仿真实就诊咨询
2. 主治回答 200-500 字，按"主诉分析 → 鉴别诊断 → 建议检查 → 初步治疗方向"结构
3. 回答必须包含免责声明
4. 多样化场景：成人/儿童/老人，急/慢，轻/重
5. 不要给出确定性诊断（用"考虑XX可能"）

输出 JSON：
{{
  "items": [
    {{"user": "患者问题", "assistant": "主治回答"}},
    ...
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
