"""生成路由分类训练数据。

输出格式 (JSONL)：
{
  "query": "我最近咳嗽两周，吃什么药好？",
  "labels": {"physician": 1, "pharmacist": 1, "radiologist": 0}
}

每条数据由 LLM 一次性生成 query + 三个角色标签。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    DATA_DIR,
    append_jsonl,
    call_llm,
    load_jsonl,
    parse_json_safely,
)

OUTPUT = DATA_DIR / "router_train.jsonl"
TARGET_COUNT = 1000  # 目标生成总数

# ── 路由分类的核心规则（让 LLM 学会标注） ──────────────────────────

ROUTING_RULES = """
ClawTeam 是一个医疗多智能体系统，有 3 个角色：
1. physician（主治医生）：负责诊断路径、鉴别诊断、初步治疗方向
2. pharmacist（临床药师）：负责用药安全、药物相互作用、剂量、禁忌
3. radiologist（影像科医生）：负责影像检查的适应证、CT/MRI/X光/超声等检查建议与解读

路由规则：
- physician 始终为 1（任何医疗问题都需要主治意见）
- pharmacist 为 1：问题涉及药物、剂量、用药安全、相互作用、副作用、过敏、禁忌
- radiologist 为 1：问题涉及影像检查（CT/MRI/X光/超声/造影等）、检查报告解读
"""

GENERATION_PROMPTS = [
    # 不同场景下让 LLM 生成多样化数据
    {
        "scene": "纯诊断问题（只需要主治）",
        "examples": "如：头痛三天怎么回事？/ 老人最近食欲不振是什么原因？",
        "expected": '{"physician": 1, "pharmacist": 0, "radiologist": 0}',
    },
    {
        "scene": "用药咨询（需要主治+药师）",
        "examples": "如：孕妇感冒能吃布洛芬吗？/ 高血压患者怎么吃阿司匹林？",
        "expected": '{"physician": 1, "pharmacist": 1, "radiologist": 0}',
    },
    {
        "scene": "影像检查问题（需要主治+影像科）",
        "examples": "如：腰痛要做 CT 还是 MRI？/ 胸片显示肺纹理增粗严重吗？",
        "expected": '{"physician": 1, "pharmacist": 0, "radiologist": 1}',
    },
    {
        "scene": "复杂综合问题（主治+药师+影像科都需要）",
        "examples": "如：肺结节 8mm 要不要做 CT 复查并且预防性吃药？/ 长期服降压药的人腰椎 MRI 异常该怎么处理？",
        "expected": '{"physician": 1, "pharmacist": 1, "radiologist": 1}',
    },
    {
        "scene": "症状+影像问题（主治+影像科）",
        "examples": "如：体检发现甲状腺结节要不要做超声造影？",
        "expected": '{"physician": 1, "pharmacist": 0, "radiologist": 1}',
    },
    {
        "scene": "用药+症状问题（主治+药师）",
        "examples": "如：孩子发烧 39 度可以同时吃布洛芬和对乙酰氨基酚吗？",
        "expected": '{"physician": 1, "pharmacist": 1, "radiologist": 0}',
    },
]


def build_generation_prompt(scene: dict, batch_size: int = 5) -> tuple[str, str]:
    system = (
        "你是医疗 NLP 数据标注员，负责生成路由分类训练数据。"
        + ROUTING_RULES
        + "\n请严格按 JSON 格式输出，不要解释。"
    )

    user = f"""请按下面的场景生成 {batch_size} 条不同的中文医疗问题，并标注角色标签。

场景：{scene['scene']}
参考样例：{scene['examples']}
期望标签：{scene['expected']}

要求：
1. 问题口语化，模仿患者或家属真实提问
2. 长度 15-80 字
3. 内容多样，覆盖不同症状、人群、场景
4. 确保标签符合场景描述

输出 JSON 格式（数组）：
{{
  "items": [
    {{"query": "问题文本", "labels": {{"physician": 1, "pharmacist": 0, "radiologist": 0}}}},
    ...
  ]
}}
"""
    return system, user


def normalize_labels(labels: dict) -> dict[str, int]:
    """确保标签是 0/1 整数，且 physician 至少为 1。"""
    result = {
        "physician": int(bool(labels.get("physician", 1))),
        "pharmacist": int(bool(labels.get("pharmacist", 0))),
        "radiologist": int(bool(labels.get("radiologist", 0))),
    }
    # 主治始终参与
    result["physician"] = 1
    return result


def main() -> None:
    existing = load_jsonl(OUTPUT)
    print(f"[INFO] 已有数据 {len(existing)} 条，目标 {TARGET_COUNT} 条")

    if len(existing) >= TARGET_COUNT:
        print("[OK] 数据已足够，跳过。")
        return

    seen_queries = {item.get("query", "") for item in existing}

    batch_size = 5
    while len(existing) < TARGET_COUNT:
        # 轮询所有场景，保持数据均衡
        scene = GENERATION_PROMPTS[len(existing) % len(GENERATION_PROMPTS)]
        system, user = build_generation_prompt(scene, batch_size=batch_size)

        try:
            response = call_llm(system, user, temperature=0.9, response_format_json=True)
            payload = parse_json_safely(response)
            if not payload or "items" not in payload:
                print(f"[WARN] 解析失败，跳过批次。响应: {response[:200]}")
                continue

            new_count = 0
            for item in payload["items"]:
                query = (item.get("query") or "").strip()
                labels = item.get("labels", {})
                if not query or query in seen_queries:
                    continue
                if not isinstance(labels, dict):
                    continue

                record = {
                    "query": query,
                    "labels": normalize_labels(labels),
                    "scene": scene["scene"],
                }
                append_jsonl(record, OUTPUT)
                existing.append(record)
                seen_queries.add(query)
                new_count += 1

            print(f"[+{new_count}] 当前总数: {len(existing)}/{TARGET_COUNT} (场景: {scene['scene']})")

        except Exception as exc:
            print(f"[ERR] 批次失败: {exc}")
            continue

    print(f"[DONE] 共生成 {len(existing)} 条到 {OUTPUT}")


if __name__ == "__main__":
    main()
