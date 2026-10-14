"""生成 Round 2 多视角辩论格式训练数据。

⭐ 这是修复"模型不会按 Round 2 格式输出"的关键脚本。

输出格式：
{
  "messages": [
    {"role": "system", "content": "[role prompt]"},
    {"role": "user", "content": "[case] + [3 其他专家 R1 意见] + [自己 R1] + [R2 任务]"},
    {"role": "assistant", "content": "## 同意 ... ## 反对 ... ## 修正 ... ## R2 最终意见 ..."}
  ],
  "source": "round2",
  "view": "round2_debate"
}

每个角色生成 800 条，确保 LoRA 学到 Round 2 格式。
"""

from __future__ import annotations

import argparse
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

TRAINING_DIR = DATA_DIR / "training"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

ROLES = {
    "surgeon": {
        "label": "肿瘤外科医生",
        "prompt_file": "SURGEON.md",
        "output": TRAINING_DIR / "surgeon_train.jsonl",
        "structure_keywords": "可切除性 / 术式 / 淋巴清扫 / 围手术期",
        "common_disagreements": "外科想切 vs 内科想保守 / 手术 vs SBRT 替代 / 直接手术 vs 新辅助",
    },
    "medical_oncologist": {
        "label": "肿瘤内科医生",
        "prompt_file": "MEDICAL_ONCOLOGIST.md",
        "output": TRAINING_DIR / "oncologist_train.jsonl",
        "structure_keywords": "治疗目标 / 推荐方案 / 剂量周期 / 不良反应 / 与其他治疗协调",
        "common_disagreements": "保守化疗 vs 直接手术 / 化疗剂量 vs 同步放疗剂量 / 标准方案 vs 临床试验",
    },
}

# 典型肿瘤场景 seed（用于多样化生成）
TUMOR_SCENARIOS = [
    "肺腺癌 EGFR+ T2N1M0", "肺鳞癌 PD-L1 高表达 T3N2M0",
    "胃癌 HER2+ T3N2M0 幽门梗阻", "胃癌 MSI-H T2N0M0 早期",
    "结肠癌 KRAS突变 肝转移", "直肠癌中低位 T3N1M0 MRF阳性",
    "肝细胞癌 BCLC-B 期 门静脉癌栓", "胰腺癌 borderline resectable",
    "乳腺癌 HER2+ T2N1M0 ER+", "三阴乳腺癌 BRCA1 阳性 T2N0",
    "卵巢浆液性癌 IIIC 期 BRCA2", "前列腺癌 Gleason 4+3 中危",
    "甲状腺乳头状癌 BRAF+ T2N1a", "食管鳞癌 T2N1M0 中段",
    "口咽鳞癌 HPV阴性 T3N2b", "肾透明细胞癌 T2N0M1",
    "膀胱癌 肌层浸润 T2N0M0", "胆囊癌 T2 偶发",
    "GIST KIT 11外显子 5cm",
]


def load_role_prompt(role_key: str) -> str:
    role = ROLES[role_key]
    p = BACKEND_DIR / "workspace" / "roles" / role["prompt_file"]
    if p.exists():
        return p.read_text(encoding="utf-8")
    return f"你是一名{role['label']}。"


def build_round2_prompt(role_key: str, scenario: str) -> tuple[str, str]:
    """让 DeepSeek 一次性生成完整 Round 2 训练样本（含病例 + 4 专家意见 + Round 2 回答）。"""
    role = ROLES[role_key]

    system = (
        f"你是医学训练数据生成员。需要为 Tumor Board 多学科会诊系统生成 Round 2 多视角辩论训练数据。\n"
        f"你将生成完整的会诊场景：病例 + 4 个专家（病理/外科/内科/放疗）的 Round 1 意见 + "
        f"{role['label']}的 Round 2 反驳/修正回答。\n"
        f"严格基于真实临床实践，不编造错误医学事实。输出 JSON。"
    )

    user = f"""请基于场景【{scenario}】生成 1 条完整的 Round 2 训练样本。

## 输出 JSON 格式

{{
  "case": "[详细病例：50-150 字，包含年龄、性别、分期、关键指标]",
  "round1_opinions": {{
    "病理科": "[Round 1 意见：分期分级 + 分子标志物 + 病理建议，60-120 字]",
    "肿瘤外科": "[Round 1 意见：可切除性 + 术式建议，60-120 字]",
    "肿瘤内科": "[Round 1 意见：化疗/靶向方案，60-120 字]",
    "放疗科": "[Round 1 意见：放疗剂量与时机，60-120 字]"
  }},
  "round2_response": "[严格按以下结构，350-500 字]"
}}

## round2_response 必须严格遵循的结构

```
## 同意（Agreements）
[列出 1-2 条同意的他人观点 + 简要理由]

## 反对（Disagreements）
[列出 1-2 条不同意的他人观点 + 你的反对依据]
[如果完全同意没有反对，必须明确写"无明显分歧"]

## 修正（Revisions）
[基于他人新信息，是否修正你 Round 1 的判断？]
[如果无需修正，写"坚持 Round 1 判断"]

## Round 2 最终意见
[结合上述思考，按"{role['structure_keywords']}"结构输出，120-200 字]
```

## 关键要求

1. **{role['label']}视角**：所有反驳/修正必须站在{role['label']}角度
2. **真实辩论**：必须有至少 1 个反对/修正点（典型冲突：{role['common_disagreements']}）
3. **医学事实准确**：所有分期、剂量、术式、药物必须真实
4. **结构严格**：四段标题必须用 `## 同意`、`## 反对`、`## 修正`、`## Round 2 最终意见`

只输出上述 JSON，不要解释。
"""
    return system, user


def generate_round2_for_role(role_key: str, target: int = 800) -> int:
    """为指定角色生成 Round 2 训练样本。"""
    role = ROLES[role_key]
    role_prompt = load_role_prompt(role_key)
    output_path = role["output"]

    # 检查已有的 round2 数据
    existing_round2 = 0
    if output_path.exists():
        for r in load_jsonl(output_path):
            if r.get("source") == "round2":
                existing_round2 += 1
        print(f"[{role_key}-r2] 已有 Round 2 数据: {existing_round2} 条")

    if existing_round2 >= target:
        print(f"[{role_key}-r2] 目标已达成，跳过")
        return existing_round2

    seen = set()
    if output_path.exists():
        for r in load_jsonl(output_path):
            msgs = r.get("messages", [])
            if len(msgs) >= 2:
                seen.add(msgs[1].get("content", "")[:200])

    print(f"[{role_key}-r2] 目标 {target}，开始生成...")
    count = existing_round2

    while count < target:
        scenario = random.choice(TUMOR_SCENARIOS)
        system, user = build_round2_prompt(role_key, scenario)

        try:
            response = call_llm(
                system, user, temperature=0.85, response_format_json=True
            )
            payload = parse_json_safely(response)
            if not payload:
                continue

            case = payload.get("case", "").strip()
            r1_ops = payload.get("round1_opinions", {})
            r2_response = payload.get("round2_response", "").strip()

            if not case or not r1_ops or not r2_response:
                continue

            # 拼接 user_text（4 个专家 Round 1 意见 + Round 2 任务）
            others_text = "\n\n".join(
                f"【{specialty}】{opinion}"
                for specialty, opinion in r1_ops.items()
                if specialty != role["label"].replace("肿瘤", "").replace("医生", "")
            )

            own_r1 = r1_ops.get(role["label"].replace("医生", ""), "")
            if not own_r1:
                # 兜底：用关键专家名
                fallback_keys = ["肿瘤外科", "外科"] if role_key == "surgeon" else ["肿瘤内科", "内科"]
                for k in fallback_keys:
                    if k in r1_ops:
                        own_r1 = r1_ops[k]
                        break

            user_text = (
                f"【病例】\n{case}\n\n"
                f"【你的 Round 1 意见】\n{own_r1}\n\n"
                f"【其他专家 Round 1 意见】\n{others_text}\n\n"
                f"【Round 2 任务】\n请按 '## 同意 / ## 反对 / ## 修正 / ## Round 2 最终意见' 四段格式回答。"
            )

            if user_text[:200] in seen:
                continue

            record = {
                "messages": [
                    {"role": "system", "content": role_prompt},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": r2_response},
                ],
                "source": "round2",
                "view": "round2_debate",
                "scenario": scenario,
            }
            append_jsonl(record, output_path)
            seen.add(user_text[:200])
            count += 1

            if count % 20 == 0:
                print(f"[{role_key}-r2] 进度 {count}/{target}")

        except Exception as exc:
            print(f"[{role_key}-r2] 失败: {exc}")
            continue

    print(f"[{role_key}-r2] ✅ 完成，共 {count} 条")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default="all", choices=["surgeon", "medical_oncologist", "all"])
    parser.add_argument("--target", type=int, default=800)
    args = parser.parse_args()

    random.seed(42)

    if args.role in ("surgeon", "all"):
        generate_round2_for_role("surgeon", args.target)
    if args.role in ("medical_oncologist", "all"):
        generate_round2_for_role("medical_oncologist", args.target)


if __name__ == "__main__":
    main()
