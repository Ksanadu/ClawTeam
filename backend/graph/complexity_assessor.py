"""复杂度评估器（Complexity Assessor）— 不是 Router！

按 TA 反馈：Router 是为了省成本（大小模型池），我们这里不是 Router，
是判断病例复杂度，决定是否需要触发多轮 MDT 辩论。

参考 MDAgents (NeurIPS 2024) 的 Medical Complexity Check 设计：
- simple   → 单 agent 直接答（如基础肿瘤知识题，省 token）
- moderate → 4 角色独立给意见 + 简单聚合
- complex  → 4 角色 + 多轮辩论 + 仲裁（最重，但效果最好）
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from config import get_settings
from graph.llm import build_llm_config_from_settings, get_llm

logger = logging.getLogger(__name__)


class CaseComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class ComplexityDecision:
    level: CaseComplexity
    reason: str
    method: str  # "llm" | "bert" | "keyword"


# ───────────── 规则增强判断（baseline + safety trigger） ─────────────

_COMPLEX_KEYWORDS = [
    "鉴别诊断", "MDT", "多学科", "多轮", "反驳", "修正", "仲裁", "争议", "疑难",
    "晚期", "局部晚期", "转移", "复发", "进展", "不可切除", "边界可切除",
    "新辅助", "辅助治疗", "同步放化疗", "放化疗", "靶向", "免疫",
    "EGFR", "ALK", "ROS1", "BRAF", "KRAS", "HER2", "PD-L1", "MSI", "dMMR", "BRCA",
    "肝肾功能", "合并症", "高龄", "妊娠", "ECOG", "PS", "心衰", "肾衰", "肝硬化",
]

_SIMPLE_KEYWORDS = [
    "什么是", "定义", "请解释", "病理类型",
    "TNM 分期标准", "WHO 分类", "概念", "区别是什么",
]

_EXPLICIT_DEBATE_KEYWORDS = [
    "MDT", "多学科", "多轮", "辩论", "反驳", "修正", "仲裁", "不同意见", "争议",
]

_TREATMENT_MODALITY_KEYWORDS = [
    "手术", "切除", "外科", "放疗", "SBRT", "IMRT", "化疗", "靶向", "免疫",
    "新辅助", "辅助治疗", "同步放化疗",
]

_BIOMARKER_KEYWORDS = [
    "EGFR", "ALK", "ROS1", "BRAF", "KRAS", "HER2", "PD-L1", "MSI", "dMMR", "BRCA",
]

_TUMOR_CONTEXT_KEYWORDS = [
    "癌", "肿瘤", "肉瘤", "淋巴瘤", "白血病",
    "cancer", "tumor", "tumour", "carcinoma", "adenocarcinoma", "lymphoma", "sarcoma",
]

_PATIENT_CONTEXT_KEYWORDS = [
    "患者", "病人", "男性", "女性", "岁", "病理", "影像", "CT", "MRI", "PET",
    "诊断", "分期", "病例", "case", "patient",
]


def _matched_keywords(case: str, keywords: list[str]) -> list[str]:
    case_lower = case.lower()
    hits = []
    for keyword in keywords:
        if keyword.lower() in case_lower:
            hits.append(keyword)
    return hits


def _advanced_stage_hits(case: str) -> list[str]:
    patterns = [
        (r"\bT[34][a-d]?\b", "T3/T4"),
        (r"\bN[23][a-d]?\b", "N2/N3"),
        (r"\bM1[a-d]?\b", "M1"),
        (r"(III|IV)[A-D]?期", "III/IV期"),
        (r"stage\s*(III|IV)[A-D]?", "stage III/IV"),
    ]
    hits = []
    for pattern, label in patterns:
        if re.search(pattern, case, flags=re.IGNORECASE):
            hits.append(label)
    return sorted(set(hits))


def _looks_like_patient_case(case: str) -> bool:
    return bool(_matched_keywords(case, _TUMOR_CONTEXT_KEYWORDS)) and bool(
        _matched_keywords(case, _PATIENT_CONTEXT_KEYWORDS)
    )


def assess_by_keyword(case: str) -> ComplexityDecision:
    """规则增强评估。

    原则：复杂肿瘤治疗宁可触发 MDT 辩论，也不要误判为 moderate 后跳过 Round 2。
    """
    simple_hits = _matched_keywords(case, _SIMPLE_KEYWORDS)
    explicit_debate_hits = _matched_keywords(case, _EXPLICIT_DEBATE_KEYWORDS)
    complex_hits = _matched_keywords(case, _COMPLEX_KEYWORDS)
    modality_hits = _matched_keywords(case, _TREATMENT_MODALITY_KEYWORDS)
    biomarker_hits = _matched_keywords(case, _BIOMARKER_KEYWORDS)
    stage_hits = _advanced_stage_hits(case)
    patient_case = _looks_like_patient_case(case)

    if simple_hits and not patient_case and not complex_hits and not modality_hits:
        return ComplexityDecision(
            level=CaseComplexity.SIMPLE,
            reason=f"规则增强：知识查询关键词 {simple_hits[:3]}",
            method="keyword",
        )

    reason_parts = []
    if explicit_debate_hits:
        reason_parts.append(f"显式要求协作/辩论 {explicit_debate_hits[:3]}")
    if stage_hits:
        reason_parts.append(f"高级别分期 {stage_hits}")
    if len(set(modality_hits)) >= 2:
        reason_parts.append(f"多治疗路径取舍 {sorted(set(modality_hits))[:5]}")
    if biomarker_hits and modality_hits:
        reason_parts.append(f"分子标志物影响治疗 {sorted(set(biomarker_hits))[:5]}")
    if len(set(complex_hits)) >= 2:
        reason_parts.append(f"复杂决策关键词 {sorted(set(complex_hits))[:6]}")
    if patient_case and len(case) >= 100 and any(word in case for word in ["方案", "治疗", "会诊", "下一步", "如何处理"]):
        reason_parts.append("完整肿瘤病例且询问治疗方案")

    if reason_parts:
        return ComplexityDecision(
            level=CaseComplexity.COMPLEX,
            reason="规则增强：" + "；".join(reason_parts),
            method="keyword",
        )

    if patient_case:
        return ComplexityDecision(
            level=CaseComplexity.MODERATE,
            reason="规则增强：常规肿瘤病例，需要多专科独立意见",
            method="keyword",
        )

    return ComplexityDecision(
        level=CaseComplexity.MODERATE,
        reason="规则增强：未触发 simple/complex 强规则，交由 LLM 辅助判断",
        method="keyword",
    )


async def assess_by_llm(case: str) -> ComplexityDecision:
    """规则优先，边界病例再用 LLM 评估复杂度。"""
    rule_decision = assess_by_keyword(case)
    if rule_decision.level in {CaseComplexity.SIMPLE, CaseComplexity.COMPLEX}:
        return ComplexityDecision(
            level=rule_decision.level,
            reason=f"规则优先：{rule_decision.reason}",
            method="rule+llm",
        )

    settings = get_settings()
    llm_config = build_llm_config_from_settings(
        settings, temperature=0.0, streaming=False
    )
    llm = get_llm(llm_config)

    prompt = (
        "你是肿瘤多学科会诊系统的病例复杂度评估员。\n"
        "根据以下病例描述，判断其复杂度等级。请偏向安全：只要治疗路径存在明显取舍，就判 complex。\n"
        "- simple：基础知识问答（如定义、TNM 标准），单个专家可解决\n"
        "- moderate：常规、单一路径的肿瘤病例，4 个专科独立给意见即可\n"
        "- complex：需要多轮 MDT 辩论的病例，例如晚期/复发/转移、N2/N3/M1、"
        "新辅助 vs 手术 vs 放化疗取舍、分子标志物影响治疗、合并症/高龄/肝肾功能限制、"
        "用户明确要求 MDT/反驳/多轮讨论\n"
        "\n"
        "只输出一个词：simple / moderate / complex。"
    )

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": case},
        ])
        content = getattr(response, "content", "").strip().lower()

        match = re.search(r"\b(simple|moderate|complex)\b", content)
        label = match.group(1) if match else content

        if label == "simple":
            level = CaseComplexity.SIMPLE
        elif label == "complex":
            level = CaseComplexity.COMPLEX
        else:
            level = CaseComplexity.MODERATE

        if level == CaseComplexity.SIMPLE and _looks_like_patient_case(case):
            level = CaseComplexity.MODERATE

        return ComplexityDecision(
            level=level,
            reason=f"{rule_decision.reason}；LLM 边界评估：{content}",
            method="rule+llm",
        )
    except Exception as exc:
        logger.warning("LLM complexity assessment failed: %s, falling back to keyword", exc)
        return assess_by_keyword(case)


def assess_by_bert(case: str) -> ComplexityDecision | None:
    """用之前训练好的 BERT Router 模型评估（已训完，作为对比组）。"""
    try:
        from eval.inference.load_router import load_trained_router
        router = load_trained_router()
        if router is None:
            return None

        # 之前的 BERT Router 是多标签分类（physician/pharmacist/radiologist）
        # 我们把"调用了几个角色"映射到复杂度
        probs = router.predict(case)
        active = sum(1 for p in probs.values() if p > 0.5)

        if active <= 1:
            level = CaseComplexity.SIMPLE
        elif active == 2:
            level = CaseComplexity.MODERATE
        else:
            level = CaseComplexity.COMPLEX

        return ComplexityDecision(
            level=level,
            reason=f"BERT 评估：检测到 {active} 个相关专科",
            method="bert",
        )
    except Exception as exc:
        logger.debug("BERT assessor not available: %s", exc)
        return None


async def assess_complexity(
    case: str,
    method: Literal["llm", "bert", "keyword"] = "llm",
) -> ComplexityDecision:
    """统一入口。"""
    if method == "bert":
        result = assess_by_bert(case)
        if result is not None:
            return result
        # 回退到 keyword
        return assess_by_keyword(case)
    if method == "keyword":
        return assess_by_keyword(case)
    return await assess_by_llm(case)
