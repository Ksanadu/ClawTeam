"""协调器（Coordinator）— v3.1 真协作 Harness 版

核心机制（参考 MDAgents NeurIPS 2024 + MDTeamGPT 2025）：

1. 复杂度评估 → 简单题单 agent / 中等题独立聚合 / 复杂题多轮辩论
2. Round 1：4 个角色并行独立思考（看不到他人）
3. Round 2：每个角色看到他人意见后，强制给出"同意/反对/修正"
4. Round 3：协调器加权聚合 + 主治仲裁

不是顺序流水线，而是真正的多视角碰撞。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import get_settings
from graph.complexity_assessor import (
    CaseComplexity,
    ComplexityDecision,
    assess_complexity,
)
from graph.llm import build_llm_config_from_settings, get_llm
from graph.roles.base_role import RoleAgent, RoleOpinion, RoleType
from graph.roles.medical_oncologist import MedicalOncologistAgent
from graph.roles.pathologist import PathologistAgent
from graph.roles.radiation_oncologist import RadiationOncologistAgent
from graph.roles.surgeon import SurgeonAgent

logger = logging.getLogger(__name__)


# ───────────── 角色权重（按问题相关性加权投票）─────────────

ROLE_RELEVANCE_KEYWORDS = {
    RoleType.PATHOLOGIST: ["分期", "TNM", "病理", "分化", "标志物", "EGFR", "ALK", "PD-L1"],
    RoleType.SURGEON: ["手术", "切除", "可切性", "术式", "淋巴清扫", "围手术期"],
    RoleType.MEDICAL_ONCOLOGIST: ["化疗", "靶向", "免疫治疗", "新辅助", "辅助", "药物", "用药"],
    RoleType.RADIATION_ONCOLOGIST: ["放疗", "Gy", "IMRT", "SBRT", "剂量", "靶区", "OAR"],
}


DECISION_DOMAIN_KEYWORDS = {
    "diagnosis": [
        "病理", "组织学", "分型", "分化", "免疫组化", "IHC", "TNM", "分期",
        "EGFR", "ALK", "ROS1", "PD-L1", "HER2", "KRAS", "BRCA", "标志物",
        "分子", "突变", "pCR", "MPR",
    ],
    "surgery": [
        "手术", "切除", "可切除", "可切性", "术式", "淋巴结清扫", "淋巴清扫",
        "围手术期", "R0", "肺叶切除", "袖式", "边缘可切除",
    ],
    "systemic": [
        "靶向", "化疗", "免疫治疗", "PD-1", "检查点", "TKI", "奥希替尼",
        "新辅助", "辅助", "药物", "用药", "周期", "维持", "耐药", "复发风险",
    ],
    "radiation": [
        "放疗", "PORT", "SBRT", "IMRT", "VMAT", "Gy", "剂量", "分割",
        "靶区", "OAR", "同步放化疗", "序贯",
    ],
    "treatment_path": [
        "治疗路径", "治疗方案", "时间线", "MDT", "会诊", "先手术",
        "先治疗", "再手术", "术前", "术后", "手术还是", "直接手术",
        "新辅助", "降期", "根治性", "综合治疗",
    ],
}


DOMAIN_ROLE_BONUS = {
    "diagnosis": {
        RoleType.PATHOLOGIST: 0.45,
        RoleType.MEDICAL_ONCOLOGIST: 0.15,
    },
    "surgery": {
        RoleType.SURGEON: 0.45,
        RoleType.MEDICAL_ONCOLOGIST: 0.15,
        RoleType.RADIATION_ONCOLOGIST: 0.10,
    },
    "systemic": {
        RoleType.MEDICAL_ONCOLOGIST: 0.45,
        RoleType.SURGEON: 0.15,
        RoleType.PATHOLOGIST: 0.10,
    },
    "radiation": {
        RoleType.RADIATION_ONCOLOGIST: 0.45,
        RoleType.MEDICAL_ONCOLOGIST: 0.15,
        RoleType.SURGEON: 0.10,
    },
    "treatment_path": {
        RoleType.SURGEON: 0.30,
        RoleType.MEDICAL_ONCOLOGIST: 0.30,
        RoleType.RADIATION_ONCOLOGIST: 0.15,
        RoleType.PATHOLOGIST: 0.10,
    },
}


def _count_domain_hits(case_text: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if keyword.lower() in case_text)


def _active_decision_domains(case_text: str) -> set[str]:
    return {
        domain
        for domain, keywords in DECISION_DOMAIN_KEYWORDS.items()
        if _count_domain_hits(case_text, keywords) > 0
    }


def compute_role_weights(case: str) -> dict[RoleType, float]:
    """根据病例中的决策域动态分配 4 个角色权重（用于共识聚合）。"""
    case_text = case.lower()
    weights = {role: 0.6 for role in RoleType}

    for role, keywords in ROLE_RELEVANCE_KEYWORDS.items():
        hits = _count_domain_hits(case_text, keywords)
        weights[role] += min(0.20, hits * 0.05)

    domains = _active_decision_domains(case_text)
    for domain in domains:
        for role, bonus in DOMAIN_ROLE_BONUS[domain].items():
            weights[role] += bonus

    # 综合治疗路径问题应由治疗主责科室牵头；病理提供诊断/分型约束。
    if "treatment_path" in domains:
        weights[RoleType.SURGEON] = max(weights[RoleType.SURGEON], 1.05)
        weights[RoleType.MEDICAL_ONCOLOGIST] = max(
            weights[RoleType.MEDICAL_ONCOLOGIST],
            1.05,
        )
        if "diagnosis" in domains:
            weights[RoleType.PATHOLOGIST] = min(weights[RoleType.PATHOLOGIST], 1.00)

    return {
        role: round(max(0.5, min(1.5, weight)), 2)
        for role, weight in weights.items()
    }


# ───────────── 数据类 ─────────────

@dataclass
class MDTSession:
    """一次 MDT 会诊的完整记录（用于评测和论文）。"""

    case: str
    complexity: ComplexityDecision | None = None
    round1_opinions: list[RoleOpinion] = field(default_factory=list)
    round2_opinions: list[RoleOpinion] = field(default_factory=list)
    final_decision: str = ""
    role_weights: dict[str, float] = field(default_factory=dict)
    revision_rate: float = 0.0  # Round 2 修正率（论文核心指标）
    disagreement_count: int = 0
    latency_ms: int = 0


# ───────────── Coordinator ─────────────

class Coordinator:
    """肿瘤 MDT 协调器 — 真协作 Harness 实现。

    根据复杂度选择不同协作模式：
    - SIMPLE   → 单 agent (默认 internist 角色，或 LLM 直接回答)
    - MODERATE → 4 角色 Round 1；完整模式继续 Round 2，消融时可跳过
    - COMPLEX  → 4 角色 Round 1 + Round 2 辩论 + Round 3 共识仲裁
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self._roles: dict[RoleType, RoleAgent] = {
            RoleType.PATHOLOGIST: PathologistAgent(base_dir),
            RoleType.SURGEON: SurgeonAgent(base_dir),
            RoleType.MEDICAL_ONCOLOGIST: MedicalOncologistAgent(base_dir),
            RoleType.RADIATION_ONCOLOGIST: RadiationOncologistAgent(base_dir),
        }

    async def consult(
        self,
        case: str,
        *,
        memory_context: str = "",
        complexity_method: str = "llm",
        force_complexity: CaseComplexity | None = None,
        skip_round2: bool = False,  # 消融实验：关闭 Round 2 辩论
        attachments: list[dict[str, Any]] | None = None,
    ) -> MDTSession:
        """执行一次 MDT 会诊。

        Args:
            case: 病例描述
            memory_context: 历史病例参考（选做）
            complexity_method: "llm" / "bert" / "keyword"
            force_complexity: 实验时强制指定复杂度
            skip_round2: 消融实验时跳过辩论
        """
        start = time.monotonic()
        session = MDTSession(case=case)

        # ── Step 1: 复杂度评估 ──────────────────────
        if force_complexity is not None:
            session.complexity = ComplexityDecision(
                level=force_complexity,
                reason="手动指定（实验）",
                method="manual",
            )
        else:
            session.complexity = await assess_complexity(case, method=complexity_method)
        logger.info(
            "Complexity: %s (%s)",
            session.complexity.level.value,
            session.complexity.reason,
        )

        # ── Step 2: 根据复杂度选择策略 ──────────────
        if session.complexity.level == CaseComplexity.SIMPLE:
            # 单 agent 路径：用最相关的角色直接答
            session.final_decision = await self._simple_path(case, memory_context, attachments)
        else:
            # MDT 路径：4 角色协作
            session.role_weights = {
                k.value: v for k, v in compute_role_weights(case).items()
            }

            # ── Round 1: 并行独立思考 ───────────
            round1 = await self._run_round1(case, memory_context, attachments)
            session.round1_opinions = round1

            if skip_round2:
                # 消融实验 E2：直接聚合 Round 1，不跑辩论
                session.final_decision = await self._aggregate(
                    case, round1, weights=compute_role_weights(case)
                )
            else:
                # 完整 MDT：所有非 simple 病例进入 Round 2 辩论
                round2 = await self._run_round2(case, round1, memory_context, attachments)
                session.round2_opinions = round2

                # 计算论文核心指标：修正率
                session.revision_rate = self._compute_revision_rate(round1, round2)
                session.disagreement_count = sum(len(op.disagreements) for op in round2)

                # ── Round 3: 协调器仲裁 ──────────
                session.final_decision = await self._arbitrate(
                    case, round2, weights=compute_role_weights(case)
                )

        session.latency_ms = int((time.monotonic() - start) * 1000)
        return session

    # ───────────── 内部方法 ─────────────

    async def _simple_path(
        self,
        case: str,
        memory_context: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> str:
        """简单题：用最相关的角色（默认肿瘤内科）直接答。"""
        agent = self._roles[RoleType.MEDICAL_ONCOLOGIST]
        opinion = await agent.aconsult_round1(case, memory_context, attachments=attachments)
        return opinion.content

    async def _run_round1(
        self,
        case: str,
        memory_context: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> list[RoleOpinion]:
        """Round 1：4 角色并行独立思考。"""
        tasks = [
            agent.aconsult_round1(case, memory_context, attachments=attachments)
            for agent in self._roles.values()
        ]
        opinions = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [op for op in opinions if isinstance(op, RoleOpinion)]
        for op in opinions:
            if isinstance(op, Exception):
                logger.error("Round 1 role failed: %s", op)
        return valid

    async def _run_round2(
        self,
        case: str,
        round1: list[RoleOpinion],
        memory_context: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> list[RoleOpinion]:
        """Round 2：每个角色看完他人意见后，强制给出反驳/修正。

        ⭐ 这是真协作 Harness 的核心。
        """
        tasks = []
        for own in round1:
            others = [op for op in round1 if op.role != own.role]
            agent = self._roles.get(own.role)
            if agent is None:
                continue
            tasks.append(
                agent.aconsult_round2(
                    case, own, others, memory_context, attachments=attachments
                )
            )
        opinions = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [op for op in opinions if isinstance(op, RoleOpinion)]
        for op in opinions:
            if isinstance(op, Exception):
                logger.error("Round 2 role failed: %s", op)
        return valid

    @staticmethod
    def _compute_revision_rate(
        round1: list[RoleOpinion], round2: list[RoleOpinion]
    ) -> float:
        """计算 Round 2 修正率（论文核心指标）。

        定义：Round 2 中明确给出"修正"内容的角色占比。
        """
        if not round2:
            return 0.0
        revised = sum(1 for op in round2 if op.revisions and op.revisions[0] != "坚持 Round 1 判断")
        return revised / len(round2)

    async def _aggregate(
        self,
        case: str,
        opinions: list[RoleOpinion],
        weights: dict[RoleType, float],
    ) -> str:
        """中等复杂度的简单聚合（无辩论场景）。"""
        return await self._build_synthesis(
            case, opinions, weights, mode="aggregate"
        )

    async def _arbitrate(
        self,
        case: str,
        opinions: list[RoleOpinion],
        weights: dict[RoleType, float],
    ) -> str:
        """复杂度 COMPLEX 的最终仲裁（主治视角）。"""
        return await self._build_synthesis(
            case, opinions, weights, mode="arbitrate"
        )

    async def _build_synthesis(
        self,
        case: str,
        opinions: list[RoleOpinion],
        weights: dict[RoleType, float],
        mode: str,
    ) -> str:
        if not opinions:
            return "暂无可用专家意见。"

        # 拼接所有意见 + 权重
        opinion_text = "\n\n".join(
            f"【{op.role_label}（权重 {weights.get(op.role, 1.0):.1f}）】\n{op.content}"
            for op in opinions
        )

        if mode == "arbitrate":
            instruction = (
                "你是 Tumor Board 主任，请综合以下 4 位专家 Round 2 的意见做最终仲裁。\n"
                "仲裁边界：\n"
                "- 专家权重只是辅助依据，不得替代临床证据、患者风险和安全性评估。\n"
                "- 不要声称“所有专家一致”，除非多位专家意见中明确一致。\n"
                "- 不要新增专家意见中没有出现的关键研究、指南名称或证据等级。\n"
                "- 避免长篇综述，最终输出通常控制在 700-1000 字。\n"
                "请按以下结构输出：\n"
                "1. 核心结论：用 2-4 句给出推荐路径。\n"
                "2. 主要分歧与仲裁：列出关键冲突，说明如何结合权重、证据和风险裁决。\n"
                "3. 治疗时间线：按阶段列出新辅助/手术/辅助/放疗安排。\n"
                "4. 条件分支：说明何时改为同步放化疗、何时考虑 PORT 等。\n"
                "5. 风险提醒：列出最关键监测点，并注明建议仅供 MDT 讨论参考。"
            )
        else:
            instruction = (
                "你是 Tumor Board 协调员，请综合以下 4 位专家 Round 1 的独立意见：\n"
                "1. 整合各专科观点，不要新增专家未提出的关键证据\n"
                "2. 专家权重仅作为辅助，不要机械地谁高谁赢\n"
                "3. 输出简洁综合治疗方案，通常控制在 500-800 字\n"
                "4. 末尾加免责声明"
            )

        settings = get_settings()
        llm = get_llm(
            build_llm_config_from_settings(settings, temperature=0.2, streaming=False)
        )

        try:
            response = await llm.ainvoke([
                {"role": "system", "content": instruction},
                {"role": "user", "content": f"【病例】\n{case}\n\n{opinion_text}"},
            ])
            content = getattr(response, "content", "")
            if isinstance(content, str):
                return content.strip()
            return str(content or "").strip()
        except Exception as exc:
            logger.error("Synthesis failed: %s", exc)
            # 降级：直接拼接
            parts = [f"**{op.role_label}**：{op.content[:200]}..." for op in opinions]
            return (
                "\n\n---\n\n".join(parts)
                + "\n\n> 以上仅供参考，具体诊疗请以实际就医为准。"
            )
