"""推荐气泡 API（v3.1 重构）

设计目标：基于用户历史 + 共享记忆 + 角色权重生成个性化 Tumor Board 案例推荐，
而不是简单的随机问题。这是 ClawTeam "自我进化" 机制的轻量原型。

数据源（4 层加权融合）：
1. 最近会话主题（最近 N 个 session 讨论的肿瘤类型）
2. memory_v2 检索（如果开启）
3. 角色调用频次（用户最常涉及哪些专科）
4. 冷启动 fallback（30 个经典 Tumor Board 案例）

输出格式：
- text: 推荐问题
- reason: 为什么推荐（"因为你之前讨论了 X"）
- case_data: 完整病例（点击后直接加载）
- score: 相关性分数
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from graph.agent import agent_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# ───────────── 经典案例库（冷启动 + fallback）─────────────

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_CASES_PATH = _BACKEND_DIR / "eval" / "datasets" / "data" / "tumor_board_cases.jsonl"


def _load_classic_cases() -> list[dict]:
    """加载 30 个经典 Tumor Board 案例。"""
    if not _CASES_PATH.exists():
        return []
    cases = []
    with open(_CASES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return cases


# ───────────── 数据源 1: 最近会话主题 ─────────────

# 肿瘤类型识别关键词
TUMOR_TYPE_KEYWORDS = {
    "lung": ["肺癌", "肺腺癌", "肺鳞癌", "小细胞肺癌", "肺结节", "EGFR", "ALK"],
    "gastric": ["胃癌", "胃腺癌", "胃肿瘤"],
    "liver": ["肝癌", "肝细胞癌", "HCC", "肝脏肿瘤"],
    "breast": ["乳腺癌", "乳腺", "HER2", "ER", "BRCA"],
    "colorectal": ["结肠癌", "直肠癌", "大肠癌", "结直肠"],
    "esophageal": ["食管癌", "食管"],
    "pancreatic": ["胰腺癌", "胰头癌"],
    "thyroid": ["甲状腺癌", "甲状腺"],
    "prostate": ["前列腺癌", "前列腺"],
    "ovarian": ["卵巢癌", "卵巢"],
    "cervical": ["宫颈癌", "宫颈"],
    "lymphoma": ["淋巴瘤", "DLBCL"],
    "head_neck": ["头颈癌", "口咽癌", "喉癌"],
    "renal": ["肾癌", "肾透明细胞癌"],
    "sarcoma": ["肉瘤", "GIST", "间质瘤"],
}


def detect_tumor_type(text: str) -> list[str]:
    """从文本检测肿瘤类型。"""
    detected = []
    for tumor_type, keywords in TUMOR_TYPE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            detected.append(tumor_type)
    return detected


def get_recent_tumor_topics(limit: int = 5) -> Counter:
    """从最近会话提取讨论过的肿瘤类型，按时间倒序加权。

    最近 1 次权重 1.0，第 2 次 0.7，第 3 次 0.5...
    """
    weights = Counter()
    if agent_manager.session_manager is None:
        return weights

    try:
        sessions = agent_manager.session_manager.list_sessions()[:limit]
    except Exception as exc:
        logger.warning("Failed to list sessions: %s", exc)
        return weights

    for idx, session_summary in enumerate(sessions):
        weight = max(0.3, 1.0 - idx * 0.2)  # 最近一次 1.0，最远 0.3
        session_id = session_summary.get("id")
        if not session_id:
            continue
        try:
            record = agent_manager.session_manager.load_session_record(session_id)
        except Exception:
            continue
        all_text = " ".join(
            str(msg.get("content", "")) for msg in record.get("messages", [])
        )
        for tumor_type in detect_tumor_type(all_text):
            weights[tumor_type] += weight

    return weights


# ───────────── 数据源 2: memory_v2 检索 ─────────────

def get_memory_v2_recommendations(query_seed: str, top_k: int = 3) -> list[str]:
    """如果 memory_v2 启用，从历史病例库检索相似 case。"""
    try:
        from memory_module_v2.service.config import get_memory_backend
        if get_memory_backend() != "v2":
            return []

        from memory_module_v2.service.api import search_memory
        from memory_module_v2.domain.enums import SearchMode

        results = search_memory(
            query=query_seed, mode=SearchMode.HYBRID_CROSS, top_k=top_k
        )
        return [hit.verbatim_snippet for hit in results.hits if hit.verbatim_snippet]
    except Exception as exc:
        logger.debug("memory_v2 not available: %s", exc)
        return []


# ───────────── 数据源 3: 角色调用频次 ─────────────

# 角色 → 关注哪些肿瘤类型（用于"如果你常用 X 角色，推荐 Y 类肿瘤"）
ROLE_TO_TUMOR_AFFINITY = {
    "surgeon": ["lung", "gastric", "colorectal", "liver", "breast"],
    "medical_oncologist": ["lung", "breast", "lymphoma", "ovarian"],
    "pathologist": ["lung", "breast", "thyroid"],
    "radiation_oncologist": ["lung", "head_neck", "cervical", "prostate"],
}


def get_role_call_frequency() -> Counter:
    """从实验日志统计哪些角色被调用得最多。"""
    weights = Counter()
    log_dir = _BACKEND_DIR / "storage" / "experiment_logs"
    if not log_dir.exists():
        return weights

    for log_file in log_dir.glob("experiment_*.jsonl"):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    for role in entry.get("roles_called", []):
                        weights[role] += 1
        except Exception:
            continue
    return weights


# ───────────── 融合排序 + 推荐生成 ─────────────

def score_case(case: dict, recent_topics: Counter, role_freq: Counter) -> tuple[float, str]:
    """给一个案例打分 + 生成推荐理由。"""
    score = 0.0
    reasons = []

    tumor_type = case.get("tumor_type", "")

    # 数据源 1: 最近主题加权
    if tumor_type in recent_topics:
        topic_weight = recent_topics[tumor_type]
        score += 1.0 * topic_weight
        # 格式化中文肿瘤名称
        zh_name = {
            "lung": "肺癌", "gastric": "胃癌", "liver": "肝癌",
            "breast": "乳腺癌", "colorectal": "结直肠癌",
            "esophageal": "食管癌", "pancreatic": "胰腺癌",
            "thyroid": "甲状腺癌", "prostate": "前列腺癌",
            "ovarian": "卵巢癌", "cervical": "宫颈癌",
            "lymphoma": "淋巴瘤", "head_neck": "头颈癌",
            "renal": "肾癌", "sarcoma": "肉瘤",
        }.get(tumor_type, tumor_type)
        reasons.append(f"你最近讨论过{zh_name}")

    # 数据源 3: 角色频次匹配
    for role, affinities in ROLE_TO_TUMOR_AFFINITY.items():
        if role_freq.get(role, 0) > 0 and tumor_type in affinities:
            role_zh = {
                "surgeon": "外科",
                "medical_oncologist": "肿瘤内科",
                "pathologist": "病理科",
                "radiation_oncologist": "放疗科",
            }.get(role, role)
            score += 0.3 * (role_freq[role] / max(sum(role_freq.values()), 1))
            reasons.append(f"你常使用{role_zh}视角")
            break

    # Complexity 加分（complex case 更值得推荐）
    if case.get("complexity") == "complex":
        score += 0.2

    reason_text = " · ".join(reasons) if reasons else "经典 Tumor Board 案例"
    return score, reason_text


def generate_recommendations(top_n: int = 5) -> list[dict[str, Any]]:
    """主入口：返回 Top-N 推荐案例。"""
    classic_cases = _load_classic_cases()
    if not classic_cases:
        logger.warning("Tumor Board case library empty, returning fallback")
        return _fallback_recommendations()

    recent_topics = get_recent_tumor_topics(limit=5)
    role_freq = get_role_call_frequency()

    # 给每个案例打分
    scored = []
    for case in classic_cases:
        score, reason = score_case(case, recent_topics, role_freq)
        scored.append({
            "id": case["id"],
            "text": case["title"],
            "reason": reason,
            "case_data": case,
            "score": score,
        })

    # 排序：高分优先；分数相同时随机打乱，避免每次返回相同
    import random
    random.shuffle(scored)
    scored.sort(key=lambda x: x["score"], reverse=True)

    # 冷启动：如果所有分数都是 0，多样性优先（不同肿瘤类型各取一个）
    if all(s["score"] == 0 for s in scored):
        return _diverse_fallback(classic_cases, top_n)

    return scored[:top_n]


def _diverse_fallback(cases: list[dict], top_n: int) -> list[dict[str, Any]]:
    """冷启动：从不同肿瘤类型各取一个，保证多样性。"""
    seen_types = set()
    result = []
    for case in cases:
        tumor_type = case.get("tumor_type", "")
        if tumor_type not in seen_types:
            seen_types.add(tumor_type)
            result.append({
                "id": case["id"],
                "text": case["title"],
                "reason": "经典 Tumor Board 案例（推荐入门）",
                "case_data": case,
                "score": 0.5,
            })
            if len(result) >= top_n:
                break
    return result


def _fallback_recommendations() -> list[dict[str, Any]]:
    """完全 fallback（案例库都加载失败时）。"""
    return [
        {
            "id": "fallback_1",
            "text": "肺腺癌 T2N1M0 EGFR 阳性 治疗方案",
            "reason": "默认推荐",
            "case_data": None,
            "score": 0.0,
        },
        {
            "id": "fallback_2",
            "text": "局部进展期胃癌新辅助治疗",
            "reason": "默认推荐",
            "case_data": None,
            "score": 0.0,
        },
    ]


# ───────────── API 路由 ─────────────

@router.get("/recommend")
async def recommend() -> dict[str, Any]:
    """获取首页推荐案例气泡。"""
    recommendations = generate_recommendations(top_n=5)
    return {"recommendations": recommendations}


@router.get("/recommend/case/{case_id}")
async def get_case_detail(case_id: str) -> dict[str, Any]:
    """获取单个案例的完整内容（用户点击气泡时调用）。"""
    cases = _load_classic_cases()
    for case in cases:
        if case.get("id") == case_id:
            return {"case": case}
    return {"case": None, "error": "Case not found"}
