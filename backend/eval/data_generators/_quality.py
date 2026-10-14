"""数据质量过滤 + 去重工具（基于 LIMA 思路）。

LIMA (Zhou et al., NeurIPS 2023) 核心洞察：
"几乎所有知识都来自预训练，只需有限指令数据" —— 质量 >> 数量

我们对每条训练数据做：
1. 长度过滤（assistant 至少 30 字，最多 3000 字）
2. 内容过滤（必须有解析，不能只是"答案：A"）
3. 去重（SimHash + 用户问题前 200 字）
4. 角色相关性过滤（必须含 surgeon/oncologist 关键词）
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable


# ───────────── 长度 + 内容质量 ─────────────

def passes_length_check(record: dict, min_len: int = 30, max_len: int = 3000) -> bool:
    """assistant 内容长度合规。"""
    msgs = record.get("messages", [])
    if len(msgs) < 3:
        return False
    assistant = msgs[-1].get("content", "") if msgs[-1].get("role") == "assistant" else ""
    return min_len <= len(assistant) <= max_len


def passes_content_check(record: dict) -> bool:
    """assistant 内容是否有实质性解释。"""
    msgs = record.get("messages", [])
    if len(msgs) < 3:
        return False
    assistant = msgs[-1].get("content", "")

    # 拒绝：只有"答案：X"没有任何分析
    stripped = assistant.strip()
    if re.match(r"^答案[:：]\s*[A-Z]\.?\s*$", stripped):
        return False
    if re.match(r"^[A-Z]\.?\s*$", stripped):
        return False
    # 拒绝：长度合规但全是数字/标点（异常数据）
    if len(re.findall(r"[一-鿿]|[a-zA-Z]", stripped)) < 20:
        return False

    return True


# ───────────── 去重（SimHash 简化版）─────────────

def _normalize(text: str) -> str:
    """归一化用于去重比较。"""
    return re.sub(r"\s+", " ", text.strip().lower())[:500]


def _content_hash(record: dict) -> str:
    """用 user 内容前 500 字哈希做去重 key。"""
    msgs = record.get("messages", [])
    user_text = ""
    for m in msgs:
        if m.get("role") == "user":
            user_text = m.get("content", "")
            break
    return hashlib.md5(_normalize(user_text).encode("utf-8")).hexdigest()


def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    """精确去重（基于 user 内容前 500 字）。

    返回：(去重后列表, 删除数量)
    """
    seen_hashes: set[str] = set()
    unique = []
    removed = 0
    for r in records:
        h = _content_hash(r)
        if h in seen_hashes:
            removed += 1
            continue
        seen_hashes.add(h)
        unique.append(r)
    return unique, removed


# ───────────── 角色相关性过滤 ─────────────

def passes_role_relevance(record: dict, role_keywords: Iterable[str]) -> bool:
    """检查是否真的跟该角色相关（避免训练数据偏离主题）。"""
    msgs = record.get("messages", [])
    full_text = " ".join(m.get("content", "") for m in msgs)
    full_lower = full_text.lower()
    return any(kw.lower() in full_lower for kw in role_keywords)


# ───────────── 综合过滤 ─────────────

def filter_records(
    records: list[dict],
    role_keywords: Iterable[str] | None = None,
    *,
    min_len: int = 30,
    max_len: int = 3000,
    verbose: bool = True,
) -> dict:
    """对一批 ChatML 记录做完整质量过滤。

    返回 dict 含：
        kept: 通过的记录列表
        stats: 各阶段过滤统计
    """
    stats = {
        "input": len(records),
        "fail_length": 0,
        "fail_content": 0,
        "fail_relevance": 0,
        "duplicates": 0,
        "kept": 0,
    }

    survivors = []
    for r in records:
        if not passes_length_check(r, min_len, max_len):
            stats["fail_length"] += 1
            continue
        if not passes_content_check(r):
            stats["fail_content"] += 1
            continue
        if role_keywords and not passes_role_relevance(r, role_keywords):
            stats["fail_relevance"] += 1
            continue
        survivors.append(r)

    deduped, removed = deduplicate(survivors)
    stats["duplicates"] = removed
    stats["kept"] = len(deduped)

    if verbose:
        print(f"[quality] 输入: {stats['input']}")
        print(f"[quality]   长度不合规: -{stats['fail_length']}")
        print(f"[quality]   内容质量低: -{stats['fail_content']}")
        print(f"[quality]   角色不相关: -{stats['fail_relevance']}")
        print(f"[quality]   重复: -{stats['duplicates']}")
        print(f"[quality] 保留: {stats['kept']} ({stats['kept']/max(1,stats['input'])*100:.1f}%)")

    return {"kept": deduped, "stats": stats}
