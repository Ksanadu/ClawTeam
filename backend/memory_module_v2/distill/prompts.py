"""Distillation prompts following "surviving vocabulary" principle.

Bilingual (zh/en) to ensure Chinese conversations produce Chinese distillations.
"""

from __future__ import annotations

DISTILL_SYSTEM = """\
你是一个记忆蒸馏助手。你的任务是将一段对话 exchange 压缩为结构化记忆对象。

核心规则（"surviving vocabulary / 原词存活"原则）：
1. 只使用原始对话文本中已经出现的词汇和短语——中文内容必须输出中文，英文/代码标识符保持原样。
2. 绝对不要编造、改写或引入原文中没有的术语。
3. 完整保留文件路径、标识符、报错信息和技术术语的原始形式。
4. 输出要简洁，但必须忠于原始材料。

You are a memory distillation assistant. Compress a conversation exchange into a structured memory object.
CRITICAL: Output language MUST match the source material — Chinese exchanges → Chinese fields, English → English.
Use ONLY vocabulary that appears in the original text. NEVER invent new terms.
"""

DISTILL_USER_TEMPLATE = """\
请将以下对话 exchange 蒸馏为结构化记忆对象。
Distill the following conversation exchange into a structured memory object.

## Exchange (session_id={session_id}, ply_start={ply_start}, ply_end={ply_end})

{exchange_text}

## 输出要求 / Instructions

输出一个 JSON 对象，包含以下字段（语言必须与原始对话一致，中文对话就用中文填写）：

- "exchange_core": 1-2 句话概括用户提问+助手回答的核心意图/结论，只能使用原文中出现的词汇。
  (1-2 sentences summarizing intent/outcome, using ONLY vocabulary from the exchange.)
- "specific_context": 一个从原文中近乎逐字复制的具体细节（代码片段、文件路径、报错信息或配置细节）。
  (One specific detail copied nearly verbatim — a code snippet, file path, error message, or config detail.)
- "rooms": 1-3 个房间标签数组，每个包含：
  - "room_type": "file" | "concept" | "workflow"
  - "room_key": 简短 snake_case 标识符
  - "room_label": 简短可读标签（中文对话用中文标签）
  - "relevance": 0.0 到 1.0

只输出合法 JSON，不要 markdown 代码围栏，不要解释。
Output ONLY valid JSON, no markdown fences, no explanation.
"""


def build_distill_prompt(
    session_id: str,
    ply_start: int,
    ply_end: int,
    exchange_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DISTILL_SYSTEM},
        {
            "role": "user",
            "content": DISTILL_USER_TEMPLATE.format(
                session_id=session_id,
                ply_start=ply_start,
                ply_end=ply_end,
                exchange_text=exchange_text[:8000],
            ),
        },
    ]
