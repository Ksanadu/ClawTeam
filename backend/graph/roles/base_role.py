"""角色智能体基类 —— 每个角色复用同一 LLM，通过不同 system prompt 实现专业化。

v3.1 升级：
- 支持多轮辩论（Round 1 独立 / Round 2 反驳 / Round 3 共识）
- 支持基于他人意见的反驳与修正
- 支持 LoRA 切换（外科 / 内科可加载训练好的 LoRA）
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from config import get_settings
from graph.llm import ResolvedLLMConfig, build_llm_config_from_settings, get_llm

logger = logging.getLogger(__name__)


class RoleType(str, Enum):
    """肿瘤多学科会诊（Tumor Board）的 4 个专科。"""

    PATHOLOGIST = "pathologist"
    SURGEON = "surgeon"
    MEDICAL_ONCOLOGIST = "medical_oncologist"
    RADIATION_ONCOLOGIST = "radiation_oncologist"


@dataclass
class RoleOpinion:
    """单个角色的诊断意见。

    在 Round 1 中：仅含 content（独立思考）
    在 Round 2 中：含 agreements / disagreements / revisions（反驳与修正）
    """

    role: RoleType
    role_label: str
    content: str
    round_num: int = 1  # 当前轮次
    agreements: list[str] = field(default_factory=list)  # Round 2: 同意的观点
    disagreements: list[str] = field(default_factory=list)  # Round 2: 反对的观点
    revisions: list[str] = field(default_factory=list)  # Round 2: 修正的判断
    evidence: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class RoleAgent:
    """肿瘤 MDT 角色基类。

    每个角色：
    - Round 1：独立思考给意见
    - Round 2：看到他人意见后反驳/修正
    - 共享同一 LLM 实例，通过不同 prompt 区分
    - 可选挂载 LoRA adapter（仅外科 / 内科）
    """

    role_type: RoleType = RoleType.PATHOLOGIST
    role_label: str = "病理科医生"
    prompt_file: str = "PATHOLOGIST.md"

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self._system_prompt: str | None = None

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = self._load_prompt()
        return self._system_prompt

    def _load_prompt(self) -> str:
        prompt_path = self.base_dir / "workspace" / "roles" / self.prompt_file
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        logger.warning("Role prompt not found: %s", prompt_path)
        return f"你是一名{self.role_label}，请从专业角度回答肿瘤会诊问题。"

    def _build_llm(self, temperature: float = 0.3):
        llm, _source = self._build_llm_with_source(temperature)
        return llm

    def _build_llm_with_source(self, temperature: float = 0.3) -> tuple[Any, str]:
        # 优先尝试加载 LoRA（如果配置了）
        try:
            from eval.inference.load_lora_role import load_local_base_role, load_lora_role
            lora_agent = load_lora_role(self.role_type.value)
            if lora_agent is not None:
                return lora_agent, "lora"
            local_base_agent = load_local_base_role(self.role_type.value)
            if local_base_agent is not None:
                return local_base_agent, "local_base"
        except ImportError:
            pass

        settings = get_settings()
        role_upper = self.role_type.value.upper()
        role_override_keys = [
            f"{role_upper}_LLM_PROVIDER",
            f"{role_upper}_LLM_MODEL",
            f"{role_upper}_LLM_API_KEY",
            f"{role_upper}_LLM_BASE_URL",
        ]
        has_role_override = any(os.getenv(key) for key in role_override_keys)

        if has_role_override:
            llm_config = ResolvedLLMConfig(
                provider=os.getenv(f"{role_upper}_LLM_PROVIDER", settings.llm_provider),
                model=os.getenv(f"{role_upper}_LLM_MODEL", settings.llm_model),
                api_key=os.getenv(f"{role_upper}_LLM_API_KEY", settings.llm_api_key),
                base_url=os.getenv(f"{role_upper}_LLM_BASE_URL", settings.llm_base_url),
                temperature=temperature,
                streaming=False,
            )
            source_prefix = f"api:role:{self.role_type.value}"
        else:
            llm_config = build_llm_config_from_settings(
                settings, temperature=temperature, streaming=False
            )
            source_prefix = "api:global"

        llm = get_llm(llm_config)
        return llm, f"{source_prefix}:{llm.__class__.__name__}"

    async def aconsult_round1(
        self,
        case: str,
        memory_context: str = "",
        attachments: list[dict[str, Any]] | None = None,
    ) -> RoleOpinion:
        """Round 1：独立思考，看不到他人意见。"""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        if memory_context:
            messages.append({
                "role": "system",
                "content": f"[相关历史病例参考]\n{memory_context}",
            })

        messages.append({
            "role": "user",
            "content": (
                f"【病例】\n{case}\n\n"
                f"【任务】请你作为{self.role_label}，独立给出 Round 1 意见。"
                f"按你 prompt 中的 Round 1 输出格式回答。"
                f"不要参考其他专家意见（你看不到他们）。"
            ),
        })

        source = "unknown"
        status = "ok"
        try:
            llm, source = self._build_llm_with_source(temperature=0.3)
            content = await self._call_llm(llm, messages)
        except Exception as exc:
            status = "failed"
            logger.error("Role %s Round 1 failed: %s", self.role_type.value, exc)
            content = f"[{self.role_label} Round 1 暂时不可用: {exc}]"

        return RoleOpinion(
            role=self.role_type,
            role_label=self.role_label,
            content=content,
            round_num=1,
            tool_calls=[{
                "type": "model_call",
                "role": self.role_type.value,
                "round": 1,
                "backend": source,
                "status": status,
            }],
        )

    async def aconsult_round2(
        self,
        case: str,
        own_round1: RoleOpinion,
        others_round1: list[RoleOpinion],
        memory_context: str = "",
        attachments: list[dict[str, Any]] | None = None,
    ) -> RoleOpinion:
        """Round 2：看到所有他人意见后，强制给出"同意/反对/修正"。

        这是真协作 Harness 的核心——必须 review 他人意见。
        """
        # 构造他人意见的拼接
        others_text = "\n\n".join(
            f"【{op.role_label}的 Round 1 意见】\n{op.content}"
            for op in others_round1
        )

        round2_instruction = (
            "现在你已经看到所有其他专家的 Round 1 意见。\n\n"
            "请严格按以下结构回答（不要省略任何一项）：\n\n"
            "## 同意（Agreements）\n"
            "[列出你同意的他人观点 + 简要理由]\n\n"
            "## 反对（Disagreements）\n"
            "[列出你不同意的他人观点 + 你的反对依据]\n"
            "[如果你完全同意没有反对，必须明确写'无明显分歧']\n\n"
            "## 修正（Revisions）\n"
            "[基于他人新信息，是否修正你 Round 1 的判断？]\n"
            "[如果无需修正，写'坚持 Round 1 判断']\n\n"
            "## Round 2 最终意见\n"
            "[结合上述思考，给出你的更新版意见]"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        if memory_context:
            messages.append({
                "role": "system",
                "content": f"[相关历史病例参考]\n{memory_context}",
            })

        messages.append({
            "role": "user",
            "content": (
                f"【病例】\n{case}\n\n"
                f"【你的 Round 1 意见】\n{own_round1.content}\n\n"
                f"【其他专家的 Round 1 意见】\n{others_text}\n\n"
                f"【Round 2 任务】\n{round2_instruction}"
            ),
        })

        source = "unknown"
        status = "ok"
        try:
            llm, source = self._build_llm_with_source(temperature=0.3)
            content = await self._call_llm(llm, messages)
        except Exception as exc:
            status = "failed"
            logger.error("Role %s Round 2 failed: %s", self.role_type.value, exc)
            content = own_round1.content + f"\n\n[Round 2 失败，沿用 Round 1: {exc}]"

        # 简单解析（论文里也可以做更精细的解析）
        agreements, disagreements, revisions = self._parse_round2(content)

        return RoleOpinion(
            role=self.role_type,
            role_label=self.role_label,
            content=content,
            round_num=2,
            agreements=agreements,
            disagreements=disagreements,
            revisions=revisions,
            tool_calls=[{
                "type": "model_call",
                "role": self.role_type.value,
                "round": 2,
                "backend": source,
                "status": status,
            }],
        )

    @staticmethod
    def _parse_round2(text: str) -> tuple[list[str], list[str], list[str]]:
        """从 Round 2 输出中解析 同意/反对/修正 部分（用于度量协作有效性）。"""
        agreements, disagreements, revisions = [], [], []
        current = None
        for line in text.splitlines():
            stripped = line.strip()
            if "同意" in stripped and stripped.startswith(("##", "**")):
                current = agreements
                continue
            if "反对" in stripped and stripped.startswith(("##", "**")):
                current = disagreements
                continue
            if "修正" in stripped and stripped.startswith(("##", "**")):
                current = revisions
                continue
            if "最终意见" in stripped or "Round 2 最终" in stripped:
                current = None
                continue
            if current is not None and stripped:
                if not stripped.startswith(("#", "**", "[")):
                    current.append(stripped)
        return agreements, disagreements, revisions

    async def _call_llm(self, llm: Any, messages: list[dict[str, str]]) -> str:
        # 如果是 LoRA agent
        if hasattr(llm, "generate") and not hasattr(llm, "ainvoke"):
            system_msg = messages[0]["content"]
            user_msg = messages[-1]["content"]
            return llm.generate(
                system_prompt=system_msg,
                user_text=user_msg,
                max_new_tokens=1024,
                temperature=0.3,
            )

        # 标准 LangChain LLM
        response = await llm.ainvoke(messages)
        return self._extract_content(response)

    @staticmethod
    def _extract_content(response: Any) -> str:
        content = getattr(response, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "".join(parts).strip()
        return str(content or "").strip()
