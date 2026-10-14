"""Pathologist role for Tumor Board.

This role is optionally multimodal. When pathology images are attached, it can
call a Qwen-VL compatible API. When no image is attached, it must be explicit
that it is reasoning from text reports only.
"""

from __future__ import annotations

import os
from typing import Any

from graph.llm import ResolvedLLMConfig, get_llm
from graph.roles.base_role import RoleAgent, RoleOpinion, RoleType


class PathologistAgent(RoleAgent):
    role_type = RoleType.PATHOLOGIST
    role_label = "病理科医生"
    prompt_file = "PATHOLOGIST.md"
    TEXT_ONLY_PREFIX = "未收到病理图像，仅基于文本病理报告/病例信息判断。"

    def _image_attachments(
        self, attachments: list[dict[str, Any]] | None
    ) -> list[dict[str, str]]:
        images: list[dict[str, str]] = []
        for index, item in enumerate(attachments or [], start=1):
            url = (
                item.get("url")
                or item.get("data_url")
                or item.get("image_url")
                or item.get("content")
                or ""
            )
            content_type = str(
                item.get("content_type") or item.get("mime_type") or item.get("type") or ""
            ).lower()
            if not isinstance(url, str) or not url:
                continue
            is_image = (
                content_type.startswith("image/")
                or url.startswith("data:image/")
                or url.lower().split("?")[0].endswith(
                    (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
                )
            )
            if not is_image:
                continue
            images.append(
                {
                    "url": url,
                    "name": str(item.get("name") or f"pathology_image_{index}"),
                    "content_type": content_type or "image/*",
                }
            )
        return images

    def _vision_enabled(self) -> bool:
        return os.getenv("PATHOLOGIST_VISION_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
        }

    def _build_vision_llm(self, temperature: float = 0.2):
        api_key = (
            os.getenv("PATHOLOGIST_VISION_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
        )
        if not api_key:
            raise RuntimeError("PATHOLOGIST_VISION_API_KEY or DASHSCOPE_API_KEY is not configured")

        provider = os.getenv("PATHOLOGIST_VISION_PROVIDER", "bailian")
        model = os.getenv("PATHOLOGIST_VISION_MODEL", "qwen3-vl-plus")
        base_url = os.getenv(
            "PATHOLOGIST_VISION_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        llm = get_llm(
            ResolvedLLMConfig(
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                streaming=False,
            )
        )
        return llm, f"vision:pathologist:{model}"

    @staticmethod
    def _text_only_notice(case: str, image_status: str) -> str:
        return (
            f"{case}\n\n"
            "[Pathology image status]\n"
            f"{image_status}\n"
            "Do not claim that you reviewed a slide, microscopy image, "
            "immunohistochemistry image, CT image, or any other visual material unless "
            "a vision model actually analyzed the attached image in this turn. Reason only "
            "from the textual pathology report and explicitly state when visual confirmation "
            "is unavailable."
        )

    @classmethod
    def _ensure_text_only_prefix(cls, content: str) -> str:
        text = str(content or "").strip()
        variants = (
            cls.TEXT_ONLY_PREFIX,
            "未收到病理图像，仅基于文本判断。",
            "未收到病理图像，仅基于文本病理报告判断。",
            "未收到图片，仅基于文本病理报告/病例信息判断。",
        )
        for phrase in variants:
            if text.startswith(phrase):
                return text
            text = text.replace(phrase, "").strip()
        if not text:
            return cls.TEXT_ONLY_PREFIX
        return f"{cls.TEXT_ONLY_PREFIX}\n\n{text}"

    @staticmethod
    def _image_user_blocks(prompt: str, images: list[dict[str, str]]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image["url"]},
                }
            )
        return blocks

    def _vision_system_prompt(self) -> str:
        return (
            f"{self.system_prompt}\n\n"
            "[Multimodal safety rules]\n"
            "- Only describe visual findings that are actually visible in the attached images.\n"
            "- If no image is attached or image quality is insufficient, say so explicitly.\n"
            "- Do not invent stains, magnification, tumor grade, margins, necrosis, mitotic "
            "figures, or biomarkers that are not visible or stated.\n"
            "- Separate image-based observations from text-report-based inferences.\n"
            "- This is for MDT discussion only and does not replace a licensed pathology report."
        )

    async def aconsult_round1(
        self,
        case: str,
        memory_context: str = "",
        attachments: list[dict[str, Any]] | None = None,
    ) -> RoleOpinion:
        images = self._image_attachments(attachments)
        if not images:
            opinion = await super().aconsult_round1(
                self._text_only_notice(
                    case,
                    "No pathology image was provided in this turn.",
                ),
                memory_context,
                attachments=attachments,
            )
            opinion.content = self._ensure_text_only_prefix(opinion.content)
            opinion.tool_calls.append(
                {
                    "type": "attachment_status",
                    "role": self.role_type.value,
                    "round": 1,
                    "status": "no_image",
                    "image_count": 0,
                }
            )
            return opinion

        if not self._vision_enabled():
            opinion = await super().aconsult_round1(
                self._text_only_notice(
                    case,
                    "Pathology image attachments were provided, but vision analysis is disabled.",
                ),
                memory_context,
                attachments=attachments,
            )
            opinion.content = (
                "[病理图像已上传，但 PATHOLOGIST_VISION_ENABLED=false，未进行图像阅片。]\n\n"
                + opinion.content
            )
            opinion.tool_calls.append(
                {
                    "type": "model_call",
                    "role": self.role_type.value,
                    "round": 1,
                    "backend": "vision:pathologist",
                    "status": "disabled",
                    "image_count": len(images),
                }
            )
            return opinion

        source = "vision:pathologist"
        status = "ok"
        try:
            llm, source = self._build_vision_llm(temperature=0.2)
            image_names = ", ".join(image["name"] for image in images)
            prompt = (
                f"[Case]\n{case}\n\n"
                f"[Attached pathology images]\n{image_names}\n\n"
                "[Task]\n"
                "Give an independent Round 1 pathology opinion for the tumor board. "
                "First state which conclusions are based on attached images and which are "
                "based only on text. If the images do not support a finding, say it is not assessable."
            )
            response = await llm.ainvoke(
                [
                    {"role": "system", "content": self._vision_system_prompt()},
                    {"role": "user", "content": self._image_user_blocks(prompt, images)},
                ]
            )
            content = self._extract_content(response)
        except Exception as exc:
            status = "failed"
            opinion = await super().aconsult_round1(
                self._text_only_notice(
                    case,
                    "Pathology image attachments were provided, but the vision model call failed.",
                ),
                memory_context,
                attachments=attachments,
            )
            opinion.content = (
                f"[病理图像已上传，但视觉模型调用失败：{exc}。以下仅基于文本病理信息。]\n\n"
                + opinion.content
            )
            opinion.tool_calls.append(
                {
                    "type": "model_call",
                    "role": self.role_type.value,
                    "round": 1,
                    "backend": source,
                    "status": status,
                    "image_count": len(images),
                }
            )
            return opinion

        return RoleOpinion(
            role=self.role_type,
            role_label=self.role_label,
            content=content,
            round_num=1,
            tool_calls=[
                {
                    "type": "model_call",
                    "role": self.role_type.value,
                    "round": 1,
                    "backend": source,
                    "status": status,
                    "image_count": len(images),
                }
            ],
        )

    async def aconsult_round2(
        self,
        case: str,
        own_round1: RoleOpinion,
        others_round1: list[RoleOpinion],
        memory_context: str = "",
        attachments: list[dict[str, Any]] | None = None,
    ) -> RoleOpinion:
        images = self._image_attachments(attachments)
        if not images or not self._vision_enabled():
            image_status = (
                "No pathology image was provided in this turn."
                if not images
                else "Pathology image attachments were provided, but vision analysis is disabled."
            )
            opinion = await super().aconsult_round2(
                self._text_only_notice(case, image_status),
                own_round1,
                others_round1,
                memory_context,
                attachments=attachments,
            )
            if not images:
                opinion.content = self._ensure_text_only_prefix(opinion.content)
            opinion.tool_calls.append(
                {
                    "type": "attachment_status",
                    "role": self.role_type.value,
                    "round": 2,
                    "status": "no_image" if not images else "vision_disabled",
                    "image_count": len(images),
                }
            )
            return opinion

        others_text = "\n\n".join(
            f"[{op.role_label} Round 1]\n{op.content}" for op in others_round1
        )
        source = "vision:pathologist"
        status = "ok"
        try:
            llm, source = self._build_vision_llm(temperature=0.2)
            prompt = (
                f"[Case]\n{case}\n\n"
                f"[Your Round 1 opinion]\n{own_round1.content}\n\n"
                f"[Other specialists' Round 1 opinions]\n{others_text}\n\n"
                "[Task]\n"
                "Give the pathologist Round 2 response. Use sections: Agreements, "
                "Disagreements, Revisions, Round 2 final opinion. Do not invent image "
                "findings; if an image does not resolve a dispute, say it remains unassessable."
            )
            response = await llm.ainvoke(
                [
                    {"role": "system", "content": self._vision_system_prompt()},
                    {"role": "user", "content": self._image_user_blocks(prompt, images)},
                ]
            )
            content = self._extract_content(response)
        except Exception as exc:
            status = "failed"
            opinion = await super().aconsult_round2(
                self._text_only_notice(
                    case,
                    "Pathology image attachments were provided, but the Round 2 vision model call failed.",
                ),
                own_round1,
                others_round1,
                memory_context,
                attachments=attachments,
            )
            opinion.content = (
                f"[病理图像已上传，但 Round 2 视觉模型调用失败：{exc}。以下仅基于文本信息。]\n\n"
                + opinion.content
            )
            opinion.tool_calls.append(
                {
                    "type": "model_call",
                    "role": self.role_type.value,
                    "round": 2,
                    "backend": source,
                    "status": status,
                    "image_count": len(images),
                }
            )
            return opinion

        agreements, disagreements, revisions = self._parse_round2(content)
        return RoleOpinion(
            role=self.role_type,
            role_label=self.role_label,
            content=content,
            round_num=2,
            agreements=agreements,
            disagreements=disagreements,
            revisions=revisions,
            tool_calls=[
                {
                    "type": "model_call",
                    "role": self.role_type.value,
                    "round": 2,
                    "backend": source,
                    "status": status,
                    "image_count": len(images),
                }
            ],
        )
