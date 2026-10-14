"""加载 LoRA 微调的角色 Agent，替换 base_role.py 中的 LLM 调用。

环境变量（每个角色独立配置）：
- USE_LORA_PHYSICIAN=true
- LORA_PHYSICIAN_BASE=Qwen/Qwen2.5-0.5B-Instruct
- LORA_PHYSICIAN_PATH=eval/models/physician_lora

- USE_LORA_PHARMACIST=true
- LORA_PHARMACIST_BASE=Qwen/Qwen2.5-0.5B-Instruct
- LORA_PHARMACIST_PATH=eval/models/pharmacist_lora

- USE_LORA_RADIOLOGIST=true
- LORA_RADIOLOGIST_BASE=Qwen/Qwen2-VL-2B-Instruct
- LORA_RADIOLOGIST_PATH=eval/models/radiologist_vl_lora
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 全局缓存：(role_type) -> LoraRoleAgent
_lora_cache: dict[str, "LoraRoleAgent"] = {}
_local_base_cache: dict[str, "LoraRoleAgent"] = {}


class LoraRoleAgent:
    """加载基座模型 + LoRA adapter，封装为角色推理接口。

    每个角色独立加载（可能用不同基座，比如影像科用 VL 模型）。
    """

    def __init__(
        self,
        role: str,
        base_model: str,
        adapter_path: Path | None,
        is_multimodal: bool = False,
    ) -> None:
        import torch
        from transformers import AutoTokenizer

        self.role = role
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.is_multimodal = is_multimodal
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch

        if is_multimodal:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
            base = AutoModelForVision2Seq.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.tokenizer = self.processor.tokenizer
        else:
            from transformers import AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.processor = None

        # 加载 LoRA adapter
        if adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base, str(adapter_path))
        else:
            self.model = base
        self.model.eval()

        logger.info(
            "LoraRoleAgent loaded: role=%s, base=%s, adapter=%s, multimodal=%s",
            role, base_model, adapter_path, is_multimodal,
        )

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        image: Any | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        torch = self._torch

        if self.is_multimodal and image is not None:
            return self._generate_multimodal(system_prompt, user_text, image, max_new_tokens, temperature)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    def _generate_multimodal(
        self,
        system_prompt: str,
        user_text: str,
        image: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        torch = self._torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
            )
        response = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]
        return response.strip()


def load_lora_role(role: str) -> LoraRoleAgent | None:
    """根据角色名加载对应 LoRA。

    v3.1 支持的角色（仅外科 / 内科训练）：
    - 'surgeon'              → USE_LORA_SURGEON
    - 'medical_oncologist'   → USE_LORA_MEDICAL_ONCOLOGIST
    - 'pathologist'          → 默认不训练（用 API + prompt）
    - 'radiation_oncologist' → 默认不训练
    """
    global _lora_cache
    if role in _lora_cache:
        return _lora_cache[role]

    role_upper = role.upper()
    if os.getenv(f"USE_LORA_{role_upper}", "false").lower() not in ("1", "true", "yes"):
        return None

    base_model = os.getenv(f"LORA_{role_upper}_BASE", "")
    adapter_path_str = os.getenv(f"LORA_{role_upper}_PATH", "")

    if not base_model or not adapter_path_str:
        logger.warning("LoRA config incomplete for role=%s", role)
        return None

    backend_dir = Path(__file__).resolve().parent.parent.parent
    adapter_path = (backend_dir / adapter_path_str).resolve()

    if not adapter_path.exists():
        logger.warning("LoRA adapter not found for role=%s: %s", role, adapter_path)
        return None

    is_multimodal = role == "radiologist" or "VL" in base_model.upper()

    try:
        agent = LoraRoleAgent(
            role=role,
            base_model=base_model,
            adapter_path=adapter_path,
            is_multimodal=is_multimodal,
        )
        _lora_cache[role] = agent
        return agent
    except Exception as exc:
        logger.error("Failed to load LoRA for role=%s: %s", role, exc)
        return None


def load_local_base_role(role: str) -> LoraRoleAgent | None:
    """Load a local base model for a role without attaching a LoRA adapter.

    This supports strict LoRA ablations:
    local Qwen base experts vs the same local Qwen base plus LoRA.
    """
    global _local_base_cache

    role_upper = role.upper()
    if os.getenv(f"USE_LOCAL_BASE_{role_upper}", "false").lower() not in ("1", "true", "yes"):
        return None

    base_model = (
        os.getenv(f"LOCAL_BASE_{role_upper}_MODEL")
        or os.getenv(f"LORA_{role_upper}_BASE")
        or ""
    )
    if not base_model:
        logger.warning("Local base config incomplete for role=%s", role)
        return None

    is_multimodal = role == "radiologist" or "VL" in base_model.upper()
    cache_key = f"{base_model}|{is_multimodal}"
    if cache_key in _local_base_cache:
        return _local_base_cache[cache_key]

    try:
        agent = LoraRoleAgent(
            role=role,
            base_model=base_model,
            adapter_path=None,
            is_multimodal=is_multimodal,
        )
        _local_base_cache[cache_key] = agent
        return agent
    except Exception as exc:
        logger.error("Failed to load local base for role=%s: %s", role, exc)
        return None
