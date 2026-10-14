"""数据生成器公共工具：读取 .env、调用 LLM、保存 JSONL、断点续传。"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# 让脚本能 import backend 下的模块
EVAL_DIR = Path(__file__).resolve().parent.parent  # backend/eval
BACKEND_DIR = EVAL_DIR.parent  # backend
sys.path.insert(0, str(BACKEND_DIR))

DATA_DIR = EVAL_DIR / "datasets" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_env() -> dict[str, str]:
    """读取 backend/config/.env 中的 LLM 配置。"""
    env_path = BACKEND_DIR / "config" / ".env"
    config: dict[str, str] = {}

    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip().strip('"').strip("'")

    # 兼容已经导出到环境变量的情况
    for key in ["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_PROVIDER"]:
        if key not in config and os.getenv(key):
            config[key] = os.getenv(key, "")

    return config


def get_llm_client():
    """获取 OpenAI 兼容的 LLM 客户端（DeepSeek/智谱/百炼/OpenAI 通用）。"""
    from openai import OpenAI

    config = load_env()
    api_key = config.get("LLM_API_KEY", "")
    base_url = config.get("LLM_BASE_URL", "https://api.deepseek.com")

    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY 未配置。请在 backend/config/.env 中设置。"
        )

    return OpenAI(api_key=api_key, base_url=base_url)


def get_model_name() -> str:
    return load_env().get("LLM_MODEL", "deepseek-chat")


def call_llm(
    system: str,
    user: str,
    *,
    temperature: float = 0.7,
    max_retries: int = 3,
    response_format_json: bool = False,
) -> str:
    """同步调用 LLM，自带重试。"""
    client = get_llm_client()
    model = get_model_name()

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
            }
            if response_format_json:
                # DeepSeek 支持 json_object，其他厂商按需调整
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_error = exc
            wait = 2 ** attempt
            print(f"[LLM] 第 {attempt + 1} 次失败: {exc}，等待 {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"LLM 调用持续失败: {last_error}")


def save_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """保存为 JSONL 格式（一行一个 JSON）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[OK] 已保存 {len(records)} 条到 {output_path}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，文件不存在返回空列表。"""
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def append_jsonl(record: dict[str, Any], output_path: Path) -> None:
    """追加单条记录（用于实时保存，支持中断恢复）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_json_safely(text: str) -> dict[str, Any] | None:
    """从 LLM 输出中提取 JSON（容错，可能带 markdown 代码块）。"""
    text = text.strip()
    # 去 markdown 代码块
    if text.startswith("```"):
        lines = text.split("\n")
        # 去掉首尾的 ``` 行
        text = "\n".join(line for line in lines if not line.startswith("```"))

    # 找第一个 { 和最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
