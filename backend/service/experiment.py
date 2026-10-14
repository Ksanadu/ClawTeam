"""实验模式管理 —— 支持 4 组对比实验和日志记录。"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentMode(str, Enum):
    """四组对比实验模式。"""

    SINGLE_AGENT = "single"  # 单智能体基线
    MULTI_NO_MEMORY = "multi_no_memory"  # 多角色，无共享记忆
    MULTI_WITH_MEMORY = "multi_memory"  # 多角色 + 共享记忆
    MULTI_FULL = "multi_full"  # 多角色 + 共享记忆 + Guardian

    @property
    def use_multi_agent(self) -> bool:
        return self != ExperimentMode.SINGLE_AGENT

    @property
    def use_shared_memory(self) -> bool:
        return self in (ExperimentMode.MULTI_WITH_MEMORY, ExperimentMode.MULTI_FULL)

    @property
    def use_guardian(self) -> bool:
        return self == ExperimentMode.MULTI_FULL


@dataclass
class ExperimentLog:
    """单次实验日志记录。"""

    session_id: str = ""
    experiment_mode: str = ""
    query: str = ""
    roles_called: list[str] = field(default_factory=list)
    routing_reason: str = ""
    role_opinions: dict[str, str] = field(default_factory=dict)
    final_answer: str = ""
    guardian_verdict: str = ""
    latency_ms: int = 0
    total_tokens: int = 0
    tool_calls_count: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentLogger:
    """实验日志记录器 —— 写入 JSONL 文件。"""

    def __init__(self) -> None:
        self._log_dir: Path | None = None

    def configure(self, base_dir: Path) -> None:
        self._log_dir = base_dir / "storage" / "experiment_logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, entry: ExperimentLog) -> None:
        if self._log_dir is None:
            logger.warning("ExperimentLogger not configured, skipping log")
            return

        entry.timestamp = time.time()
        log_file = self._log_dir / f"experiment_{entry.experiment_mode}.jsonl"

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.error("Failed to write experiment log: %s", exc)

    def load_logs(self, mode: str) -> list[dict[str, Any]]:
        """读取指定实验模式的所有日志。"""
        if self._log_dir is None:
            return []

        log_file = self._log_dir / f"experiment_{mode}.jsonl"
        if not log_file.exists():
            return []

        logs = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return logs


# 全局单例
experiment_logger = ExperimentLogger()
