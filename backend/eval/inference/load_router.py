"""加载训练好的路由分类器，供 Coordinator 使用。

环境变量：
- USE_TRAINED_ROUTER=true   启用训练好的 BERT 路由
- TRAINED_ROUTER_PATH=eval/models/router_bert
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_router_instance: "TrainedRouter | None" = None


class TrainedRouter:
    """加载训练好的 BERT 多标签路由分类器。"""

    def __init__(self, model_path: Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 读 label config
        config_path = model_path / "label_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.labels = config.get("labels", ["physician", "pharmacist", "radiologist"])
            self.threshold = config.get("threshold", 0.5)
            self.max_length = config.get("max_length", 128)
        else:
            self.labels = ["physician", "pharmacist", "radiologist"]
            self.threshold = 0.5
            self.max_length = 128

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device).eval()

        self._torch = torch
        logger.info("TrainedRouter loaded from %s on %s", model_path, self.device)

    def predict(self, query: str) -> dict[str, float]:
        """返回每个角色的概率。"""
        torch = self._torch
        with torch.no_grad():
            enc = self.tokenizer(
                query,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        return {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}

    def route(self, query: str) -> dict[str, bool]:
        """返回每个角色是否需要参与（基于阈值）。"""
        probs = self.predict(query)
        decisions = {label: prob >= self.threshold for label, prob in probs.items()}
        # 主治始终参与
        decisions["physician"] = True
        return decisions


def load_trained_router() -> TrainedRouter | None:
    """全局单例加载（懒加载，第一次调用时初始化）。"""
    global _router_instance

    if os.getenv("USE_TRAINED_ROUTER", "false").lower() not in ("1", "true", "yes"):
        return None

    if _router_instance is not None:
        return _router_instance

    model_path_str = os.getenv("TRAINED_ROUTER_PATH", "eval/models/router_bert")
    # 相对路径相对于 backend 目录
    backend_dir = Path(__file__).resolve().parent.parent.parent
    model_path = (backend_dir / model_path_str).resolve()

    if not model_path.exists():
        logger.warning("Trained router path not found: %s", model_path)
        return None

    try:
        _router_instance = TrainedRouter(model_path)
        return _router_instance
    except Exception as exc:
        logger.error("Failed to load trained router: %s", exc)
        return None
