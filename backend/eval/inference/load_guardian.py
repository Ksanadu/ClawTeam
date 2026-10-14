"""加载训练好的 Guardian 安全守卫，替换 prompt-based Guardian。

环境变量：
- USE_TRAINED_GUARDIAN=true
- TRAINED_GUARDIAN_PATH=eval/models/guardian_bert
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_guardian_instance: "TrainedGuardian | None" = None


class TrainedGuardian:
    """加载训练好的 BERT 5 类安全分类器。"""

    def __init__(self, model_path: Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        config_path = model_path / "label_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.labels = config.get("labels", ["safe", "injection", "privilege", "privacy", "dangerous"])
            self.id2label = {int(k): v for k, v in config.get("id2label", {}).items()}
            self.max_length = config.get("max_length", 128)
        else:
            self.labels = ["safe", "injection", "privilege", "privacy", "dangerous"]
            self.id2label = {i: l for i, l in enumerate(self.labels)}
            self.max_length = 128

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device).eval()

        self._torch = torch
        logger.info("TrainedGuardian loaded from %s on %s", model_path, self.device)

    def classify(self, text: str) -> tuple[str, float]:
        """返回 (label, confidence)。"""
        torch = self._torch
        with torch.no_grad():
            enc = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        idx = int(probs.argmax())
        label = self.id2label.get(idx, self.labels[idx])
        return label, float(probs[idx])

    def is_safe(self, text: str) -> bool:
        label, _ = self.classify(text)
        return label == "safe"


def load_trained_guardian() -> TrainedGuardian | None:
    global _guardian_instance

    if os.getenv("USE_TRAINED_GUARDIAN", "false").lower() not in ("1", "true", "yes"):
        return None

    if _guardian_instance is not None:
        return _guardian_instance

    model_path_str = os.getenv("TRAINED_GUARDIAN_PATH", "eval/models/guardian_bert")
    backend_dir = Path(__file__).resolve().parent.parent.parent
    model_path = (backend_dir / model_path_str).resolve()

    if not model_path.exists():
        logger.warning("Trained guardian path not found: %s", model_path)
        return None

    try:
        _guardian_instance = TrainedGuardian(model_path)
        return _guardian_instance
    except Exception as exc:
        logger.error("Failed to load trained guardian: %s", exc)
        return None
