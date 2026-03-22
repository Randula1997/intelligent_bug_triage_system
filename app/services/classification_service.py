from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ClassificationService:
    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path
        self.enabled = False
        self.tokenizer = None
        self.model = None
        self.id_to_label: dict[int, str] = {}

        if checkpoint_path and Path(checkpoint_path).exists():
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
            config_labels = getattr(self.model.config, "id2label", {})
            self.id_to_label = {int(key): value for key, value in config_labels.items()}
            self.model.eval()
            self.enabled = True

    def predict(self, text: str, top_k: int) -> list[dict[str, float | str]]:
        if not self.enabled or self.tokenizer is None or self.model is None:
            return []

        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1)[0]
        limit = min(top_k, probabilities.shape[0])
        scores, indices = torch.topk(probabilities, k=limit)

        predictions: list[dict[str, float | str]] = []
        for score, index in zip(scores.tolist(), indices.tolist()):
            developer_name = self.id_to_label.get(index, str(index))
            predictions.append(
                {
                    "developer_name": developer_name,
                    "classifier_score": float(score),
                }
            )
        return predictions
