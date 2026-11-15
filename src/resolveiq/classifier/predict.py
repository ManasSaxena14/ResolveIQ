"""Load a trained ticket classifier and predict queues with confidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from resolveiq.classifier.data import build_text, clean_text
from resolveiq.classifier.train import predict_proba, read_meta, select_device


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    needs_review: bool
    probabilities: dict[str, float]


class TicketClassifier:
    """Fine-tuned queue classifier.

    Input text is cleaned exactly as during training (line breaks normalized,
    personal data masked). Predictions below the review threshold chosen on
    the validation set are flagged with ``needs_review``.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        labels: Sequence[str],
        max_length: int,
        review_threshold: float,
        device: Any,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.labels = list(labels)
        self.max_length = max_length
        self.review_threshold = review_threshold
        self.device = device

    @classmethod
    def load(
        cls, model_dir: Path, review_threshold: float | None = None, device: str | None = None
    ) -> TicketClassifier:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        meta = read_meta(model_dir)
        torch_device = select_device(device)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(torch_device)
        model.eval()
        threshold = review_threshold
        if threshold is None:
            threshold = float(meta.get("review_threshold", 0.0))
        return cls(
            model=model,
            tokenizer=AutoTokenizer.from_pretrained(model_dir),
            labels=meta["labels"],
            max_length=int(meta["max_length"]),
            review_threshold=threshold,
            device=torch_device,
        )

    def predict(self, texts: Sequence[str], batch_size: int = 32) -> list[Prediction]:
        cleaned = [clean_text(text) for text in texts]
        if any(not text for text in cleaned):
            raise ValueError("Ticket text must not be empty")
        probabilities = predict_proba(
            self.model, self.tokenizer, cleaned, self.max_length, self.device, batch_size
        )
        predictions = []
        for row in probabilities:
            best = int(row.argmax())
            confidence = float(row[best])
            predictions.append(
                Prediction(
                    label=self.labels[best],
                    confidence=round(confidence, 4),
                    needs_review=confidence < self.review_threshold,
                    probabilities={
                        label: round(float(p), 4) for label, p in zip(self.labels, row, strict=True)
                    },
                )
            )
        return predictions

    def predict_ticket(self, body: str, subject: str = "") -> Prediction:
        """Predict one ticket from its subject and body, as stored in the dataset."""
        return self.predict([build_text(subject, body)])[0]
