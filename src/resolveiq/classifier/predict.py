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


# Common function words that carry no routing signal on their own.
# fmt: off
STOPWORDS = frozenset({
    "the", "and", "for", "you", "your", "our", "are", "was", "were", "this", "that", "with",
    "have", "has", "had", "not", "but", "can", "could", "would", "should", "will", "from",
    "into", "about", "been", "they", "them", "their", "there", "what", "when", "which", "who",
    "how", "all", "any", "its", "out", "now", "just", "where",
    "please", "thank", "thanks", "hello", "dear", "regards",
})
# fmt: on


def word_scores(
    text: str, offsets: Sequence[Sequence[int]], scores: Sequence[float], k: int = 8
) -> list[tuple[str, float]]:
    """Merge token scores into words and keep the ``k`` strongest positive ones.

    A token whose span starts where the previous one ended continues that word when
    both touch with letters or digits. Special tokens have an empty span and are
    skipped, punctuation is dropped, and repeated words add up.
    """
    words: dict[str, float] = {}
    current, current_end, total = "", -1, 0.0

    def flush() -> None:
        word = current.lower()
        if len(word) > 2 and word.isalpha() and word not in STOPWORDS:
            words[word] = words.get(word, 0.0) + total

    for (start, end), score in zip(offsets, scores, strict=True):
        if start == end:
            continue
        piece = text[start:end]
        # Word pieces continue a word only when letters touch; punctuation stands alone.
        if start == current_end and piece[:1].isalnum() and current[-1:].isalnum():
            current += piece
            total += score
        else:
            flush()
            current, total = text[start:end], score
        current_end = end
    flush()
    positive = sorted(((w, s) for w, s in words.items() if s > 0), key=lambda i: -i[1])[:k]
    if not positive:
        return []
    top = positive[0][1]
    return [(word, round(score / top, 4)) for word, score in positive]


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

    def word_weights(self, body: str, subject: str = "", k: int = 8) -> list[tuple[str, float]]:
        """Words that pushed the model toward its predicted queue, strongest first.

        Uses gradient times input on the token embeddings for the predicted class: one
        forward and one backward pass. Word pieces are merged back into whole words with
        the tokenizer's character offsets, and the scores are scaled so the strongest
        word is 1.
        """
        import torch

        text = build_text(subject, body)
        if not text:
            return []
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        inputs = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        embeddings_layer = self.model.get_input_embeddings()
        with torch.enable_grad():
            embeds = embeddings_layer(inputs["input_ids"]).detach().requires_grad_(True)
            logits = self.model(
                inputs_embeds=embeds, attention_mask=inputs["attention_mask"]
            ).logits[0]
            logits[int(logits.argmax())].backward()
            scores = (embeds.grad[0] * embeds[0]).sum(dim=-1).detach().float().cpu().tolist()
        self.model.zero_grad(set_to_none=True)
        return word_scores(text, offsets, scores, k)

    def predict_ticket(self, body: str, subject: str = "") -> Prediction:
        """Predict one ticket from its subject and body, as stored in the dataset."""
        return self.predict([build_text(subject, body)])[0]
