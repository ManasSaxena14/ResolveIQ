"""Queue and risk decisions from the meaning of a ticket.

Both decisions start from one sentence embedding of the ticket, made by the same
bge-small model that searches the help center. What happens next depends on how
familiar the ticket is, measured as its highest cosine similarity to the past
tickets of the training split:

* **Familiar** tickets look like past ones, so the most similar past tickets vote
  on the queue, and their share of high priority is combined with the XGBoost
  risk model.
* **Unfamiliar** tickets (in practice, the shop's own domain) are decided by
  linear models on the embedding, trained on a small hand-written set of retail
  tickets in ``data/seed``.

Between the two limits the answers are blended. At run time this needs only NumPy
and an embedder, so the full and lite runtimes share it.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from resolveiq.classifier.data import build_text, clean_text
from resolveiq.classifier.predict import STOPWORDS, Prediction

META_FILENAME = "semantic_meta.json"
ARRAYS_FILENAME = "semantic.npz"

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*[A-Za-z]")
_EPS = 1e-6


class EmbedderLike(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def softmax(scores: np.ndarray) -> np.ndarray:
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class SemanticParams:
    """Everything that is not an array; saved as JSON next to the arrays."""

    labels: list[str]
    neighbours: int = 20
    temperature: float = 0.02  # of the neighbours' similarity weights
    familiar_from: float = 0.75  # similarity below which past tickets are ignored
    familiar_full: float = 0.85  # similarity from which only past tickets decide
    queue_scale: float = 1.0  # temperature scaling of the retail queue scores
    urgency_calibration: tuple[float, float] = (1.0, 0.0)  # Platt slope and intercept
    risk_stack: tuple[float, float, float] = (1.0, 0.0, 0.0)  # model, neighbours, intercept
    review_familiar: float = 0.0
    review_unfamiliar: float = 0.0
    risk_threshold: float = 0.25
    examples: list[dict[str, Any]] = field(default_factory=list)
    version: str = "unknown"
    embedding_model: str = ""


@dataclass(frozen=True)
class Evidence:
    """What the model knows about a batch of tickets."""

    familiarity: np.ndarray  # highest similarity to a past ticket
    gate: np.ndarray  # 1: decided by past tickets, 0: by the retail models
    queue: np.ndarray  # probability of each queue
    urgency: np.ndarray  # probability of the retail urgency model
    similar_high: np.ndarray  # weighted share of similar past tickets with high priority
    nearest_example: np.ndarray  # index of the closest urgent retail example


class SemanticModel:
    """Arrays and parameters behind both decisions."""

    def __init__(
        self,
        params: SemanticParams,
        past: np.ndarray,
        past_queue: np.ndarray,
        past_high: np.ndarray,
        examples: np.ndarray,
        queue_weights: np.ndarray,
        urgency_weights: np.ndarray,
    ) -> None:
        self.params = params
        self.past = np.asarray(past, dtype=np.float32)
        self.past_queue = np.asarray(past_queue, dtype=np.int64)
        self.past_high = np.asarray(past_high, dtype=np.float32)
        self.examples = np.asarray(examples, dtype=np.float32)
        # Weights carry the intercept in their last column.
        self.queue_weights = np.asarray(queue_weights, dtype=np.float32)
        self.urgency_weights = np.asarray(urgency_weights, dtype=np.float32)
        self._urgent_rows = np.array(
            [i for i, e in enumerate(params.examples) if e["urgent"]], dtype=np.int64
        )

    # ------------------------------------------------------------------ files

    @classmethod
    def load(cls, model_dir: Path) -> SemanticModel:
        meta = json.loads((model_dir / META_FILENAME).read_text(encoding="utf-8"))
        data = np.load(model_dir / ARRAYS_FILENAME, allow_pickle=False)
        params = SemanticParams(
            **{
                **{k: meta[k] for k in SemanticParams.__dataclass_fields__ if k in meta},
                "urgency_calibration": tuple(meta["urgency_calibration"]),
                "risk_stack": tuple(meta["risk_stack"]),
            }
        )
        return cls(params, *(data[name] for name in _ARRAYS))

    def save(self, model_dir: Path, summary: dict[str, Any] | None = None) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        arrays = {name: getattr(self, name) for name in _ARRAYS}
        arrays["past"] = self.past.astype(np.float16)
        arrays["past_queue"] = self.past_queue.astype(np.int8)
        arrays["past_high"] = self.past_high.astype(np.int8)
        np.savez_compressed(model_dir / ARRAYS_FILENAME, **arrays)
        meta = {**self.params.__dict__, **(summary or {})}
        (model_dir / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # -------------------------------------------------------------- decisions

    def queue_scores(self, vectors: np.ndarray) -> np.ndarray:
        w = self.queue_weights
        return (vectors @ w[:, :-1].T + w[:, -1]) * self.params.queue_scale

    def urgency_scores(self, vectors: np.ndarray) -> np.ndarray:
        """Calibrated log-odds of the retail urgency model."""
        slope, intercept = self.params.urgency_calibration
        w = self.urgency_weights
        return (vectors @ w[:-1] + w[-1]) * slope + intercept

    def familiarity_gate(self, similarity: np.ndarray) -> np.ndarray:
        p = self.params
        return np.clip((similarity - p.familiar_from) / (p.familiar_full - p.familiar_from), 0, 1)

    def evidence(self, vectors: np.ndarray, chunk: int = 512) -> Evidence:
        parts = [self._evidence(vectors[i : i + chunk]) for i in range(0, len(vectors), chunk)]
        return Evidence(
            **{
                name: np.concatenate([getattr(part, name) for part in parts])
                for name in Evidence.__dataclass_fields__
            }
        )

    def _evidence(self, vectors: np.ndarray) -> Evidence:
        vectors = np.asarray(vectors, dtype=np.float32)
        rows = np.arange(len(vectors))
        similarity = vectors @ self.past.T
        k = min(self.params.neighbours, similarity.shape[1])
        index = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        near = np.take_along_axis(similarity, index, axis=1)
        best = near.max(axis=1)
        weights = np.exp((near - best[:, None]) / self.params.temperature)
        weights /= weights.sum(axis=1, keepdims=True)
        votes = np.zeros((len(vectors), len(self.params.labels)))
        for j in range(k):
            np.add.at(votes, (rows, self.past_queue[index[:, j]]), weights[:, j])
        similar_high = (weights * self.past_high[index]).sum(axis=1)

        gate = self.familiarity_gate(best)
        queue = gate[:, None] * votes + (1 - gate[:, None]) * softmax(self.queue_scores(vectors))
        if len(self._urgent_rows):
            closeness = vectors @ self.examples[self._urgent_rows].T
            nearest = self._urgent_rows[closeness.argmax(axis=1)]
        else:
            nearest = np.full(len(vectors), -1)
        return Evidence(
            familiarity=best,
            gate=gate,
            queue=queue,
            urgency=sigmoid(self.urgency_scores(vectors)),
            similar_high=similar_high,
            nearest_example=nearest,
        )

    def familiar_risk(self, model_probability: np.ndarray, similar_high: np.ndarray) -> np.ndarray:
        """The risk model and the similar past tickets combined, for familiar tickets."""
        a, b, c = self.params.risk_stack
        return sigmoid(a * logit(model_probability) + b * soft_share(similar_high) + c)


_ARRAYS = ("past", "past_queue", "past_high", "examples", "queue_weights", "urgency_weights")


def soft_share(share: np.ndarray) -> np.ndarray:
    """Log-odds of a share, pulled in from 0 and 1 so that unanimous votes stay finite."""
    return logit(0.05 + 0.9 * np.asarray(share, dtype=np.float64))


class TextEncoder:
    """Embeds cleaned ticket texts and remembers the last few, as both models need them."""

    def __init__(self, embedder: EmbedderLike, size: int = 64) -> None:
        self.embedder = embedder
        self.size = size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        fresh: dict[str, np.ndarray] = {}
        if missing:
            fresh = dict(zip(missing, self.embedder.embed_documents(missing), strict=True))
        rows = [fresh[t] if t in fresh else self._cache[t] for t in texts]
        for text, vector in fresh.items():
            self._cache[text] = vector
        while len(self._cache) > self.size:
            self._cache.popitem(last=False)
        return np.asarray(rows, dtype=np.float32)


def content_words(text: str, limit: int) -> list[str]:
    """Distinct words worth testing, longest first when there are more than ``limit``."""
    words = dict.fromkeys(
        w.lower() for w in _WORD.findall(text) if len(w) > 2 and w.lower() not in STOPWORDS
    )
    if len(words) <= limit:
        return list(words)
    return sorted(words, key=len, reverse=True)[:limit]


def without_word(text: str, word: str) -> str:
    return re.sub(rf"\b{re.escape(word)}\b", " ", text, flags=re.IGNORECASE)


class SemanticClassifier:
    """Routing queue of a ticket, with the probability of every queue."""

    def __init__(self, model: SemanticModel, encoder: TextEncoder, max_words: int = 20) -> None:
        self.model = model
        self.encoder = encoder
        self.max_words = max_words
        self.labels = list(model.params.labels)

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        cleaned = [clean_text(text) for text in texts]
        if any(not text for text in cleaned):
            raise ValueError("Ticket text must not be empty")
        evidence = self.model.evidence(self.encoder.encode(cleaned))
        params = self.model.params
        predictions = []
        for row, gate in zip(evidence.queue, evidence.gate, strict=True):
            best = int(row.argmax())
            threshold = params.review_familiar if gate >= 0.5 else params.review_unfamiliar
            predictions.append(
                Prediction(
                    label=self.labels[best],
                    confidence=round(float(row[best]), 4),
                    needs_review=bool(row[best] < threshold),
                    probabilities={
                        label: round(float(p), 4) for label, p in zip(self.labels, row, strict=True)
                    },
                )
            )
        return predictions

    def predict_ticket(self, body: str, subject: str = "") -> Prediction:
        return self.predict([build_text(subject, body)])[0]

    def word_weights(self, body: str, subject: str = "", k: int = 8) -> list[tuple[str, float]]:
        """Words whose removal lowers the chosen queue's log-odds, strongest = 1."""
        text = build_text(subject, body)
        words = content_words(text, self.max_words)
        if not words:
            return []
        vectors = self.encoder.encode([text, *(without_word(text, w) for w in words)])
        probabilities = self.model.evidence(vectors).queue
        scores = logit(probabilities[:, int(probabilities[0].argmax())])
        drops = [(w, float(scores[0] - s)) for w, s in zip(words, scores[1:], strict=True)]
        found = sorted((d for d in drops if d[1] > 0.01), key=lambda d: -d[1])[:k]
        return [(w, round(drop / found[0][1], 4)) for w, drop in found]


class SemanticRisk:
    """Escalation risk: the XGBoost model for familiar tickets, meaning for the rest."""

    def __init__(
        self,
        model: SemanticModel,
        encoder: TextEncoder,
        risk_model: Any,
        max_words: int = 20,
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.risk_model = risk_model
        self.max_words = max_words
        self.threshold = model.params.risk_threshold
        self.version = f"{model.params.version} + {getattr(risk_model, 'version', 'unknown')}"

    def probabilities(
        self, texts: Sequence[str], ticket_types: Sequence[str]
    ) -> tuple[np.ndarray, Evidence, list[Any]]:
        evidence = self.model.evidence(self.encoder.encode(list(texts)))
        base = self.risk_model.score_texts(list(texts), list(ticket_types))
        familiar = self.model.familiar_risk(
            np.array([s.escalation_probability for s in base]), evidence.similar_high
        )
        blended = evidence.gate * familiar + (1 - evidence.gate) * evidence.urgency
        return blended, evidence, base

    def score_texts(
        self, texts: Sequence[str], ticket_types: Sequence[str], k: int = 3
    ) -> list[Any]:
        from resolveiq.risk.predict import RiskScore

        if not texts:
            return []
        blended, evidence, base = self.probabilities(texts, ticket_types)
        scores = []
        for i, probability in enumerate(blended):
            value = round(float(probability), 4)
            reasons = self._reasons(i, value, evidence, base[i])
            scores.append(
                RiskScore(
                    escalation_probability=value,
                    alert=value >= self.threshold,
                    reasons=reasons[:k],
                    details=list(base[i].details) if evidence.gate[i] >= 0.5 else [],
                )
            )
        return scores

    def _reasons(self, i: int, value: float, evidence: Evidence, base: Any) -> list[str]:
        if evidence.gate[i] >= 0.5:
            share = float(evidence.similar_high[i])
            past = (
                [f"{share:.0%} of the most similar past tickets were high priority"]
                if share >= 0.5
                else []
            )
            return [*past, *base.reasons]
        nearest = int(evidence.nearest_example[i])
        if value < self.threshold / 2 or nearest < 0:
            return []
        example = self.model.params.examples[nearest]["text"]
        short = example if len(example) <= 90 else example[:87].rstrip() + "..."
        return [f'Close in meaning to an urgent example: "{short}"']

    def score(self, subject: str, body: str, ticket_type: str = "") -> Any:
        return self.score_texts([build_text(subject, body)], [ticket_type])[0]

    def word_weights(self, subject: str, body: str, k: int = 8) -> list[tuple[str, float]]:
        """Words that raise (positive) or lower the score when they are in the text."""
        text = build_text(subject, body)
        if float(self.model.evidence(self.encoder.encode([text])).gate[0]) >= 0.5:
            return self.risk_model.word_weights(subject, body, k)
        words = content_words(text, self.max_words)
        if not words:
            return []
        vectors = self.encoder.encode([text, *(without_word(text, w) for w in words)])
        scores = self.model.urgency_scores(vectors)
        effects = [
            (w, round(float(scores[0] - s), 4)) for w, s in zip(words, scores[1:], strict=True)
        ]
        effects = [e for e in effects if abs(e[1]) > 0.02]
        return sorted(effects, key=lambda e: -abs(e[1]))[:k]


def semantic_ready(model_dir: Path) -> bool:
    return (model_dir / META_FILENAME).is_file() and (model_dir / ARRAYS_FILENAME).is_file()
