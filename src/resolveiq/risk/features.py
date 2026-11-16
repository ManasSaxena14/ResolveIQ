"""Features for the escalation risk model.

The model predicts at the moment a ticket arrives, so every feature here is
computed from what is known at that moment: the ticket text, the ticket type
chosen on the request form, and the queue predicted by the Phase 2 classifier.
Columns written later (priority, the agent's answer, tags) are never used.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

TICKET_TYPES = ("Incident", "Problem", "Request", "Change")

# Columns that are unknown when the ticket arrives, or that define the label.
# None of them may appear as a feature (checked in the tests).
LEAKAGE_COLUMNS = (
    "priority",
    "answer",
    "queue",
    "tag_1",
    "tag_2",
    "tag_3",
    "tag_4",
    "tag_5",
    "tag_6",
    "tag_7",
    "tag_8",
)

# Phrases that support teams associate with urgent or sensitive tickets. Each
# group becomes one feature: the number of distinct phrases from the group
# found in the ticket.
KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "urgency": (
        "urgent",
        "urgently",
        "asap",
        "as soon as possible",
        "immediately",
        "immediate",
        "critical",
        "emergency",
    ),
    "outage": (
        "outage",
        "down",
        "not working",
        "stopped working",
        "crash",
        "crashed",
        "crashes",
        "unavailable",
        "offline",
        "disruption",
    ),
    "security": (
        "breach",
        "security",
        "hacked",
        "unauthorized",
        "vulnerability",
        "compromised",
        "malware",
        "ransomware",
        "phishing",
    ),
    "data_loss": ("data loss", "lost data", "corrupted", "deleted", "missing data"),
    "money": ("refund", "charged", "overcharged", "double charge", "billing error", "invoice"),
    "churn": ("cancel", "cancellation", "terminate", "switch to another", "close my account"),
    "legal": ("legal", "lawyer", "lawsuit", "attorney", "regulator", "gdpr", "compliance"),
    "frustration": (
        "still",
        "again",
        "multiple times",
        "unacceptable",
        "frustrated",
        "frustrating",
        "disappointed",
        "no response",
    ),
}

KEYWORD_LABELS = {
    "urgency": "urgency",
    "outage": "an outage",
    "security": "a security issue",
    "data_loss": "data loss",
    "money": "money",
    "churn": "cancelling",
    "legal": "legal or compliance terms",
    "frustration": "repeated contact or frustration",
}

TEXT_SCORE_FEATURE = "text_risk_score"

TEXT_FEATURES = (
    "char_count",
    "word_count",
    "exclamation_count",
    "question_count",
    "caps_ratio",
    "sentiment_compound",
    "sentiment_negative",
)


def _keyword_pattern(phrases: Sequence[str]) -> re.Pattern[str]:
    alternatives = sorted((re.escape(p) for p in phrases), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(alternatives) + r")\b", re.IGNORECASE)


_KEYWORD_PATTERNS = {group: _keyword_pattern(p) for group, p in KEYWORD_GROUPS.items()}


def label_from_priority(value: str, positive_values: Sequence[str]) -> int:
    """1 if the source column value counts as an escalation, else 0."""
    positives = {v.strip().lower() for v in positive_values}
    return int(value.strip().lower() in positives)


def keyword_matches(text: str) -> dict[str, list[str]]:
    """Distinct phrases found per keyword group, in order of first appearance."""
    matches: dict[str, list[str]] = {}
    for group, pattern in _KEYWORD_PATTERNS.items():
        found: list[str] = []
        for match in pattern.finditer(text):
            phrase = match.group(1).lower()
            if phrase not in found:
                found.append(phrase)
        matches[group] = found
    return matches


@lru_cache(maxsize=1)
def _sentiment_analyzer() -> Any:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def sentiment(text: str) -> tuple[float, float]:
    """VADER compound score (-1 to 1) and share of negative words."""
    scores = _sentiment_analyzer().polarity_scores(text)
    return float(scores["compound"]), float(scores["neg"])


def text_statistics(text: str) -> dict[str, float]:
    letters = [ch for ch in text if ch.isalpha()]
    caps = sum(ch.isupper() for ch in letters)
    compound, negative = sentiment(text)
    return {
        "char_count": float(len(text)),
        "word_count": float(len(text.split())),
        "exclamation_count": float(text.count("!")),
        "question_count": float(text.count("?")),
        "caps_ratio": round(caps / len(letters), 4) if letters else 0.0,
        "sentiment_compound": compound,
        "sentiment_negative": negative,
    }


@dataclass(frozen=True)
class FeatureSchema:
    """Ordered feature names.

    ``queues`` are the Phase 2 queue labels. ``text_score`` adds the score of a
    TF-IDF model trained to recognize escalation wording.
    """

    queues: tuple[str, ...]
    text_score: bool = True

    @property
    def names(self) -> list[str]:
        return [
            *TEXT_FEATURES,
            *(f"kw_{group}" for group in KEYWORD_GROUPS),
            *(f"type_{t.lower()}" for t in TICKET_TYPES),
            *(f"queue_{queue_slug(q)}" for q in self.queues),
            *([TEXT_SCORE_FEATURE] if self.text_score else []),
        ]

    def as_dict(self) -> dict[str, object]:
        return {
            "queues": list(self.queues),
            "text_score": self.text_score,
            "feature_names": self.names,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> FeatureSchema:
        schema = cls(
            queues=tuple(data["queues"]),  # type: ignore[arg-type]
            text_score=bool(data.get("text_score", False)),
        )
        if schema.names != data["feature_names"]:
            raise ValueError(
                "Stored features were built with a different feature list; rebuild them"
            )
        return schema

    def queue_for_feature(self, name: str) -> str | None:
        for queue in self.queues:
            if name == f"queue_{queue_slug(queue)}":
                return queue
        return None


def queue_slug(queue: str) -> str:
    return re.sub(r"\W+", "_", queue.lower()).strip("_")


def ticket_features(
    text: str,
    ticket_type: str,
    queue_probabilities: Sequence[float],
    schema: FeatureSchema,
    text_score: float | None = None,
) -> list[float]:
    """Feature vector for one ticket, in ``schema.names`` order."""
    if len(queue_probabilities) != len(schema.queues):
        raise ValueError(
            f"Expected {len(schema.queues)} queue probabilities, got {len(queue_probabilities)}"
        )
    if schema.text_score and text_score is None:
        raise ValueError("This schema needs a text score for every ticket")
    stats = text_statistics(text)
    matches = keyword_matches(text)
    return [
        *(stats[name] for name in TEXT_FEATURES),
        *(float(len(matches[group])) for group in KEYWORD_GROUPS),
        *(float(ticket_type.strip().lower() == t.lower()) for t in TICKET_TYPES),
        *(float(p) for p in queue_probabilities),
        *([float(text_score)] if schema.text_score else []),  # type: ignore[arg-type]
    ]


def feature_matrix(
    texts: Sequence[str],
    ticket_types: Sequence[str],
    queue_probabilities: np.ndarray,
    schema: FeatureSchema,
    text_scores: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    scores = [None] * len(texts) if text_scores is None else list(text_scores)
    rows = [
        ticket_features(text, ticket_type, probs, schema, score)
        for text, ticket_type, probs, score in zip(
            texts, ticket_types, queue_probabilities, scores, strict=True
        )
    ]
    return np.asarray(rows, dtype=np.float32).reshape(len(rows), len(schema.names))


def out_of_fold_probabilities(
    texts: Sequence[str], labels: Sequence[str], folds: int, seed: int
) -> tuple[np.ndarray, list[str]]:
    """TF-IDF classifier probabilities for the training split without self-prediction.

    Each ticket is scored by a model that never saw it, so the risk model does
    not learn to trust text scores that are better than they will be on new
    tickets. Used for the queue probabilities and the text score.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from resolveiq.classifier.baseline import build_baseline

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = cross_val_predict(
        build_baseline(seed=seed),
        list(texts),
        list(labels),
        cv=splitter,
        method="predict_proba",
    )
    return probabilities.astype(np.float32), sorted(set(labels))


def load_feature_split(features_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Features, labels and ticket ids written by scripts/build_risk_features.py."""
    path = features_dir / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run scripts/build_risk_features.py first.")
    with np.load(path) as data:
        return data["features"], data["labels"].astype(int), [str(i) for i in data["ticket_ids"]]


def load_feature_schema(features_dir: Path) -> FeatureSchema:
    path = features_dir / "schema.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run scripts/build_risk_features.py first.")
    return FeatureSchema.from_dict(json.loads(path.read_text(encoding="utf-8")))
