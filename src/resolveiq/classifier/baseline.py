"""TF-IDF and logistic regression baseline.

This is the number the fine-tuned transformer must beat. It is trained on the
same cleaned text and splits, with balanced class weights to match the weighted
loss used for the transformer.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def build_baseline(seed: int = 42, c: float = 4.0) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True
                ),
            ),
            (
                "clf",
                LogisticRegression(C=c, max_iter=3000, class_weight="balanced", random_state=seed),
            ),
        ]
    )


def train_baseline(texts: Sequence[str], labels: Sequence[str], seed: int = 42, c: float = 4.0):
    model = build_baseline(seed=seed, c=c)
    model.fit(list(texts), list(labels))
    return model


def predict_with_confidence(model: Any, texts: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """Predicted labels and the probability of the predicted label."""
    probabilities = model.predict_proba(list(texts))
    classes = model.classes_
    best = probabilities.argmax(axis=1)
    return [str(classes[i]) for i in best], probabilities.max(axis=1)


def save_baseline(model: Any, path: Path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_baseline(path: Path) -> Any:
    """Load a baseline saved by this project. Only load files you created."""
    import joblib

    return joblib.load(path)
