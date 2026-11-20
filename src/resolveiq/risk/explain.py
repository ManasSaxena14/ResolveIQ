"""Per-ticket explanations of escalation scores.

SHAP values are computed with XGBoost's built-in TreeSHAP implementation
(``pred_contribs=True``), the same algorithm as ``shap.TreeExplainer``. Each
value is a feature's contribution to the score in log-odds. The features that
push the score up the most are turned into plain sentences for agents.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from resolveiq.risk.features import (
    KEYWORD_LABELS,
    TEXT_SCORE_FEATURE,
    FeatureSchema,
    keyword_matches,
)


@dataclass(frozen=True)
class Reason:
    feature: str
    value: float
    contribution: float
    text: str


def shap_values(booster: Any, features: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """Per-feature contributions; the last column is the bias term."""
    from resolveiq.risk.train import dmatrix, thread_safe

    matrix = dmatrix(features, feature_names)
    return np.asarray(thread_safe(booster).predict(matrix, pred_contribs=True))


MIN_CONTRIBUTION = 0.05


def text_model_terms(text_model: Any, text: str, k: int = 3) -> list[str]:
    """Words in ``text`` that raise the text model's escalation score the most.

    The text model is a TF-IDF and logistic regression pipeline, so a term's
    contribution is its TF-IDF weight times its coefficient.
    """
    vectorizer = text_model.named_steps["tfidf"]
    coefficients = text_model.named_steps["clf"].coef_[0]
    row = vectorizer.transform([text]).tocoo()
    contributions = row.data * coefficients[row.col]
    vocabulary = vectorizer.get_feature_names_out()
    order = np.argsort(-contributions)
    return [str(vocabulary[row.col[i]]) for i in order[:k] if contributions[i] > 0]


def text_model_contributions(text_model: Any, text: str, k: int = 8) -> list[tuple[str, float]]:
    """Terms in ``text`` with the largest effect on the text score, either direction.

    Positive weights raise the escalation score and negative weights lower it, in
    log-odds of the text model. Sorted by absolute effect.
    """
    vectorizer = text_model.named_steps["tfidf"]
    coefficients = text_model.named_steps["clf"].coef_[0]
    row = vectorizer.transform([text]).tocoo()
    contributions = row.data * coefficients[row.col]
    vocabulary = vectorizer.get_feature_names_out()
    order = np.argsort(-np.abs(contributions))
    return [
        (str(vocabulary[row.col[i]]), round(float(contributions[i]), 4))
        for i in order[:k]
        if abs(contributions[i]) > 1e-6
    ]


def describe_feature(
    name: str,
    value: float,
    text: str,
    ticket_type: str,
    schema: FeatureSchema,
    text_terms: Sequence[str] = (),
) -> str:
    """A short sentence that explains a feature value to a support agent."""
    if name.startswith("kw_"):
        group = name[3:]
        phrases = keyword_matches(text).get(group, [])
        if phrases:
            quoted = ", ".join(f'"{p}"' for p in phrases[:3])
            return f"Message mentions {quoted}"
        return f"Message does not mention {KEYWORD_LABELS.get(group, group)}"
    if name.startswith("type_"):
        return f"Ticket type is {ticket_type or 'unknown'}"
    if name == TEXT_SCORE_FEATURE:
        if text_terms:
            quoted = ", ".join(f'"{term}"' for term in text_terms)
            return f"Wording typical of high-priority tickets: {quoted}"
        return f"Wording resembles past high-priority tickets (text score {value:.2f})"
    queue = schema.queue_for_feature(name)
    if queue is not None:
        if value >= 0.5:
            return f"Predicted queue is {queue} ({value:.0%})"
        return f"Low chance of the {queue} queue ({value:.0%})"
    if name == "char_count":
        size = "Long" if value >= 600 else "Short"
        return f"{size} message ({int(value):,} characters)"
    if name == "word_count":
        return f"Message has {int(value):,} words"
    if name == "exclamation_count":
        count = int(value)
        return f"{count} exclamation mark{'s' if count != 1 else ''}"
    if name == "question_count":
        count = int(value)
        return f"{count} question mark{'s' if count != 1 else ''}"
    if name == "caps_ratio":
        return f"{value:.0%} of letters are capitals"
    if name == "sentiment_compound":
        tone = "Negative" if value < -0.05 else "Positive" if value > 0.05 else "Neutral"
        return f"{tone} tone (sentiment {value:+.2f})"
    if name == "sentiment_negative":
        return f"{value:.0%} of the wording is negative"
    return f"{name} = {value:g}"


def top_reasons(
    contributions: np.ndarray,
    feature_row: np.ndarray,
    schema: FeatureSchema,
    text: str,
    ticket_type: str,
    k: int = 3,
    text_terms: Sequence[str] = (),
    min_contribution: float = MIN_CONTRIBUTION,
) -> list[Reason]:
    """Up to ``k`` features that raise this ticket's score the most.

    Only contributions of at least ``min_contribution`` log-odds count as
    reasons, so negligible effects are not presented as explanations. One-hot
    features of the same group (for example the four ticket types) produce one
    sentence at most.
    """
    names = schema.names
    order = np.argsort(-contributions[: len(names)])
    reasons: list[Reason] = []
    seen: set[str] = set()
    for index in order:
        contribution = float(contributions[index])
        if contribution < min_contribution or len(reasons) == k:
            break
        name = names[index]
        sentence = describe_feature(
            name, float(feature_row[index]), text, ticket_type, schema, text_terms
        )
        if sentence in seen:
            continue
        seen.add(sentence)
        reasons.append(Reason(name, float(feature_row[index]), round(contribution, 4), sentence))
    return reasons


def global_importance(
    contributions: np.ndarray, feature_names: Sequence[str]
) -> list[tuple[str, float]]:
    """Mean absolute SHAP value per feature, largest first."""
    values = np.abs(contributions[:, : len(feature_names)]).mean(axis=0)
    order = np.argsort(-values)
    return [(feature_names[i], round(float(values[i]), 4)) for i in order]


def plot_shap_summary(
    contributions: np.ndarray,
    features: np.ndarray,
    feature_names: Sequence[str],
    path: Path,
    max_display: int = 15,
    seed: int = 0,
) -> None:
    """Beeswarm-style summary: one row per feature, colored by feature value."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shap = contributions[:, : len(feature_names)]
    order = np.argsort(-np.abs(shap).mean(axis=0))[:max_display][::-1]
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(order) + 1.5))
    for row, index in enumerate(order):
        values = features[:, index].astype(float)
        low, high = np.percentile(values, [5, 95])
        spread = high - low
        scaled = np.clip((values - low) / spread, 0, 1) if spread > 0 else values * 0
        jitter = rng.uniform(-0.3, 0.3, size=len(values))
        points = ax.scatter(
            shap[:, index], row + jitter, c=scaled, cmap="coolwarm", s=4, alpha=0.6, linewidths=0
        )
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order])
    ax.set_xlabel("SHAP value (effect on escalation log-odds)")
    ax.set_title("Escalation risk: feature effects on the test split")
    colorbar = fig.colorbar(points, ax=ax, pad=0.01)
    colorbar.set_label("Feature value (low to high)")
    colorbar.set_ticks([])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
