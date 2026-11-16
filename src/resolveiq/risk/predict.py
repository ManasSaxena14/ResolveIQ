"""Load a trained escalation model and score tickets.

A saved model directory holds four files:

- ``xgboost.json``: the booster, cut to its best iteration
- ``calibrator.joblib``: maps raw scores to calibrated probabilities
- ``queue_model.joblib``: the TF-IDF queue classifier that provides queue features
- ``text_model.joblib``: the TF-IDF model behind the text score (if used)
- ``risk_meta.json``: feature names, queues, alert threshold and training summary
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from resolveiq.classifier.data import build_text
from resolveiq.risk.explain import Reason, shap_values, text_model_terms, top_reasons
from resolveiq.risk.features import FeatureSchema, feature_matrix
from resolveiq.risk.train import (
    ProbabilityCalibrator,
    TreeModel,
    dmatrix,
    load_booster,
    thread_safe,
)

BOOSTER_FILENAME = "xgboost.json"
CALIBRATOR_FILENAME = "calibrator.joblib"
QUEUE_MODEL_FILENAME = "queue_model.joblib"
TEXT_MODEL_FILENAME = "text_model.joblib"
META_FILENAME = "risk_meta.json"


@dataclass(frozen=True)
class RiskScore:
    escalation_probability: float
    alert: bool
    reasons: list[str] = field(default_factory=list)
    details: list[Reason] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "escalation_probability": self.escalation_probability,
            "alert": self.alert,
            "reasons": self.reasons,
        }


def save_risk_model(
    out_dir: Path,
    tree: TreeModel,
    calibrator: ProbabilityCalibrator,
    queue_model: Any,
    text_model: Any,
    meta: dict[str, Any],
) -> None:
    import joblib

    out_dir.mkdir(parents=True, exist_ok=True)
    tree.save(out_dir / BOOSTER_FILENAME)
    joblib.dump(calibrator, out_dir / CALIBRATOR_FILENAME)
    joblib.dump(queue_model, out_dir / QUEUE_MODEL_FILENAME)
    if text_model is not None:
        joblib.dump(text_model, out_dir / TEXT_MODEL_FILENAME)
    (out_dir / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")


class RiskModel:
    """Calibrated escalation probability, alert flag and top reasons per ticket."""

    def __init__(
        self,
        booster: Any,
        calibrator: ProbabilityCalibrator,
        queue_model: Any,
        schema: FeatureSchema,
        threshold: float,
        text_model: Any = None,
        version: str = "unknown",
    ) -> None:
        if schema.text_score and text_model is None:
            raise ValueError("The feature schema needs a text model")
        self.booster = booster
        self.calibrator = calibrator
        self.queue_model = queue_model
        self.text_model = text_model
        self.schema = schema
        self.threshold = threshold
        self.version = version

    @classmethod
    def load(cls, model_dir: Path) -> RiskModel:
        """Load a model saved by this project. Only load files you created."""
        import joblib

        meta_path = model_dir / META_FILENAME
        if not meta_path.is_file():
            raise FileNotFoundError(f"{meta_path} not found. Train the risk model first.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        schema = FeatureSchema.from_dict(meta["schema"])
        text_path = model_dir / TEXT_MODEL_FILENAME
        return cls(
            booster=load_booster(model_dir / BOOSTER_FILENAME),
            calibrator=joblib.load(model_dir / CALIBRATOR_FILENAME),
            queue_model=joblib.load(model_dir / QUEUE_MODEL_FILENAME),
            schema=schema,
            threshold=float(meta["threshold"]),
            text_model=joblib.load(text_path) if schema.text_score else None,
            version=str(meta.get("version", "unknown")),
        )

    def features(self, texts: Sequence[str], ticket_types: Sequence[str]) -> np.ndarray:
        queue_probabilities = self.queue_model.predict_proba(list(texts))
        classes = [str(c) for c in self.queue_model.classes_]
        if classes != list(self.schema.queues):
            order = [classes.index(q) for q in self.schema.queues]
            queue_probabilities = queue_probabilities[:, order]
        text_scores = (
            self.text_model.predict_proba(list(texts))[:, 1] if self.schema.text_score else None
        )
        return feature_matrix(texts, ticket_types, queue_probabilities, self.schema, text_scores)

    def score_texts(
        self, texts: Sequence[str], ticket_types: Sequence[str], k: int = 3
    ) -> list[RiskScore]:
        """Score already cleaned ticket texts (subject and body combined)."""
        if not texts:
            return []
        features = self.features(texts, ticket_types)
        raw = thread_safe(self.booster).predict(dmatrix(features, self.schema.names))
        probabilities = self.calibrator.transform(raw)
        contributions = shap_values(self.booster, features, self.schema.names)
        scores = []
        for i, (text, ticket_type) in enumerate(zip(texts, ticket_types, strict=True)):
            terms = text_model_terms(self.text_model, text) if self.text_model else []
            reasons = top_reasons(
                contributions[i], features[i], self.schema, text, ticket_type, k, terms
            )
            probability = round(float(probabilities[i]), 4)
            scores.append(
                RiskScore(
                    escalation_probability=probability,
                    alert=probability >= self.threshold,
                    reasons=[r.text for r in reasons],
                    details=reasons,
                )
            )
        return scores

    def score(self, subject: str, body: str, ticket_type: str = "") -> RiskScore:
        """Score one raw ticket. The text is cleaned and masked first."""
        return self.score_texts([build_text(subject, body)], [ticket_type])[0]


def score_csv(model: RiskModel, input_path: Path, output_path: Path) -> int:
    """Score a CSV of tickets and write them ranked by escalation probability.

    The input needs ``subject`` and ``body`` columns; ``type`` and ``id`` are
    optional. The output keeps the id (or row number), never the ticket text.
    """
    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and not {"subject", "body"} <= set(rows[0]):
        raise ValueError("The input CSV needs 'subject' and 'body' columns")
    texts = [build_text(r.get("subject", ""), r.get("body", "")) for r in rows]
    types = [r.get("type", "") or "" for r in rows]
    scores = model.score_texts(texts, types)
    ranked = sorted(
        enumerate(zip(rows, scores, strict=True), start=1),
        key=lambda item: item[1][1].escalation_probability,
        reverse=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["rank", "id", "escalation_probability", "alert", "reason_1", "reason_2", "reason_3"]
        )
        for rank, (number, (row, score)) in enumerate(ranked, start=1):
            reasons = [*score.reasons, "", "", ""][:3]
            row_id = row.get("id") or str(number)
            writer.writerow(
                [rank, row_id, score.escalation_probability, int(score.alert), *reasons]
            )
    return len(rows)
