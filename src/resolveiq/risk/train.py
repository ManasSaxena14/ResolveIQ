"""Training of the escalation risk models.

Three models are trained on the same features: a majority-class reference, a
logistic regression baseline on scaled features, and gradient-boosted trees
(XGBoost) with early stopping on the validation split. The best tree model is
then calibrated on the validation split so its scores can be read as
probabilities.
"""

from __future__ import annotations

import itertools
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from resolveiq.risk.evaluate import pr_auc

logger = logging.getLogger(__name__)


def xgboost_threads() -> int | None:
    """Thread count for XGBoost, or ``None`` for its default (all cores).

    On macOS, PyTorch and XGBoost link different OpenMP runtimes. Running
    multithreaded XGBoost after PyTorch has started its thread pool crashes the
    process, so a single thread is used whenever PyTorch is already loaded (for
    example in a service that also serves the transformer classifier).
    """
    if sys.platform == "darwin" and "torch" in sys.modules:
        return 1
    return None


def dmatrix(features: np.ndarray, feature_names: Sequence[str], labels: Any = None) -> Any:
    """An XGBoost DMatrix that respects :func:`xgboost_threads`."""
    import xgboost as xgb

    threads = xgboost_threads()
    return xgb.DMatrix(
        features,
        label=labels,
        feature_names=list(feature_names),
        nthread=threads,
    )


def thread_safe(booster: Any) -> Any:
    """Apply :func:`xgboost_threads` to a booster before predicting with it."""
    threads = xgboost_threads()
    if threads is not None:
        booster.set_param({"nthread": threads})
    return booster


def build_logistic_baseline(seed: int = 42) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            ),
        ]
    )


def scale_pos_weight(labels: Sequence[int] | np.ndarray) -> float:
    """Ratio of negatives to positives, the usual XGBoost imbalance setting."""
    labels = np.asarray(labels)
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("The training labels contain no escalations")
    return float((len(labels) - positives) / positives)


@dataclass
class TreeModel:
    """A trained booster cut to its best iteration, plus how it was trained."""

    booster: Any
    feature_names: list[str]
    params: dict[str, float]
    best_iteration: int
    val_pr_auc: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        return thread_safe(self.booster).predict(dmatrix(features, self.feature_names))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(path)


def load_booster(path: Path) -> Any:
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(path)
    return booster


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Sequence[str],
    params: Mapping[str, float],
    seed: int = 42,
) -> TreeModel:
    """Fit XGBoost with early stopping on validation PR-AUC."""
    import xgboost as xgb

    params = dict(params)
    rounds = int(params.pop("n_estimators", 1000))
    early_stopping = int(params.pop("early_stopping_rounds", 50))
    booster_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "seed": seed,
        "scale_pos_weight": scale_pos_weight(y_train),
        "eta": float(params.pop("learning_rate", 0.05)),
        "max_depth": int(params.pop("max_depth", 5)),
        **params,
    }
    threads = xgboost_threads()
    if threads is not None:
        booster_params["nthread"] = threads
    names = list(feature_names)
    train_matrix = dmatrix(x_train, names, y_train)
    val_matrix = dmatrix(x_val, names, y_val)
    booster = xgb.train(
        booster_params,
        train_matrix,
        num_boost_round=rounds,
        evals=[(val_matrix, "validation")],
        early_stopping_rounds=early_stopping,
        verbose_eval=False,
    )
    best_iteration = int(booster.best_iteration)
    # Keep only the trees up to the best iteration, so the saved booster gives
    # the same scores without any early-stopping bookkeeping.
    tree = TreeModel(
        booster=booster[: best_iteration + 1],
        feature_names=names,
        params={**booster_params, "n_estimators": best_iteration + 1},
        best_iteration=best_iteration,
        val_pr_auc=0.0,
    )
    tree.val_pr_auc = pr_auc(y_val, tree.predict(x_val))
    return tree


@dataclass
class TuningResult:
    best: TreeModel
    trials: list[dict[str, float]] = field(default_factory=list)


def grid_candidates(
    base: Mapping[str, float], grid: Mapping[str, Sequence[float]]
) -> list[dict[str, float]]:
    """Every combination of the grid values, on top of the base parameters."""
    keys = sorted(grid)
    return [
        {**base, **dict(zip(keys, values, strict=True))}
        for values in itertools.product(*(grid[k] for k in keys))
    ]


def tune_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Sequence[str],
    base_params: Mapping[str, float],
    grid: Mapping[str, Sequence[float]],
    seed: int = 42,
    on_trial: Callable[[int, dict[str, float]], None] | None = None,
) -> TuningResult:
    """Grid search scored by validation PR-AUC; returns the best model."""
    best: TreeModel | None = None
    trials: list[dict[str, float]] = []
    for number, params in enumerate(grid_candidates(base_params, grid), start=1):
        model = train_xgboost(x_train, y_train, x_val, y_val, feature_names, params, seed)
        trial = {
            **{k: params[k] for k in sorted(grid)},
            "best_iteration": model.best_iteration,
            "val_pr_auc": round(model.val_pr_auc, 4),
        }
        trials.append(trial)
        logger.info("trial %d: %s", number, trial)
        if on_trial:
            on_trial(number, trial)
        if best is None or model.val_pr_auc > best.val_pr_auc:
            best = model
    if best is None:
        raise ValueError("The tuning grid is empty")
    return TuningResult(best=best, trials=trials)


class ProbabilityCalibrator:
    """Maps raw model scores to calibrated probabilities.

    ``isotonic`` fits a monotone step function and ``sigmoid`` fits Platt
    scaling on the log-odds. Both are fitted on the validation split.
    """

    def __init__(self, method: str = "isotonic") -> None:
        if method not in {"isotonic", "sigmoid"}:
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self._model: Any = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> ProbabilityCalibrator:
        scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self._model.fit(scores, labels)
        else:
            from sklearn.linear_model import LogisticRegression

            self._model = LogisticRegression()
            self._model.fit(_logit(scores).reshape(-1, 1), labels)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Calibrator is not fitted")
        scores = np.asarray(scores, dtype=float)
        if self.method == "isotonic":
            return np.clip(self._model.predict(scores), 0.0, 1.0)
        return self._model.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))
