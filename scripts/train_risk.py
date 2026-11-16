"""Train the escalation risk models and choose the alert threshold.

Trains a logistic regression baseline, XGBoost with the configured starting
parameters, and XGBoost tuned over the configured grid. The tuned model is
calibrated on the validation split, and the alert threshold is chosen on the
validation split from the configured cost ratio. Every model and every tuning
trial is logged to MLflow. Test-set scores come from scripts/evaluate_risk.py.

Usage:
    python scripts/train_risk.py [--config PATH] [--no-mlflow]
"""

from __future__ import annotations

import argparse
import time

import joblib
import numpy as np

from resolveiq.config import load_config
from resolveiq.risk.evaluate import alert_stats, binary_metrics, choose_threshold
from resolveiq.risk.features import load_feature_schema, load_feature_split
from resolveiq.risk.predict import QUEUE_MODEL_FILENAME, TEXT_MODEL_FILENAME, save_risk_model
from resolveiq.risk.train import (
    ProbabilityCalibrator,
    build_logistic_baseline,
    train_xgboost,
    tune_xgboost,
)
from resolveiq.tracking import git_commit, tracked_run

LOGISTIC_FILENAME = "logistic_baseline.joblib"
DEFAULT_XGBOOST_FILENAME = "xgboost_default.json"


def _log(mlflow, params: dict, metrics: dict, prefix: str = "val_") -> None:
    if mlflow:
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_metrics({f"{prefix}{k}": v for k, v in metrics.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to a config file")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    settings = load_config(args.config)
    if args.no_mlflow:
        settings.tracking.enabled = False
    cfg = settings.risk
    features_dir = settings.resolve(cfg.features_dir)
    out_dir = settings.resolve(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = load_feature_schema(features_dir)
    x_train, y_train, _ = load_feature_split(features_dir, "train")
    x_val, y_val, _ = load_feature_split(features_dir, "validation")
    target = cfg.precision_target
    summary: dict[str, dict[str, float]] = {}

    prior = float(y_train.mean())
    summary["majority"] = binary_metrics(y_val, np.full(len(y_val), prior), target)

    with tracked_run(settings, "escalation-risk", run_name="logistic-regression") as mlflow:
        started = time.perf_counter()
        logistic = build_logistic_baseline(cfg.seed).fit(x_train, y_train)
        seconds = round(time.perf_counter() - started, 1)
        summary["logistic_regression"] = binary_metrics(
            y_val, logistic.predict_proba(x_val)[:, 1], target
        )
        joblib.dump(logistic, out_dir / LOGISTIC_FILENAME)
        _log(
            mlflow,
            {"model": "logistic-regression", "scaling": "standard", "class_weight": "balanced"},
            {**summary["logistic_regression"], "train_seconds": seconds},
        )

    base_params = cfg.xgboost.model_dump()
    with tracked_run(settings, "escalation-risk", run_name="xgboost") as mlflow:
        default_tree = train_xgboost(
            x_train, y_train, x_val, y_val, schema.names, base_params, cfg.seed
        )
        default_tree.save(out_dir / DEFAULT_XGBOOST_FILENAME)
        summary["xgboost"] = binary_metrics(y_val, default_tree.predict(x_val), target)
        _log(
            mlflow,
            {"model": "xgboost", **base_params, "best_iteration": default_tree.best_iteration},
            summary["xgboost"],
        )

    with tracked_run(settings, "escalation-risk", run_name="xgboost-tuned") as mlflow:
        started = time.perf_counter()

        def log_trial(number: int, trial: dict[str, float]) -> None:
            if mlflow:
                with mlflow.start_run(run_name=f"trial-{number:02d}", nested=True):
                    mlflow.log_params({k: v for k, v in trial.items() if k != "val_pr_auc"})
                    mlflow.log_metric("val_pr_auc", trial["val_pr_auc"])

        tuning = tune_xgboost(
            x_train,
            y_train,
            x_val,
            y_val,
            schema.names,
            base_params,
            cfg.tuning_grid,
            cfg.seed,
            on_trial=log_trial,
        )
        tree = tuning.best
        raw_val = tree.predict(x_val)
        summary["xgboost_tuned"] = binary_metrics(y_val, raw_val, target)

        calibrator = ProbabilityCalibrator(cfg.calibration).fit(raw_val, y_val)
        calibrated_val = calibrator.transform(raw_val)
        summary["xgboost_tuned_calibrated"] = binary_metrics(y_val, calibrated_val, target)

        thresholds = {
            str(ratio): choose_threshold(y_val, calibrated_val, miss_cost=ratio)
            for ratio in cfg.cost_ratios
        }
        threshold = choose_threshold(y_val, calibrated_val, miss_cost=cfg.alert_cost_ratio)
        stats = alert_stats(y_val, calibrated_val, threshold)
        seconds = round(time.perf_counter() - started, 1)
        tuned_params = {k: tree.params[k] for k in sorted(cfg.tuning_grid)}
        _log(
            mlflow,
            {
                "model": "xgboost-tuned",
                **tuned_params,
                "best_iteration": tree.best_iteration,
                "trials": len(tuning.trials),
                "calibration": cfg.calibration,
                "alert_cost_ratio": cfg.alert_cost_ratio,
                "alert_threshold": threshold,
            },
            {
                **summary["xgboost_tuned_calibrated"],
                "alert_caught": stats.caught,
                "alert_rate": stats.alert_rate,
                "tuning_seconds": seconds,
            },
        )
        meta = {
            "version": f"xgboost-{git_commit()}",
            "label": {
                "source_column": cfg.label.source_column,
                "positive_values": cfg.label.positive_values,
                "is_proxy": True,
            },
            "schema": schema.as_dict(),
            "params": tree.params,
            "best_iteration": tree.best_iteration,
            "calibration": cfg.calibration,
            "alert_cost_ratio": cfg.alert_cost_ratio,
            "threshold": threshold,
            "thresholds_by_cost_ratio": thresholds,
            "train_positive_rate": round(prior, 4),
            "validation": summary,
            "tuning_trials": tuning.trials,
        }
        text_path = out_dir / TEXT_MODEL_FILENAME
        save_risk_model(
            out_dir,
            tree,
            calibrator,
            joblib.load(out_dir / QUEUE_MODEL_FILENAME),
            joblib.load(text_path) if schema.text_score else None,
            meta,
        )
        if mlflow:
            mlflow.log_artifacts(str(out_dir), artifact_path="risk_model")

    print(f"Escalation rate in training data: {prior:.3f}")
    print(
        f"{'Model (validation)':<28} {'PR-AUC':>7} {'R@P' + str(int(target * 100)):>7} "
        f"{'Brier':>7} {'ROC-AUC':>8}"
    )
    for name, metrics in summary.items():
        print(
            f"{name:<28} {metrics['pr_auc']:>7.4f} {metrics['recall_at_precision']:>7.4f} "
            f"{metrics['brier']:>7.4f} {metrics['roc_auc']:>8.4f}"
        )
    print(f"Best grid parameters: {tuned_params}, {tree.best_iteration + 1} trees")
    print(
        f"Alert threshold {threshold:.2f} (cost ratio {cfg.alert_cost_ratio:g}:1) catches "
        f"{stats.caught:.1%} of validation escalations with {stats.alert_rate:.1%} of tickets "
        f"alerted"
    )
    print(f"Model saved to {out_dir}")


if __name__ == "__main__":
    main()
