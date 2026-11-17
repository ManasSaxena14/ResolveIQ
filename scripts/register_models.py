"""Register the production classifier and risk model in the MLflow model registry.

Each model folder is logged as an MLflow model, registered under the names in
``service.registry`` and given the ``production`` alias, which the API loads.

Usage:
    python scripts/register_models.py [--only classifier|risk]
"""

from __future__ import annotations

import argparse
import json
import sys

from resolveiq.config import load_config
from resolveiq.registry import register_model
from resolveiq.tracking import git_commit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to a config file")
    parser.add_argument("--only", choices=["classifier", "risk"])
    args = parser.parse_args()

    settings = load_config(args.config)
    if not settings.tracking.enabled:
        print("tracking.enabled is false; the registry needs MLflow.", file=sys.stderr)
        return 1
    service = settings.service
    jobs = {
        "classifier": (settings.resolve(service.classifier_dir), "classifier"),
        "risk": (settings.resolve(service.risk_dir), "escalation-risk"),
    }
    for kind, (model_dir, experiment) in jobs.items():
        if args.only and kind != args.only:
            continue
        meta_name = "classifier_meta.json" if kind == "classifier" else "risk_meta.json"
        if not (model_dir / meta_name).is_file():
            print(f"{model_dir} has no trained {kind}; train it first.", file=sys.stderr)
            return 1
        meta = json.loads((model_dir / meta_name).read_text(encoding="utf-8"))
        tags = {"git_commit": git_commit(), "source_dir": str(model_dir.name)}
        if kind == "classifier":
            tags["val_macro_f1"] = str(meta.get("best_val_macro_f1", ""))
        else:
            tags["model_version"] = str(meta.get("version", ""))
        version = register_model(settings, kind, model_dir, experiment, tags)
        print(f"Registered {kind} as version {version} with alias {service.registry.alias}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
