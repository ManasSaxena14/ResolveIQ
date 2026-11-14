"""Run and score the RAG evaluation.

Commands:
    run    Answer every question in a split, write a results CSV and log the run.
    score  Compute answer correctness from a results CSV after hand labeling.

Examples:
    python eval/run_rag_eval.py run --split dev --retrieval-only
    python eval/run_rag_eval.py run --split dev --run-name hybrid-top5
    python eval/run_rag_eval.py score results/rag/hybrid-top5.csv

The holdout split is reserved for the final run. Tune settings on dev only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from resolveiq.config import PROJECT_ROOT, Settings, load_config
from resolveiq.rag.evaluation import (
    evaluate_case,
    format_metrics,
    load_testset,
    read_results,
    select_split,
    summarize,
    write_results,
)
from resolveiq.rag.generate import AnswerEngine
from resolveiq.rag.llm import LLMError
from resolveiq.tracking import tracked_run

TESTSET_PATH = PROJECT_ROOT / "eval" / "rag_testset.csv"


def _log_metrics(mlflow, metrics: dict) -> None:
    mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})


def run(settings: Settings, args: argparse.Namespace) -> int:
    cases = select_split(load_testset(TESTSET_PATH), args.split)
    if args.split in {"holdout", "all"}:
        print("Note: the holdout split is reserved for the final evaluation run.")

    mode = "retrieval" if args.retrieval_only else "full"
    run_name = args.run_name or f"{mode}-{args.split}-{datetime.now():%Y%m%d-%H%M%S}"
    output = settings.resolve(settings.paths.results_dir) / "rag" / f"{run_name}.csv"

    engine = AnswerEngine.from_settings(settings, with_llm=not args.retrieval_only)
    rows = []
    for number, case in enumerate(cases, start=1):
        try:
            if args.retrieval_only:
                result = engine.retrieve_only(case.question)
            else:
                result = engine.answer(case.question)
        except LLMError as exc:
            print(f"Stopped at {case.question_id}: {exc}", file=sys.stderr)
            return 1
        rows.append(evaluate_case(case, result))
        print(f"[{number}/{len(cases)}] {case.question_id} refused={int(result.refused)}")

    write_results(rows, output)
    metrics = summarize(rows)

    tags = {"stage": "evaluation", "split": args.split, "mode": mode}
    with tracked_run(settings, experiment="rag", run_name=run_name, tags=tags) as mlflow:
        if mlflow:
            mlflow.log_params({k: v for k, v in engine.settings_snapshot.items() if v is not None})
            _log_metrics(mlflow, metrics)
            mlflow.log_artifact(str(output))

    print()
    print(format_metrics(metrics))
    print(f"\nResults written to {output}")
    if not args.retrieval_only:
        print("Label the 'correct' column with 1 or 0, then run the score command.")
    return 0


def score(settings: Settings, args: argparse.Namespace) -> int:
    path = Path(args.results)
    metrics = summarize(read_results(path))
    unlabeled = metrics["in_scope_questions"] - metrics["answers_labeled"]

    with tracked_run(settings, experiment="rag", run_name=f"score-{path.stem}") as mlflow:
        if mlflow:
            mlflow.set_tag("stage", "scoring")
            _log_metrics(mlflow, metrics)
            mlflow.log_artifact(str(path))

    print(format_metrics(metrics))
    if unlabeled:
        print(f"\nWarning: {unlabeled} answerable questions are not labeled yet.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", help="Path to a config file")
    parser.add_argument("--no-mlflow", action="store_true", help="Do not log to MLflow")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="Answer the test questions")
    run_parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    run_parser.add_argument("--retrieval-only", action="store_true", help="Skip generation")
    run_parser.add_argument("--run-name", help="Name for the results file and MLflow run")

    score_parser = commands.add_parser("score", help="Score a hand-labeled results file")
    score_parser.add_argument("results", help="Path to a results CSV")

    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = load_config(args.config)
    if args.no_mlflow:
        settings.tracking.enabled = False
    return run(settings, args) if args.command == "run" else score(settings, args)


if __name__ == "__main__":
    sys.exit(main())
