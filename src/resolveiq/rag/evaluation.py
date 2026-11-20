"""Evaluation of the RAG assistant on a frozen, hand-written test set.

Two metrics are computed automatically for every run:

* source hit rate: share of answerable questions whose expected source file
  is among the returned sources
* refusal behaviour: share of out-of-scope questions refused, and share of
  answerable questions wrongly refused

Every full run also checks citations: the share of sentences that cite a source
and whether each citation points at a retrieved source.

Answer correctness is judged either by a person (fill the ``correct`` column
with 1 or 0 and run ``eval/run_rag_eval.py score``) or by a language model with
``run --judge``, which also records whether each claim is supported by the
passages and the judge's reason.
"""

from __future__ import annotations

import csv
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from resolveiq.rag.generate import AnswerResult
from resolveiq.rag.grading import Judgement, check_grounding

TESTSET_COLUMNS = (
    "question_id",
    "split",
    "in_scope",
    "question",
    "expected_answer",
    "source_file",
    "source_page",
)
RESULT_COLUMNS = (
    "question_id",
    "split",
    "in_scope",
    "question",
    "expected_answer",
    "expected_source",
    "answer",
    "refused",
    "sources",
    "best_similarity",
    "source_hit",
    "refusal_correct",
    "latency_ms",
    "correct",
    "citation_coverage",
    "citations_valid",
    "supported",
    "judge_reason",
)
SPLITS = ("dev", "holdout")


@dataclass(frozen=True)
class EvalCase:
    question_id: str
    split: str
    in_scope: bool
    question: str
    expected_answer: str
    source_file: str
    source_page: int | None


def load_testset(path: Path) -> list[EvalCase]:
    """Read and validate the test set CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(TESTSET_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Test set is missing columns: {sorted(missing)}")
        cases = [
            EvalCase(
                question_id=row["question_id"].strip(),
                split=row["split"].strip(),
                in_scope=row["in_scope"].strip() == "1",
                question=row["question"].strip(),
                expected_answer=row["expected_answer"].strip(),
                source_file=row["source_file"].strip(),
                source_page=int(row["source_page"]) if row["source_page"].strip() else None,
            )
            for row in reader
        ]

    ids = [case.question_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Test set contains duplicate question ids")
    for case in cases:
        if case.split not in SPLITS:
            raise ValueError(f"{case.question_id}: unknown split '{case.split}'")
        if case.in_scope and not case.source_file:
            raise ValueError(f"{case.question_id}: in-scope questions need a source_file")
    return cases


def select_split(cases: Sequence[EvalCase], split: str) -> list[EvalCase]:
    if split == "all":
        return list(cases)
    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}'")
    return [case for case in cases if case.split == split]


def evaluate_case(
    case: EvalCase, result: AnswerResult, judgement: Judgement | None = None
) -> dict[str, object]:
    """Turn one answer into a results row with the automatic checks filled in."""
    returned_files = [source.file for source in result.sources]
    source_hit: object = int(case.source_file in returned_files) if case.in_scope else ""
    # Refusing an answerable question is always wrong, so it is scored here;
    # answered in-scope questions are left blank for a human to judge.
    correct: object = 0 if case.in_scope and result.refused else ""
    coverage: object = ""
    valid: object = ""
    if result.answer and not result.refused:
        grounding = check_grounding(result.answer, [(s.file, s.page) for s in result.sources])
        coverage, valid = grounding.coverage, int(grounding.all_valid)
    supported: object = ""
    reason = ""
    if judgement is not None:
        if judgement.correct is not None and correct == "":
            correct = int(judgement.correct)
        if judgement.supported is not None:
            supported = int(judgement.supported)
        reason = judgement.reason
    return {
        "question_id": case.question_id,
        "split": case.split,
        "in_scope": int(case.in_scope),
        "question": case.question,
        "expected_answer": case.expected_answer,
        "expected_source": case.source_file,
        "answer": result.answer,
        "refused": int(result.refused),
        "sources": ";".join(f"{s.file}:{s.page}" for s in result.sources),
        "best_similarity": round(float(result.settings.get("best_similarity", 0.0)), 4),
        "source_hit": source_hit,
        "refusal_correct": int(result.refused != case.in_scope),
        "latency_ms": round(
            sum(result.timings_ms.get(k, 0.0) for k in ("retrieval", "generation")), 1
        ),
        "correct": correct,
        "citation_coverage": coverage,
        "citations_valid": valid,
        "supported": supported,
        "judge_reason": reason,
    }


def write_results(rows: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _flag(value: object) -> int | None:
    text = str(value).strip()
    if text in {"0", "1"}:
        return int(text)
    if text == "":
        return None
    raise ValueError(f"Expected 0, 1 or empty, got '{text}'")


def _rate(values: Iterable[int | None]) -> float | None:
    known = [v for v in values if v is not None]
    return round(sum(known) / len(known), 4) if known else None


def summarize(rows: Sequence[dict[str, object]]) -> dict[str, float | int | None]:
    """Aggregate a results table into the headline metrics."""
    in_scope = [r for r in rows if _flag(r["in_scope"]) == 1]
    out_of_scope = [r for r in rows if _flag(r["in_scope"]) == 0]
    latencies = sorted(float(r["latency_ms"]) for r in rows if str(r["latency_ms"]).strip())
    labeled = [_flag(r["correct"]) for r in in_scope]
    answered = [r for r in in_scope if _flag(r["refused"]) == 0]
    coverages = [
        float(r["citation_coverage"])
        for r in answered
        if str(r.get("citation_coverage", "")).strip()
    ]

    def percentile(p: float) -> float | None:
        if not latencies:
            return None
        index = min(len(latencies) - 1, round(p * (len(latencies) - 1)))
        return latencies[index]

    return {
        "questions": len(rows),
        "in_scope_questions": len(in_scope),
        "out_of_scope_questions": len(out_of_scope),
        "answer_correct_rate": _rate(labeled),
        "answers_labeled": sum(v is not None for v in labeled),
        "source_hit_rate": _rate(_flag(r["source_hit"]) for r in in_scope),
        "out_of_scope_refusal_rate": _rate(_flag(r["refused"]) for r in out_of_scope),
        "false_refusal_rate": _rate(_flag(r["refused"]) for r in in_scope),
        "citation_coverage": round(statistics.mean(coverages), 4) if coverages else None,
        "citation_validity_rate": _rate(_flag(r.get("citations_valid", "")) for r in answered),
        "supported_rate": _rate(_flag(r.get("supported", "")) for r in answered),
        "latency_median_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": percentile(0.95),
    }


def format_metrics(metrics: dict[str, float | int | None]) -> str:
    """Render metrics as an aligned two-column table."""
    width = max(len(name) for name in metrics)
    lines = [f"{'metric'.ljust(width)}  value", f"{'-' * width}  -----"]
    for name, value in metrics.items():
        if value is None:
            shown = "n/a"
        elif name.endswith("_rate"):
            shown = f"{value * 100:.1f}%"
        else:
            shown = str(value)
        lines.append(f"{name.ljust(width)}  {shown}")
    return "\n".join(lines)
