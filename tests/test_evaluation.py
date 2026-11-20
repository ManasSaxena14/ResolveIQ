from pathlib import Path

import pytest

from resolveiq.config import PROJECT_ROOT, load_config
from resolveiq.rag.evaluation import (
    EvalCase,
    evaluate_case,
    format_metrics,
    load_testset,
    read_results,
    select_split,
    summarize,
    write_results,
)
from resolveiq.rag.generate import AnswerResult, Source

TESTSET = PROJECT_ROOT / "eval" / "rag_testset.csv"


def _case(in_scope: bool, source: str = "returns/return-policy.md") -> EvalCase:
    return EvalCase("Q1", "dev", in_scope, "question", "answer", source if in_scope else "", None)


def _result(refused: bool, files: tuple[str, ...] = (), latency: float = 100.0) -> AnswerResult:
    return AnswerResult(
        answer="" if refused else "text",
        refused=refused,
        sources=[Source(f, 1, 0.5) for f in files],
        timings_ms={"retrieval": latency / 2, "generation": latency / 2},
        settings={"best_similarity": 0.61234},
    )


class TestRepositoryTestSet:
    """Checks on the committed test set itself, so it cannot silently drift."""

    def test_has_forty_questions_with_expected_balance(self):
        cases = load_testset(TESTSET)

        assert len(cases) == 40
        assert sum(c.in_scope for c in cases) == 30
        assert sum(not c.in_scope for c in cases) == 10
        assert len(select_split(cases, "holdout")) == 10

    def test_every_expected_source_exists(self):
        kb_dir = load_config().resolve(Path("data/kb"))

        for case in load_testset(TESTSET):
            if case.in_scope:
                assert (kb_dir / case.source_file).is_file(), case.question_id


def test_testset_validation_rejects_bad_rows(tmp_path):
    path = tmp_path / "bad.csv"
    header = "question_id,split,in_scope,question,expected_answer,source_file,source_page\n"
    path.write_text(header + "Q1,dev,1,Q?,A,,\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source_file"):
        load_testset(path)

    path.write_text(header + "Q1,train,0,Q?,,,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split"):
        load_testset(path)

    path.write_text(header + "Q1,dev,0,Q?,,,\nQ1,dev,0,Q?,,,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_testset(path)

    path.write_text("question_id,question\nQ1,Q?\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_testset(path)


def test_select_split_rejects_unknown_names():
    with pytest.raises(ValueError):
        select_split([], "train")


def test_answered_in_scope_question_waits_for_a_human_label():
    row = evaluate_case(_case(True), _result(False, ("returns/return-policy.md",)))

    assert row["source_hit"] == 1
    assert row["refusal_correct"] == 1
    assert row["correct"] == ""
    assert row["latency_ms"] == 100.0
    assert row["best_similarity"] == 0.6123
    assert row["sources"] == "returns/return-policy.md:1"


def test_refused_in_scope_question_is_scored_wrong():
    row = evaluate_case(_case(True), _result(True))

    assert row["source_hit"] == 0
    assert row["refusal_correct"] == 0
    assert row["correct"] == 0


def test_out_of_scope_refusal_is_correct():
    row = evaluate_case(_case(False), _result(True))

    assert row["source_hit"] == ""
    assert row["refusal_correct"] == 1


def test_summary_metrics(tmp_path):
    rows = [
        evaluate_case(_case(True), _result(False, ("returns/return-policy.md",), 100)),
        evaluate_case(_case(True), _result(False, ("other.md",), 200)),
        evaluate_case(_case(True), _result(True, (), 300)),
        evaluate_case(_case(False), _result(True, (), 400)),
        evaluate_case(_case(False), _result(False, ("other.md",), 500)),
    ]
    rows[0]["correct"] = 1
    rows[1]["correct"] = 0
    path = tmp_path / "run.csv"
    write_results(rows, path)

    metrics = summarize(read_results(path))

    assert metrics["questions"] == 5
    assert metrics["answers_labeled"] == 3
    assert metrics["answer_correct_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert metrics["source_hit_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert metrics["false_refusal_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert metrics["out_of_scope_refusal_rate"] == 0.5
    assert metrics["latency_median_ms"] == 300.0
    assert metrics["latency_p95_ms"] == 500.0


def test_summary_without_labels_reports_none():
    rows = [evaluate_case(_case(True), _result(False, ("returns/return-policy.md",)))]

    metrics = summarize(rows)

    assert metrics["answer_correct_rate"] is None
    assert "n/a" in format_metrics(metrics)


def test_invalid_label_raises():
    row = evaluate_case(_case(True), _result(False))
    row["correct"] = "yes"

    with pytest.raises(ValueError, match="Expected 0, 1"):
        summarize([row])


def test_format_metrics_shows_rates_as_percentages():
    text = format_metrics({"source_hit_rate": 0.875, "questions": 40, "latency_p95_ms": None})

    assert "87.5%" in text
    assert "40" in text
    assert "n/a" in text


def _cited(answer: str, files: tuple[str, ...]) -> AnswerResult:
    return AnswerResult(
        answer=answer,
        refused=False,
        sources=[Source(f, 1, 0.7) for f in files],
        timings_ms={"retrieval": 10.0, "generation": 900.0},
        settings={"best_similarity": 0.7},
    )


def test_judgement_fills_correctness_support_and_citations():
    from resolveiq.rag.grading import Judgement

    result = _cited(
        "You have 30 days [return-policy.md, page 1]. Shipping is free [ship.md, page 1].",
        ("returns/return-policy.md",),
    )

    row = evaluate_case(_case(True), result, Judgement(True, False, "Adds free shipping."))

    assert row["correct"] == 1
    assert row["supported"] == 0
    assert row["judge_reason"] == "Adds free shipping."
    assert row["citation_coverage"] == 1.0
    assert row["citations_valid"] == 0


def test_judge_never_overrides_a_wrong_refusal():
    from resolveiq.rag.grading import Judgement

    row = evaluate_case(_case(True), _result(True), Judgement(True, True, "x"))

    assert row["correct"] == 0
    assert row["citation_coverage"] == ""


def test_summary_reports_citation_and_support_rates(tmp_path):
    from resolveiq.rag.grading import Judgement

    rows = [
        evaluate_case(
            _case(True),
            _cited("Thirty days [return-policy.md, page 1].", ("returns/return-policy.md",)),
            Judgement(True, True, ""),
        ),
        evaluate_case(
            _case(True),
            _cited("Thirty days. No citation here.", ("returns/return-policy.md",)),
            Judgement(False, False, ""),
        ),
    ]
    path = tmp_path / "judged.csv"
    write_results(rows, path)

    metrics = summarize(read_results(path))

    assert metrics["answer_correct_rate"] == 0.5
    assert metrics["supported_rate"] == 0.5
    assert metrics["citation_coverage"] == 0.5
    assert metrics["citation_validity_rate"] == 1.0
