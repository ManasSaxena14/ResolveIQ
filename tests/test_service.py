import pytest

from resolveiq.monitoring.logging import RequestLog
from resolveiq.schemas import AnalyzeRequest
from resolveiq.service import InputError, _classifier_version
from service_doubles import StubLLM, make_copilot


def test_analyze_logs_numbers_but_never_the_text(embedder, vector_store, caplog):
    log = RequestLog(":memory:")
    copilot = make_copilot(embedder, vector_store, log=log)
    text = "Charged twice, my email is jane@example.com"

    with caplog.at_level("INFO", logger="resolveiq.requests"):
        result = copilot.analyze(AnalyzeRequest(text=text, channel="Email"))

    row = log.recent(1)[0]
    assert row["request_id"] == result.response.request_id
    assert row["channel"] == "email"
    assert row["text_length"] == len(text)
    assert row["category"] == "Billing and Payments"
    assert row["draft_status"] == "ok"
    assert "jane@example.com" not in caplog.text
    assert "Charged twice" not in caplog.text
    assert result.partial is False


def test_product_is_added_to_the_question(embedder, vector_store):
    llm = StubLLM()
    copilot = make_copilot(embedder, vector_store, llm=llm)
    request = AnalyzeRequest(text="Why was I charged twice?", subject="Billing", product="Pro")

    assert copilot._question(request) == "Billing\n\nWhy was I charged twice?\n\nProduct: Pro"


def test_question_is_cut_to_the_engine_limit(embedder, vector_store):
    copilot = make_copilot(embedder, vector_store, max_text_chars=10_000)

    question = copilot._question(AnalyzeRequest(text="a" * 5000, subject="s"))

    assert len(question) == 4000


def test_validate_rejects_long_text(embedder, vector_store):
    copilot = make_copilot(embedder, vector_store, max_text_chars=10)
    with pytest.raises(InputError, match="longer than 10"):
        copilot.analyze(AnalyzeRequest(text="x" * 11))


def test_feedback_without_a_log_is_not_stored(embedder, vector_store):
    copilot = make_copilot(embedder, vector_store)
    copilot.request_log = None

    assert copilot.feedback("abc", "up") is False
    assert copilot.health().status == "degraded"


def test_classifier_version_from_meta(tmp_path):
    (tmp_path / "classifier_meta.json").write_text(
        '{"train_config": {"model_name": "distilbert-base-uncased", "seed": 42}}'
    )

    assert _classifier_version(tmp_path, "local") == "distilbert-base-uncased-seed-42"
    assert _classifier_version(tmp_path, "clf/v2").endswith("(clf/v2)")
    assert _classifier_version(tmp_path / "missing", "local") == "unknown"
