"""Contract tests: every endpoint returns the documented JSON shape and errors."""

import pytest
from fastapi.testclient import TestClient

from resolveiq.api.main import REQUEST_ID_HEADER, create_app
from resolveiq.rag.prompts import REFUSAL_MESSAGE
from resolveiq.schemas import AnalyzeResponse, BatchResponse, HealthResponse
from service_doubles import BrokenLLM, StubLLM, make_copilot

TICKET = {
    "text": "I was charged twice this month and nobody has replied to my last two emails.",
    "channel": "email",
    "product": "Pro plan",
}


@pytest.fixture
def copilot(embedder, vector_store):
    return make_copilot(embedder, vector_store, max_upload_bytes=2000, max_batch_rows=5)


@pytest.fixture
def client(copilot):
    with TestClient(create_app(copilot)) as test_client:
        yield test_client


def test_analyze_matches_the_schema(client):
    response = client.post("/analyze", json=TICKET)

    assert response.status_code == 200
    body = AnalyzeResponse.model_validate(response.json())
    assert body.draft_reply.status == "ok"
    assert body.draft_reply.sources[0].file == "billing/double-charges.md"
    assert body.category.label == "Billing and Payments"
    assert len(body.category.top) == 3
    assert body.risk.alert is True
    assert body.risk.reasons == ['Message mentions "twice"']
    assert {"retrieval", "generation", "classifier", "risk", "total"} <= set(body.timings_ms)
    assert set(body.versions) == {"embedder", "llm", "classifier", "risk"}
    assert response.headers[REQUEST_ID_HEADER]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"text": ""}, "empty"),
        ({"text": "   "}, "empty"),
        ({"text": "x" * 4001}, "longer than 4000"),
    ],
)
def test_analyze_rejects_bad_text_with_422(client, payload, message):
    response = client.post("/analyze", json=payload)

    assert response.status_code == 422
    assert message in str(response.json()["detail"])


def test_unknown_fields_and_ticket_types_are_rejected(client):
    assert client.post("/analyze", json={**TICKET, "priority": "high"}).status_code == 422
    assert client.post("/analyze", json={**TICKET, "ticket_type": "Urgent"}).status_code == 422


def test_llm_failure_returns_503_with_partial_result(embedder, vector_store):
    copilot = make_copilot(embedder, vector_store, llm=BrokenLLM())
    with TestClient(create_app(copilot)) as client:
        response = client.post("/analyze", json=TICKET)

    assert response.status_code == 503
    body = AnalyzeResponse.model_validate(response.json())
    assert body.draft_reply.status == "unavailable"
    assert body.draft_reply.text is None
    assert body.draft_reply.sources, "retrieved sources are still shown"
    assert body.category.label == "Billing and Payments"
    assert body.risk.escalation_probability > 0


def test_out_of_scope_question_is_refused_without_the_llm(embedder, vector_store):
    llm = StubLLM()
    copilot = make_copilot(embedder, vector_store, llm=llm)
    with TestClient(create_app(copilot)) as client:
        body = client.post("/analyze", json={"text": "Who won the football world cup?"}).json()

    assert body["draft_reply"]["status"] == "refused"
    assert body["draft_reply"]["text"] == REFUSAL_MESSAGE
    assert body["draft_reply"]["sources"] == []
    assert llm.calls == 0


def test_feedback_round_trip(client):
    request_id = client.post("/analyze", json=TICKET).json()["request_id"]

    ok = client.post("/feedback", json={"request_id": request_id, "rating": "up"})
    missing = client.post("/feedback", json={"request_id": "nope", "rating": "down"})
    invalid = client.post("/feedback", json={"request_id": request_id, "rating": "meh"})

    assert ok.status_code == 200
    assert ok.json() == {"request_id": request_id, "rating": "up", "stored": True}
    assert missing.status_code == 404
    assert invalid.status_code == 422


def test_batch_ranks_by_risk(client):
    csv_text = "id,subject,body\nA,Invoice,Question about an invoice\nB,URGENT,Server down again\n"

    response = client.post("/batch", files={"file": ("t.csv", csv_text, "text/csv")})

    assert response.status_code == 200
    body = BatchResponse.model_validate(response.json())
    assert [item.id for item in body.items] == ["B", "A"]
    assert body.items[0].rank == 1
    assert body.count == 2
    assert body.alerts == 1
    assert "body" not in response.text.lower().replace('"body"', "")


def test_batch_accepts_a_text_column_and_numbers_rows(client):
    response = client.post("/batch", files={"file": ("t.csv", "text\nhello\n", "text/csv")})

    assert response.json()["items"][0]["id"] == "1"


@pytest.mark.parametrize(
    ("content", "status", "message"),
    [
        ("id,message\n1,hello\n", 422, "'body' or 'text'"),
        ("body\n", 422, "no rows"),
        ("body\n" + "a\n" * 6, 422, "more than 5 rows"),
        ("body\nhello\n\n,\n", 422, "empty text"),
        ("body\n" + "x" * 3000 + "\n", 413, "larger than"),
    ],
)
def test_batch_errors(client, content, status, message):
    response = client.post("/batch", files={"file": ("t.csv", content, "text/csv")})

    assert response.status_code == status
    assert message in response.json()["detail"]


def test_batch_rejects_non_utf8(client):
    response = client.post("/batch", files={"file": ("t.csv", b"body\n\xff\xfe\n", "text/csv")})

    assert response.status_code == 422


def test_health_reports_components(client):
    body = HealthResponse.model_validate(client.get("/health").json())

    assert body.status == "ok"
    assert body.index == {"files": 3, "chunks": 3}
    assert all(body.components.values())


def test_health_is_degraded_without_the_knowledge_base(embedder, vector_store):
    copilot = make_copilot(embedder, vector_store, with_engine=False)
    with TestClient(create_app(copilot)) as client:
        health = client.get("/health").json()
        analyze = client.post("/analyze", json=TICKET)

    assert health["status"] == "degraded"
    assert health["components"]["retriever"] is False
    assert analyze.status_code == 503
    assert "not configured" in analyze.json()["draft_reply"]["error"]


def test_models_are_loaded_at_startup_when_not_given(embedder, vector_store):
    calls = []

    def loader():
        calls.append(1)
        return make_copilot(embedder, vector_store)

    with TestClient(create_app(loader=loader)) as client:
        assert client.get("/health").status_code == 200
    assert calls == [1]


def test_history_is_newest_first_with_feedback_and_no_text(client):
    first = client.post("/analyze", json=TICKET).json()["request_id"]
    second = client.post("/analyze", json={"text": "Server down again, urgent"}).json()
    client.post("/feedback", json={"request_id": first, "rating": "down"})

    rows = client.get("/requests", params={"limit": 10}).json()

    assert [row["request_id"] for row in rows] == [second["request_id"], first]
    assert rows[1]["rating"] == "down"
    assert rows[0]["rating"] is None
    assert "text" not in rows[0] and "text_hash" not in rows[0]
    assert client.get("/requests", params={"limit": 0}).status_code == 422


def test_stats_summarize_the_log(client):
    client.post("/analyze", json=TICKET)
    request_id = client.post("/analyze", json={"text": "Question about my invoice"}).json()[
        "request_id"
    ]
    client.post("/feedback", json={"request_id": request_id, "rating": "up"})

    stats = client.get("/stats").json()

    assert stats["requests"] == 2
    assert stats["alerts"] == 1
    assert stats["categories"] == {"Billing and Payments": 2}
    assert stats["draft_status"]["ok"] >= 1
    assert stats["feedback"]["up"] == 1
    assert stats["latency_median_ms"] is not None


def test_history_and_stats_without_a_log(embedder, vector_store):
    copilot = make_copilot(embedder, vector_store)
    copilot.request_log = None
    with TestClient(create_app(copilot)) as client:
        assert client.get("/requests").json() == []
        assert client.get("/stats").json()["requests"] == 0


def test_built_frontend_is_served_next_to_the_api(copilot, tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>ResolveIQ</title>")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app.js").write_text("console.log(1)")

    with TestClient(create_app(copilot, web_dir=tmp_path)) as client:
        assert "ResolveIQ" in client.get("/").text
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/health").json()["status"] == "ok"


def test_no_frontend_means_no_root_page(copilot, tmp_path):
    with TestClient(create_app(copilot, web_dir=tmp_path)) as client:
        assert client.get("/").status_code == 404


def _events(text: str) -> list[tuple[str, dict]]:
    import json

    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_stream_sends_analysis_first_then_the_draft(embedder, vector_store):
    from service_doubles import ChunkedLLM

    copilot = make_copilot(embedder, vector_store, llm=ChunkedLLM())
    with TestClient(create_app(copilot)) as client:
        response = client.post("/analyze/stream", json=TICKET)

    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    names = [name for name, _ in events]
    assert names[0] == "analysis" and names[1] == "sources" and names[-1] == "done"
    assert events[0][1]["category"]["label"] == "Billing and Payments"
    tokens = "".join(data["text"] for name, data in events if name == "token")
    done = events[-1][1]
    assert tokens == done["response"]["draft_reply"]["text"]
    assert names.count("token") > 1, "the draft arrives in pieces"
    assert done["partial"] is False
    AnalyzeResponse.model_validate(done["response"])


def test_stream_reports_an_unavailable_model_in_the_done_event(embedder, vector_store):
    copilot = make_copilot(embedder, vector_store, llm=BrokenLLM())
    with TestClient(create_app(copilot)) as client:
        events = _events(client.post("/analyze/stream", json=TICKET).text)

    done = events[-1][1]
    assert events[0][0] == "analysis"
    assert done["partial"] is True
    assert done["response"]["draft_reply"]["status"] == "unavailable"
    assert done["response"]["draft_reply"]["sources"]


def test_stream_validates_before_streaming(client):
    assert client.post("/analyze/stream", json={"text": " "}).status_code == 422


def test_stream_turns_unexpected_failures_into_an_error_event(embedder, vector_store):
    copilot = make_copilot(embedder, vector_store)

    def explode(*args, **kwargs):
        raise ValueError("boom")

    copilot.risk_model.score = explode
    with TestClient(create_app(copilot)) as client:
        events = _events(client.post("/analyze/stream", json=TICKET).text)

    assert events == [("error", {"detail": "The analysis failed. Try again."})]


def test_analysis_includes_word_highlights(client):
    body = client.post("/analyze", json={"text": "Charged twice, refund please. Thanks"}).json()

    assert body["highlights"]["category"] == [
        {"term": "charge", "weight": 1.0},
        {"term": "refund", "weight": 1.0},
    ]
    assert {"term": "twice", "weight": 0.8} in body["highlights"]["risk"]
    assert {"term": "thanks", "weight": -0.3} in body["highlights"]["risk"]


def test_a_failing_explanation_does_not_fail_the_analysis(embedder, vector_store):
    copilot = make_copilot(embedder, vector_store)

    def explode(*args, **kwargs):
        raise RuntimeError("no gradients")

    copilot.classifier.word_weights = explode
    with TestClient(create_app(copilot)) as client:
        response = client.post("/analyze", json=TICKET)

    assert response.status_code == 200
    assert response.json()["highlights"]["category"] == []
    assert response.json()["highlights"]["risk"]
