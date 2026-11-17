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
