import pytest

from resolveiq.monitoring.logging import RequestLog, RequestRecord, text_fingerprint


def _record(request_id: str, category: str = "Billing", created_at: str = "") -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        channel="email",
        category=category,
        confidence=0.8,
        needs_review=False,
        risk_score=0.3,
        alert=False,
        text_length=42,
        text_hash="abc",
        draft_status="ok",
        latency_ms=12.5,
        versions={"risk": "v1"},
        created_at=created_at,
    )


def test_fingerprint_hides_the_text():
    length, digest = text_fingerprint("My email is jane@example.com")

    assert length == 28
    assert len(digest) == 16
    assert "jane" not in digest
    assert text_fingerprint("My email is jane@example.com")[1] == digest


def test_records_are_returned_oldest_first(tmp_path):
    log = RequestLog(tmp_path / "logs" / "requests.db")
    log.record(_record("a", created_at="2026-01-01T00:00:00+00:00"))
    log.record(_record("b", created_at="2026-01-02T00:00:00+00:00"))
    log.record(_record("c", created_at="2026-01-03T00:00:00+00:00"))

    recent = log.recent(2)

    assert [row["request_id"] for row in recent] == ["b", "c"]
    assert recent[0]["versions"] == {"risk": "v1"}
    assert recent[0]["alert"] is False
    assert log.count() == 3
    assert "text" not in recent[0]


def test_feedback_needs_a_known_request():
    log = RequestLog(":memory:")
    log.record(_record("known"))

    assert log.add_feedback("known", "up") is True
    assert log.add_feedback("known", "down", "Wrong queue, call me on +1 415 555 0100") is True
    assert log.add_feedback("missing", "up") is False
    stats = log.feedback_stats()
    assert (stats.total, stats.up, stats.down) == (2, 1, 1)
    assert stats.as_dict()["down_rate"] == 0.5


def test_feedback_comment_is_masked():
    log = RequestLog(":memory:")
    log.record(_record("r1"))
    log.add_feedback("r1", "down", "write to jane@example.com")

    with log._transaction() as cursor:
        cursor.execute("SELECT comment FROM feedback")
        assert cursor.fetchone()[0] == "write to [EMAIL]"


def test_invalid_rating_is_rejected():
    log = RequestLog(":memory:")
    with pytest.raises(ValueError, match="rating"):
        log.add_feedback("r1", "maybe")


def test_empty_feedback_rate_is_zero():
    assert RequestLog(":memory:").feedback_stats().down_rate == 0.0
