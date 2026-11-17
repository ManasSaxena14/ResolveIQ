"""The UI handlers and formatting run without Gradio installed."""

from resolveiq.schemas import (
    BatchItem,
    BatchResponse,
    CategoryOut,
    CategoryScore,
    DraftReply,
    SourceOut,
)
from resolveiq.ui.formatting import (
    batch_rows,
    category_scores,
    render_draft,
    render_gauge,
    render_reasons,
    render_review,
)
from resolveiq.ui.gradio_app import analyze_ticket, score_batch, submit_feedback
from service_doubles import BrokenLLM, make_copilot

BASE = "https://example.org/kb/"


def test_draft_links_its_sources():
    draft = DraftReply(
        text="Refunds take 5 days.",
        status="ok",
        refused=False,
        sources=[SourceOut(file="returns/refund timelines.md", page=1, score=0.81)],
    )

    markdown = render_draft(draft, BASE)

    assert markdown.startswith("Refunds take 5 days.")
    assert (
        "[returns/refund timelines.md, page 1](https://example.org/kb/returns/refund%20timelines.md)"
        in markdown
    )
    assert "(similarity 0.81)" in markdown
    assert "](" not in render_draft(draft)


def test_refused_and_unavailable_drafts_explain_themselves():
    refused = DraftReply(text="I could not find this.", status="refused", refused=True)
    missing = DraftReply(text=None, status="unavailable", refused=False, error="LLM down")

    assert "out of scope" in render_draft(refused)
    assert render_draft(missing) == "**Draft unavailable.** LLM down"


def test_category_card():
    category = CategoryOut(
        label="Billing",
        confidence=0.55,
        needs_review=True,
        top=[CategoryScore(label="Billing", probability=0.55)],
    )

    assert category_scores(category) == {"Billing": 0.55}
    assert category_scores(category.model_copy(update={"top": []})) == {"Billing": 0.55}
    assert render_review(category).startswith("**Needs review.**")
    assert "Confident" in render_review(category.model_copy(update={"needs_review": False}))


def test_gauge_shows_probability_and_alert_state():
    alert = render_gauge(0.8, 0.23, True)
    calm = render_gauge(0.1, 0.23, False)

    assert "80%" in alert and "#dc2626" in alert and "Alert" in alert
    assert "10%" in calm and "No alert" in calm
    assert "threshold 23%" in calm
    assert "150%" not in render_gauge(1.5, 0.5, True)


def test_reasons_list():
    assert render_reasons([]) == "No feature raises the risk noticeably."
    assert "1. A\n2. B" in render_reasons(["A", "B"])


def test_batch_rows():
    result = BatchResponse(
        count=1,
        alerts=1,
        items=[
            BatchItem(
                rank=1, id="t1", escalation_probability=0.91234, alert=True, reasons=["a", "b"]
            )
        ],
    )

    assert batch_rows(result) == [[1, "t1", 0.912, "yes", "a; b"]]


def test_analyze_handler_fills_every_card(embedder, vector_store):
    copilot = make_copilot(embedder, vector_store)

    outputs = analyze_ticket(
        copilot, "Billing", "I was charged twice this month", "Problem", "email", "", BASE
    )

    draft, scores, review, gauge, reasons, footer, request_id = outputs
    assert "Sources" in draft
    assert "Billing and Payments" in scores
    assert review
    assert "<svg" in gauge
    assert "twice" in reasons
    assert request_id in footer
    assert submit_feedback(copilot, request_id, "up").startswith("Thanks")


def test_analyze_handler_reports_input_errors(embedder, vector_store):
    copilot = make_copilot(embedder, vector_store, max_text_chars=20)

    empty = analyze_ticket(copilot, "", "  ", "", "", "")
    long = analyze_ticket(copilot, "", "x" * 30, "", "", "")

    assert empty[0] == "**Cannot analyze:** Ticket text must not be empty"
    assert "longer than 20" in long[0]
    assert empty[-1] == ""


def test_analyze_handler_survives_a_broken_llm(embedder, vector_store):
    copilot = make_copilot(embedder, vector_store, llm=BrokenLLM())

    draft, *_ = analyze_ticket(copilot, "", "I was charged twice this month", "", "", "")

    assert draft.startswith("**Draft unavailable.**")


def test_feedback_handler_needs_a_request(embedder, vector_store):
    copilot = make_copilot(embedder, vector_store)

    assert submit_feedback(copilot, "", "up") == "Analyze a ticket first."
    assert "could not be saved" in submit_feedback(copilot, "unknown", "down")


def test_batch_handler(embedder, vector_store, tmp_path):
    copilot = make_copilot(embedder, vector_store)
    good = tmp_path / "good.csv"
    good.write_text("id,body\n1,server down urgent\n2,thanks\n")
    bad = tmp_path / "bad.csv"
    bad.write_text("id,message\n1,hi\n")

    rows, status = score_batch(copilot, str(good))

    assert [row[1] for row in rows] == ["1", "2"]
    assert status == "Scored 2 tickets, 1 alerts."
    assert score_batch(copilot, str(bad))[1].startswith("**Cannot score the file:**")
    assert score_batch(copilot, None)[0] == []
