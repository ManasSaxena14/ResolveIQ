"""Gradio web page for the copilot.

Left: the ticket form and batch upload. Right: three cards for the draft reply,
the predicted category and the escalation risk. Footer: model versions,
response time and feedback buttons that store a vote for the request.

The handlers call the same ``Copilot`` methods as the HTTP API.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from resolveiq.schemas import CHANNELS, TICKET_TYPES, AnalyzeRequest
from resolveiq.service import Copilot, InputError, PayloadTooLargeError
from resolveiq.ui.formatting import (
    BATCH_HEADERS,
    NOTICE,
    batch_rows,
    category_scores,
    render_draft,
    render_footer,
    render_gauge,
    render_reasons,
    render_review,
)

EXAMPLES = [
    [
        "Charged twice",
        "I was charged twice for my order this month and nobody has replied to my last two "
        "emails. Please refund the second charge.",
        "Problem",
        "email",
        "",
    ],
    [
        "Return window",
        "How many days do I have to return headphones that I do not like?",
        "Request",
        "chat",
        "Headphones",
    ],
    [
        "Service down",
        "Our whole team cannot log in since this morning. The dashboard shows an error and "
        "this is urgent because we have a deadline today.",
        "Incident",
        "web",
        "",
    ],
]

CSS = """
.card {border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 12px;}
.footer {font-size: 0.85em; opacity: 0.8;}
"""

EMPTY_OUTPUTS: tuple[Any, ...] = ("", {}, "", "", "", "", "")


def analyze_ticket(
    copilot: Copilot,
    subject: str,
    text: str,
    ticket_type: str,
    channel: str,
    product: str,
    source_base_url: str = "",
) -> tuple[Any, ...]:
    """Outputs: draft, category scores, review note, gauge, reasons, footer, request id."""
    try:
        request = AnalyzeRequest(
            text=text or "",
            subject=subject or "",
            ticket_type=ticket_type or "",
            channel=channel or "",
            product=product or "",
        )
        response = copilot.analyze(request).response
    except ValidationError as exc:
        message = "; ".join(error["msg"].removeprefix("Value error, ") for error in exc.errors())
        return (f"**Cannot analyze:** {message}", *EMPTY_OUTPUTS[1:])
    except InputError as exc:
        return (f"**Cannot analyze:** {exc}", *EMPTY_OUTPUTS[1:])
    return (
        render_draft(response.draft_reply, source_base_url),
        category_scores(response.category),
        render_review(response.category),
        render_gauge(
            response.risk.escalation_probability, response.risk.threshold, response.risk.alert
        ),
        render_reasons(response.risk.reasons),
        render_footer(response),
        response.request_id,
    )


def submit_feedback(copilot: Copilot, request_id: str, rating: str) -> str:
    if not request_id:
        return "Analyze a ticket first."
    if copilot.feedback(request_id, rating):
        return "Thanks, your feedback was saved."
    return "Feedback could not be saved for this request."


def score_batch(copilot: Copilot, file_path: str | None) -> tuple[list[list[object]], str]:
    if not file_path:
        return [], "Upload a CSV file with a 'body' or 'text' column."
    with open(file_path, "rb") as handle:
        payload = handle.read(copilot.max_upload_bytes + 1)
    try:
        result = copilot.batch(payload)
    except (InputError, PayloadTooLargeError) as exc:
        return [], f"**Cannot score the file:** {exc}"
    return batch_rows(result), f"Scored {result.count} tickets, {result.alerts} alerts."


def build_demo(copilot: Copilot, source_base_url: str = "") -> Any:
    import gradio as gr

    with gr.Blocks(title="ResolveIQ") as demo:
        gr.Markdown(f"# ResolveIQ support copilot\n{NOTICE}")
        request_id = gr.State("")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                subject = gr.Textbox(label="Subject (optional)", max_lines=1)
                text = gr.Textbox(
                    label="Ticket",
                    lines=8,
                    max_length=copilot.max_text_chars,
                    placeholder="Paste the customer's message",
                )
                with gr.Row():
                    ticket_type = gr.Dropdown(
                        ["", *TICKET_TYPES], value="", label="Ticket type (optional)"
                    )
                    channel = gr.Dropdown(["", *CHANNELS], value="", label="Channel (optional)")
                    product = gr.Textbox(label="Product (optional)", max_lines=1)
                analyze_button = gr.Button("Analyze", variant="primary")
                gr.Examples(EXAMPLES, inputs=[subject, text, ticket_type, channel, product])
                with gr.Accordion("Batch scoring (CSV)", open=False):
                    gr.Markdown(
                        "Columns: `body` or `text` required, `subject`, `type` and `id` "
                        "optional. Returns tickets ranked by escalation risk."
                    )
                    upload = gr.File(label="Tickets CSV", file_types=[".csv"], type="filepath")
                    batch_button = gr.Button("Score file")
                    batch_status = gr.Markdown()
                    batch_table = gr.Dataframe(headers=BATCH_HEADERS, interactive=False)
            with gr.Column(scale=7):
                with gr.Group(elem_classes="card"):
                    gr.Markdown("### Draft reply")
                    draft = gr.Markdown("Analyze a ticket to see a draft reply.")
                with gr.Row(equal_height=False):
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### Category")
                        category = gr.Label(num_top_classes=3, show_label=False)
                        review = gr.Markdown()
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### Escalation risk")
                        gauge = gr.HTML()
                        reasons = gr.Markdown()
        footer = gr.Markdown(elem_classes="footer")
        with gr.Row():
            up = gr.Button("Helpful", size="sm")
            down = gr.Button("Not helpful", size="sm")
            feedback_status = gr.Markdown()

        analyze_button.click(
            lambda *args: analyze_ticket(copilot, *args, source_base_url=source_base_url),
            inputs=[subject, text, ticket_type, channel, product],
            outputs=[draft, category, review, gauge, reasons, footer, request_id],
        )
        up.click(lambda rid: submit_feedback(copilot, rid, "up"), request_id, feedback_status)
        down.click(lambda rid: submit_feedback(copilot, rid, "down"), request_id, feedback_status)
        batch_button.click(
            lambda path: score_batch(copilot, path), upload, [batch_table, batch_status]
        )
    return demo


def launch_options() -> dict[str, Any]:
    """Arguments for ``demo.launch`` shared by app.py and local runs."""
    import gradio as gr

    return {"css": CSS, "theme": gr.themes.Soft()}
