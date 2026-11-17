"""Turn service responses into the Markdown and HTML shown in the web UI.

These helpers do not import Gradio, so they are unit tested in CI.
"""

from __future__ import annotations

import html
import math
from urllib.parse import quote

from resolveiq.schemas import AnalyzeResponse, BatchResponse, CategoryOut, DraftReply

NOTICE = (
    "Demo on synthetic data. Do not enter personal data. Replies are drafts for a human "
    "agent to review and are never sent automatically."
)

_ARC_LENGTH = math.pi * 80  # radius 80 semicircle


def render_draft(draft: DraftReply, source_base_url: str = "") -> str:
    if draft.status == "unavailable":
        lines = [f"**Draft unavailable.** {draft.error or ''}".strip()]
    elif draft.status == "refused":
        lines = [
            "**No answer in the knowledge base.** The question looks out of scope, so no "
            "reply was drafted.",
            "",
            f"> {draft.text}",
        ]
    else:
        lines = [draft.text or ""]
    if draft.sources:
        lines += ["", "**Sources**", ""]
        for source in draft.sources:
            name = f"{source.file}, page {source.page}"
            label = f"[{name}]({source_base_url}{quote(source.file)})" if source_base_url else name
            score = "" if source.score is None else f" (similarity {source.score:.2f})"
            lines.append(f"- {label}{score}")
    return "\n".join(lines)


def category_scores(category: CategoryOut) -> dict[str, float]:
    """Top queues and their probabilities, in the format ``gr.Label`` expects."""
    return {item.label: item.probability for item in category.top} or {
        category.label: category.confidence
    }


def render_review(category: CategoryOut) -> str:
    if category.needs_review:
        return (
            f"**Needs review.** Confidence {category.confidence:.0%} is below the review "
            "threshold, so a person should confirm the queue."
        )
    return f"Confident routing to **{category.label}** ({category.confidence:.0%})."


def render_gauge(probability: float, threshold: float, alert: bool) -> str:
    """A half-circle gauge with the alert threshold marked, as inline SVG."""
    probability = min(max(probability, 0.0), 1.0)
    filled = probability * _ARC_LENGTH
    color = "#dc2626" if alert else "#16a34a"
    angle = math.pi * (1 - threshold)
    inner = (100 + 66 * math.cos(angle), 100 - 66 * math.sin(angle))
    outer = (100 + 94 * math.cos(angle), 100 - 94 * math.sin(angle))
    status = "Alert: likely to escalate" if alert else "No alert"
    return (
        '<div style="text-align:center">'
        '<svg viewBox="0 0 200 120" width="220" role="img" '
        f'aria-label="Escalation probability {probability:.0%}">'
        '<path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#9ca3af" '
        'stroke-opacity="0.35" stroke-width="16" stroke-linecap="round"/>'
        f'<path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="{color}" '
        f'stroke-width="16" stroke-linecap="round" '
        f'stroke-dasharray="{filled:.1f} {_ARC_LENGTH:.1f}"/>'
        f'<line x1="{inner[0]:.1f}" y1="{inner[1]:.1f}" x2="{outer[0]:.1f}" y2="{outer[1]:.1f}" '
        'stroke="currentColor" stroke-width="2"/>'
        f'<text x="100" y="92" text-anchor="middle" font-size="28" font-weight="600" '
        f'fill="currentColor">{probability:.0%}</text>'
        f'<text x="100" y="116" text-anchor="middle" font-size="11" fill="currentColor">'
        f"threshold {threshold:.0%}</text>"
        "</svg>"
        f'<div style="font-weight:600;color:{color}">{status}</div>'
        "</div>"
    )


def render_reasons(reasons: list[str]) -> str:
    if not reasons:
        return "No feature raises the risk noticeably."
    return "**Top reasons**\n\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(reasons, 1))


def render_footer(response: AnalyzeResponse) -> str:
    versions = " | ".join(f"{k}: {html.escape(v)}" for k, v in response.versions.items())
    total = response.timings_ms.get("total", 0.0)
    return f"Request `{response.request_id}` | {total:.0f} ms | {versions}"


BATCH_HEADERS = ["Rank", "ID", "Escalation probability", "Alert", "Reasons"]


def batch_rows(result: BatchResponse) -> list[list[object]]:
    return [
        [
            item.rank,
            item.id,
            round(item.escalation_probability, 3),
            "yes" if item.alert else "no",
            "; ".join(item.reasons),
        ]
        for item in result.items
    ]
