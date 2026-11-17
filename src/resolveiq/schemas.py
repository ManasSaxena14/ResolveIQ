"""Request and response shapes shared by the service, the API and the UI.

The field names follow the API specification in docs/api.md. Every response
model is also the documented JSON schema of its endpoint (see ``/docs``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TICKET_TYPES = ("Incident", "Problem", "Request", "Change")
CHANNELS = ("email", "chat", "phone", "web")

DraftStatus = Literal["ok", "refused", "unavailable"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnalyzeRequest(_Model):
    text: str = Field(description="Ticket body as written by the customer")
    subject: str = Field(default="", max_length=300, description="Optional ticket subject")
    ticket_type: Literal["", "Incident", "Problem", "Request", "Change"] = Field(
        default="", description="Ticket type chosen on the request form, if known"
    )
    channel: str = Field(default="", max_length=30, description="Contact channel, for monitoring")
    product: str = Field(default="", max_length=100, description="Product the ticket is about")

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Ticket text must not be empty")
        return value


class SourceOut(_Model):
    file: str
    page: int
    score: float | None = None


class DraftReply(_Model):
    text: str | None = Field(description="Draft for an agent to edit; null when unavailable")
    status: DraftStatus
    refused: bool
    sources: list[SourceOut] = Field(default_factory=list)
    error: str | None = None


class CategoryScore(_Model):
    label: str
    probability: float


class CategoryOut(_Model):
    label: str
    confidence: float
    needs_review: bool
    top: list[CategoryScore] = Field(default_factory=list)


class RiskOut(_Model):
    escalation_probability: float
    alert: bool
    threshold: float
    reasons: list[str]


class AnalyzeResponse(_Model):
    request_id: str
    draft_reply: DraftReply
    category: CategoryOut
    risk: RiskOut
    timings_ms: dict[str, float]
    versions: dict[str, str]


class BatchItem(_Model):
    rank: int
    id: str
    escalation_probability: float
    alert: bool
    reasons: list[str]


class BatchResponse(_Model):
    count: int
    alerts: int
    items: list[BatchItem]


class FeedbackRequest(_Model):
    request_id: str = Field(min_length=1, max_length=64)
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=500)


class FeedbackResponse(_Model):
    request_id: str
    rating: Literal["up", "down"]
    stored: bool


class HealthResponse(_Model):
    status: Literal["ok", "degraded"]
    components: dict[str, bool]
    versions: dict[str, str]
    index: dict[str, int]
    uptime_seconds: float


class ErrorResponse(_Model):
    detail: str
