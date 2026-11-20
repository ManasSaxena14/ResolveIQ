"""One call that runs all three parts on a ticket.

``Copilot.analyze`` drafts a grounded reply, predicts the routing queue and
scores the escalation risk, then logs one monitoring row. The FastAPI service
and the Gradio UI are thin layers over this class.

If the language model fails, the category and risk are still returned and the
draft is marked unavailable, with the retrieved sources kept so an agent can
read them.
"""

from __future__ import annotations

import csv
import io
import logging
import time
import uuid
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from resolveiq.classifier.data import build_text
from resolveiq.config import Settings
from resolveiq.monitoring.logging import RequestLog, RequestRecord, log_event, text_fingerprint
from resolveiq.rag.grading import check_grounding
from resolveiq.rag.llm import LLMError
from resolveiq.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchItem,
    BatchResponse,
    CategoryOut,
    CategoryScore,
    DraftReply,
    GroundingOut,
    HealthResponse,
    Highlights,
    RequestSummary,
    RiskOut,
    SourceOut,
    StatsResponse,
    WordWeight,
)

logger = logging.getLogger(__name__)

TOP_CATEGORIES = 3
MAX_QUESTION_CHARS = 4000


class InputError(ValueError):
    """The request is invalid (empty text, too long, bad CSV). Maps to HTTP 422."""


class PayloadTooLargeError(ValueError):
    """An upload is larger than allowed. Maps to HTTP 413."""


class AnswerEngineLike(Protocol):
    def stream(self, question: str) -> Iterator[tuple[str, Any]]: ...
    def retrieve_only(self, question: str) -> Any: ...


class ClassifierLike(Protocol):
    def predict_ticket(self, body: str, subject: str = "") -> Any: ...


class RiskModelLike(Protocol):
    threshold: float

    def score(self, subject: str, body: str, ticket_type: str = "") -> Any: ...
    def score_texts(self, texts: list[str], ticket_types: list[str], k: int = 3) -> list[Any]: ...


@dataclass(frozen=True)
class AnalyzeResult:
    response: AnalyzeResponse
    partial: bool  # True when the draft could not be produced


class Copilot:
    def __init__(
        self,
        answer_engine: AnswerEngineLike | None,
        classifier: ClassifierLike,
        risk_model: RiskModelLike,
        versions: dict[str, str],
        request_log: RequestLog | None = None,
        max_text_chars: int = 4000,
        max_upload_bytes: int = 2_000_000,
        max_batch_rows: int = 2000,
        index_info: dict[str, int] | None = None,
    ) -> None:
        self.answer_engine = answer_engine
        self.classifier = classifier
        self.risk_model = risk_model
        self.versions = dict(versions)
        self.request_log = request_log
        self.max_text_chars = max_text_chars
        self.max_upload_bytes = max_upload_bytes
        self.max_batch_rows = max_batch_rows
        self.index_info = index_info or {}
        self.started = time.monotonic()

    @classmethod
    def from_settings(cls, settings: Settings, with_llm: bool = True) -> Copilot:
        """Load every model named in the config. Slow: call once per process."""
        from resolveiq.classifier.predict import TicketClassifier
        from resolveiq.rag.generate import AnswerEngine
        from resolveiq.rag.ingest import read_manifest
        from resolveiq.registry import resolve_model_dir
        from resolveiq.risk.predict import RiskModel

        classifier_dir = resolve_model_dir(settings, "classifier")
        risk_dir = resolve_model_dir(settings, "risk")
        classifier = TicketClassifier.load(classifier_dir.path)
        risk_model = RiskModel.load(risk_dir.path)

        try:
            answer_engine: AnswerEngine | None = AnswerEngine.from_settings(settings, with_llm)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Reply drafting is disabled: %s", exc)
            answer_engine = None

        manifest = read_manifest(settings.resolve(settings.paths.index_dir)) or {}
        llm = settings.llm
        versions = {
            "embedder": settings.rag.embedding_model,
            "llm": f"{llm.provider}/{llm.active.model}" if with_llm else "disabled",
            "classifier": _classifier_version(classifier_dir.path, classifier_dir.version),
            "risk": f"{risk_model.version} ({risk_dir.source})",
        }
        return cls(
            answer_engine=answer_engine,
            classifier=classifier,
            risk_model=risk_model,
            versions=versions,
            request_log=RequestLog(settings.resolve(settings.service.request_log)),
            max_text_chars=settings.service.max_text_chars,
            max_upload_bytes=settings.service.max_upload_bytes,
            max_batch_rows=settings.service.max_batch_rows,
            index_info={
                "files": int(manifest.get("files", 0)),
                "chunks": int(manifest.get("chunks", 0)),
            },
        )

    # ------------------------------------------------------------------ analyze

    def validate(self, request: AnalyzeRequest) -> None:
        if not request.text.strip():
            raise InputError("Ticket text must not be empty")
        if len(request.text) > self.max_text_chars:
            raise InputError(f"Ticket text is longer than {self.max_text_chars} characters")

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResult:
        """Run all three parts and return the complete result."""
        final: dict[str, Any] = {}
        for name, data in self.analyze_stream(request):
            if name == "done":
                final = data
        return AnalyzeResult(AnalyzeResponse.model_validate(final["response"]), final["partial"])

    def analyze_stream(self, request: AnalyzeRequest) -> Iterator[tuple[str, Any]]:
        """Validate now, then yield events as each part finishes.

        Events, in order: ``analysis`` (request ID, category and risk, which take
        milliseconds), ``sources`` (when retrieval found relevant articles), ``token``
        (each piece of the draft as the language model writes it) and ``done`` (the
        complete response and whether it is partial). Validation errors are raised
        here, before any event, so the API can still answer 422.
        """
        self.validate(request)
        return self._events(request)

    def _events(self, request: AnalyzeRequest) -> Iterator[tuple[str, Any]]:
        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        timings: dict[str, float] = {}

        step = time.perf_counter()
        prediction = self.classifier.predict_ticket(request.text, request.subject)
        timings["classifier"] = _elapsed_ms(step)

        step = time.perf_counter()
        risk = self.risk_model.score(request.subject, request.text, request.ticket_type)
        timings["risk"] = _elapsed_ms(step)

        step = time.perf_counter()
        highlights = self._highlights(request)
        timings["explain"] = _elapsed_ms(step)

        top = sorted(prediction.probabilities.items(), key=lambda item: item[1], reverse=True)
        category = CategoryOut(
            label=prediction.label,
            confidence=prediction.confidence,
            needs_review=prediction.needs_review,
            top=[CategoryScore(label=k, probability=v) for k, v in top[:TOP_CATEGORIES]],
        )
        risk_out = RiskOut(
            escalation_probability=risk.escalation_probability,
            alert=risk.alert,
            threshold=self.risk_model.threshold,
            reasons=list(risk.reasons),
        )
        yield (
            "analysis",
            {
                "request_id": request_id,
                "category": category.model_dump(),
                "risk": risk_out.model_dump(),
                "highlights": highlights.model_dump(),
                "versions": self.versions,
            },
        )

        draft = yield from self._draft_events(request, timings)
        timings["total"] = _elapsed_ms(started)

        response = AnalyzeResponse(
            request_id=request_id,
            draft_reply=draft,
            category=category,
            risk=risk_out,
            timings_ms=timings,
            versions=self.versions,
            highlights=highlights,
        )
        self._record(request, response)
        yield "done", {"response": response.model_dump(), "partial": draft.status == "unavailable"}

    def _highlights(self, request: AnalyzeRequest) -> Highlights:
        """Word weights from each model that can explain itself; never fails the request."""
        found: dict[str, list[WordWeight]] = {}
        sources = {
            "category": getattr(self.classifier, "word_weights", None),
            "risk": getattr(self.risk_model, "word_weights", None),
        }
        for name, explain in sources.items():
            if not callable(explain):
                continue
            try:
                if name == "category":
                    pairs = explain(request.text, request.subject)
                else:
                    pairs = explain(request.subject, request.text)
            except Exception as exc:  # an explanation must never break the analysis
                logger.warning("Could not explain the %s: %s", name, exc)
                continue
            found[name] = [WordWeight(term=term, weight=weight) for term, weight in pairs]
        return Highlights(**found)

    def _question(self, request: AnalyzeRequest) -> str:
        parts = [request.subject.strip(), request.text.strip()]
        if request.product.strip():
            parts.append(f"Product: {request.product.strip()}")
        # The answer engine rejects questions over 4,000 characters.
        return "\n\n".join(part for part in parts if part)[:MAX_QUESTION_CHARS]

    def _draft_events(
        self, request: AnalyzeRequest, timings: dict[str, float]
    ) -> Generator[tuple[str, Any], None, DraftReply]:
        """Yield ``sources`` and ``token`` events, then return the finished draft."""
        if self.answer_engine is None:
            return DraftReply(
                text=None,
                status="unavailable",
                refused=False,
                error="Reply drafting is not configured (no knowledge base index)",
            )
        question = self._question(request)
        sources: list[Any] | None = None
        result: Any = None
        try:
            for kind, value in self.answer_engine.stream(question):
                if kind == "sources":
                    sources, step_timings = value
                    timings.update(step_timings)
                    yield "sources", [source.model_dump() for source in _sources(sources)]
                elif kind == "token":
                    yield "token", {"text": value}
                else:
                    result = value
        except (LLMError, RuntimeError) as exc:
            log_event("llm_unavailable", error=type(exc).__name__)
            refused = False
            if sources is None:
                fallback = self.answer_engine.retrieve_only(question)
                timings["retrieval"] = fallback.timings_ms.get("retrieval", 0.0)
                sources, refused = fallback.sources, fallback.refused
            return DraftReply(
                text=None,
                status="unavailable",
                refused=refused,
                sources=_sources(sources),
                error="The language model is unavailable; showing the retrieved sources only",
            )
        timings["retrieval"] = result.timings_ms.get("retrieval", 0.0)
        if "generation" in result.timings_ms:
            timings["generation"] = result.timings_ms["generation"]
        grounding = None
        if not result.refused:
            check = check_grounding(result.answer, [(s.file, s.page) for s in result.sources])
            grounding = GroundingOut(**check.as_dict())
        return DraftReply(
            text=result.answer,
            status="refused" if result.refused else "ok",
            refused=result.refused,
            sources=_sources(result.sources),
            grounding=grounding,
        )

    def _record(self, request: AnalyzeRequest, response: AnalyzeResponse) -> None:
        length, digest = text_fingerprint(request.text)
        log_event(
            "analyze",
            request_id=response.request_id,
            text_length=length,
            text_hash=digest,
            channel=request.channel,
            category=response.category.label,
            risk=response.risk.escalation_probability,
            draft=response.draft_reply.status,
            timings_ms=response.timings_ms,
        )
        if self.request_log is None:
            return
        self.request_log.record(
            RequestRecord(
                request_id=response.request_id,
                channel=request.channel.strip().lower(),
                category=response.category.label,
                confidence=response.category.confidence,
                needs_review=response.category.needs_review,
                risk_score=response.risk.escalation_probability,
                alert=response.risk.alert,
                text_length=length,
                text_hash=digest,
                draft_status=response.draft_reply.status,
                latency_ms=response.timings_ms["total"],
                versions=response.versions,
            )
        )

    # -------------------------------------------------------------------- batch

    def batch(self, payload: bytes) -> BatchResponse:
        """Rank a CSV of tickets by escalation risk.

        The CSV needs a ``body`` (or ``text``) column; ``subject``, ``type`` and
        ``id`` are optional. Ticket text is never returned.
        """
        if len(payload) > self.max_upload_bytes:
            raise PayloadTooLargeError(f"The file is larger than {self.max_upload_bytes} bytes")
        try:
            content = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InputError("The file must be UTF-8 encoded CSV") from exc
        reader = csv.DictReader(io.StringIO(content))
        columns = {name.strip().lower() for name in (reader.fieldnames or [])}
        body_column = "body" if "body" in columns else "text" if "text" in columns else None
        if body_column is None:
            raise InputError("The CSV needs a 'body' or 'text' column")
        rows = [{k.strip().lower(): (v or "") for k, v in row.items() if k} for row in reader]
        if not rows:
            raise InputError("The CSV has no rows")
        if len(rows) > self.max_batch_rows:
            raise InputError(f"The CSV has more than {self.max_batch_rows} rows")

        texts = [build_text(row.get("subject", ""), row[body_column]) for row in rows]
        empty = [i + 1 for i, text in enumerate(texts) if not text.strip()]
        if empty:
            raise InputError(f"Rows with empty text: {empty[:10]}")
        types = [row.get("type", "") for row in rows]
        scores = self.risk_model.score_texts(texts, types)
        ranked = sorted(
            enumerate(scores), key=lambda item: item[1].escalation_probability, reverse=True
        )
        items = [
            BatchItem(
                rank=rank,
                id=rows[index].get("id") or str(index + 1),
                escalation_probability=score.escalation_probability,
                alert=score.alert,
                reasons=list(score.reasons),
            )
            for rank, (index, score) in enumerate(ranked, start=1)
        ]
        log_event("batch", rows=len(items), alerts=sum(item.alert for item in items))
        return BatchResponse(
            count=len(items), alerts=sum(item.alert for item in items), items=items
        )

    # ---------------------------------------------------------- feedback/health

    def feedback(self, request_id: str, rating: str, comment: str = "") -> bool:
        if self.request_log is None:
            return False
        stored = self.request_log.add_feedback(request_id, rating, comment)
        log_event("feedback", request_id=request_id, rating=rating, stored=stored)
        return stored

    def recent_requests(self, limit: int = 50) -> list[RequestSummary]:
        """Newest requests first, with the latest feedback vote of each."""
        if self.request_log is None:
            return []
        rows = list(reversed(self.request_log.recent(limit)))
        ratings = self.request_log.ratings([row["request_id"] for row in rows])
        fields = set(RequestSummary.model_fields) - {"rating"}
        return [
            RequestSummary(
                **{k: v for k, v in row.items() if k in fields},
                rating=ratings.get(row["request_id"]),
            )
            for row in rows
        ]

    def stats(self) -> StatsResponse:
        if self.request_log is None:
            return StatsResponse(
                requests=0,
                alerts=0,
                needs_review=0,
                mean_risk=None,
                mean_confidence=None,
                categories={},
                draft_status={},
                latency_median_ms=None,
                latency_p95_ms=None,
                feedback={"total": 0, "up": 0, "down": 0, "down_rate": 0.0},
            )
        return StatsResponse(
            **self.request_log.summary(), feedback=self.request_log.feedback_stats().as_dict()
        )

    def health(self) -> HealthResponse:
        components = {
            "retriever": self.answer_engine is not None,
            "llm_configured": getattr(self.answer_engine, "llm", None) is not None,
            "classifier": self.classifier is not None,
            "risk": self.risk_model is not None,
            "request_log": self.request_log is not None,
        }
        status = "ok" if all(components.values()) else "degraded"
        return HealthResponse(
            status=status,
            components=components,
            versions=self.versions,
            index=self.index_info,
            uptime_seconds=round(time.monotonic() - self.started, 1),
        )


def _sources(sources: list[Any]) -> list[SourceOut]:
    return [SourceOut(file=s.file, page=s.page, score=s.score) for s in sources]


def _classifier_version(model_dir: Any, registry_version: str) -> str:
    import json
    from pathlib import Path

    meta_path = Path(model_dir) / "classifier_meta.json"
    try:
        config = json.loads(meta_path.read_text(encoding="utf-8"))["train_config"]
        name = f"{config['model_name']}-seed-{config['seed']}"
    except (OSError, KeyError, ValueError):
        name = "unknown"
    return name if registry_version == "local" else f"{name} ({registry_version})"


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
