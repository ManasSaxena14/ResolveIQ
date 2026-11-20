"""HTTP API for ResolveIQ.

Run locally with::

    uvicorn resolveiq.api.main:app --port 8000

Models are loaded once at startup. Interactive documentation of every request
and response shape is served at ``/docs``.
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    # PyTorch and XGBoost bundle different OpenMP runtimes on macOS and crash when
    # both start worker threads in one process. Must run before either is imported.
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from resolveiq import __version__
from resolveiq.api.limits import API_KEY_HEADER, AccessPolicy
from resolveiq.config import PROJECT_ROOT
from resolveiq.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    RequestSummary,
    StatsResponse,
)
from resolveiq.service import Copilot, InputError, PayloadTooLargeError

logger = logging.getLogger("resolveiq.api")

REQUEST_ID_HEADER = "X-Request-ID"
# The React app is built into this folder by `npm run build` in frontend/.
WEB_DIR = PROJECT_ROOT / "frontend" / "dist"
# Routes that do model work or read the log; /health and the web app are never limited.
LIMITED_PREFIXES = ("/analyze", "/batch", "/feedback", "/requests", "/stats")


def _load_from_settings() -> Copilot:
    from resolveiq.config import get_settings
    from resolveiq.registry import fetch_from_hub

    settings = get_settings()
    fetch_from_hub(settings)
    return Copilot.from_settings(settings)


def _default_access() -> AccessPolicy:
    from resolveiq.config import get_settings

    return AccessPolicy.from_config(get_settings().service.rate_limit)


def create_app(
    copilot: Copilot | None = None,
    loader: Callable[[], Copilot] = _load_from_settings,
    web_dir: Path | None = WEB_DIR,
    access: AccessPolicy | None = None,
) -> FastAPI:
    """Build the app. Pass a ready ``copilot`` in tests; otherwise ``loader`` runs at startup.

    When ``web_dir`` holds a built frontend, it is served at ``/`` next to the API.
    ``access`` enables rate limiting and API keys; without it every request is allowed.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, "copilot", None) is None:
            started = time.perf_counter()
            app.state.copilot = loader()
            logger.info("Models loaded in %.1f s", time.perf_counter() - started)
        yield

    app = FastAPI(
        title="ResolveIQ API",
        version=__version__,
        description=(
            "Drafts a grounded reply, predicts the routing queue and scores the escalation "
            "risk of a support ticket. Replies are drafts for an agent to review."
        ),
        lifespan=lifespan,
    )
    app.state.copilot = copilot

    def service(request: Request) -> Copilot:
        return request.app.state.copilot

    @app.middleware("http")
    async def rate_limit(request: Request, call_next: Any) -> Response:
        if access is None or not request.url.path.startswith(LIMITED_PREFIXES):
            return await call_next(request)
        decision = access.check(
            request.headers.get(API_KEY_HEADER),
            request.client.host if request.client else None,
            request.headers.get("X-Forwarded-For"),
        )
        if decision is None:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
        if not decision.allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Too many requests. Try again in {decision.retry_after} seconds."
                },
                headers=decision.headers(),
            )
        response = await call_next(request)
        response.headers.update(decision.headers())
        return response

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id[:64]
        logger.info(
            "%s %s %s %.1f ms id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            request_id[:64],
        )
        return response

    @app.exception_handler(InputError)
    async def input_error(request: Request, exc: InputError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(PayloadTooLargeError)
    async def too_large(request: Request, exc: PayloadTooLargeError) -> JSONResponse:
        return JSONResponse(status_code=413, content={"detail": str(exc)})

    @app.post(
        "/analyze",
        response_model=AnalyzeResponse,
        responses={
            422: {"description": "Empty or too long text"},
            503: {
                "model": AnalyzeResponse,
                "description": "Language model unavailable; category and risk still returned",
            },
        },
    )
    def analyze(body: AnalyzeRequest, request: Request) -> Any:
        result = service(request).analyze(body)
        if result.partial:
            return JSONResponse(status_code=503, content=result.response.model_dump())
        return result.response

    @app.post(
        "/analyze/stream",
        response_class=StreamingResponse,
        responses={
            200: {
                "content": {"text/event-stream": {}},
                "description": (
                    "Server-sent events: analysis (category and risk), sources, token "
                    "(pieces of the draft), done (the complete /analyze response and "
                    "whether it is partial), or error"
                ),
            },
            422: {"description": "Empty or too long text"},
        },
    )
    def analyze_stream(body: AnalyzeRequest, request: Request) -> StreamingResponse:
        # Validation runs here, so bad input still gets a normal 422 response.
        events = service(request).analyze_stream(body)

        def encode() -> Iterator[str]:
            try:
                for name, data in events:
                    yield f"event: {name}\ndata: {json.dumps(data)}\n\n"
            except Exception:
                logger.exception("Streaming analysis failed")
                payload = json.dumps({"detail": "The analysis failed. Try again."})
                yield f"event: error\ndata: {payload}\n\n"

        return StreamingResponse(
            encode(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post(
        "/batch",
        response_model=BatchResponse,
        responses={
            413: {"model": ErrorResponse, "description": "File too large"},
            422: {"model": ErrorResponse, "description": "Missing columns or empty rows"},
        },
    )
    async def batch(request: Request, file: Annotated[UploadFile, File()]) -> BatchResponse:
        copilot = service(request)
        payload = await file.read(copilot.max_upload_bytes + 1)
        return copilot.batch(payload)

    @app.post(
        "/feedback",
        response_model=FeedbackResponse,
        responses={404: {"model": ErrorResponse, "description": "Unknown request ID"}},
    )
    def feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse:
        stored = service(request).feedback(body.request_id, body.rating, body.comment)
        if not stored:
            raise HTTPException(status_code=404, detail=f"Unknown request ID {body.request_id}")
        return FeedbackResponse(request_id=body.request_id, rating=body.rating, stored=True)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        return service(request).health()

    @app.get("/requests", response_model=list[RequestSummary])
    def recent_requests(
        request: Request, limit: Annotated[int, Query(ge=1, le=500)] = 50
    ) -> list[RequestSummary]:
        """Newest analyzed tickets with their outputs and feedback. Never ticket text."""
        return service(request).recent_requests(limit)

    @app.get("/stats", response_model=StatsResponse)
    def stats(request: Request) -> StatsResponse:
        """Totals over the request log for the monitoring dashboard."""
        return service(request).stats()

    # Mounted last so every API route above takes precedence.
    if web_dir is not None and (web_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return app


app = create_app(access=_default_access())
