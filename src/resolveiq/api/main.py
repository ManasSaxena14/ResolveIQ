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

import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from resolveiq import __version__
from resolveiq.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
)
from resolveiq.service import Copilot, InputError, PayloadTooLargeError

logger = logging.getLogger("resolveiq.api")

REQUEST_ID_HEADER = "X-Request-ID"


def _load_from_settings() -> Copilot:
    from resolveiq.config import get_settings

    return Copilot.from_settings(get_settings())


def create_app(
    copilot: Copilot | None = None, loader: Callable[[], Copilot] = _load_from_settings
) -> FastAPI:
    """Build the app. Pass a ready ``copilot`` in tests; otherwise ``loader`` runs at startup."""

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

    return app


app = create_app()
