"""Answer a support question from the knowledge base."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from resolveiq.config import Settings
from resolveiq.rag.llm import LLMClient, create_llm_client, stream_text
from resolveiq.rag.prompts import REFUSAL_MESSAGE, build_messages, is_refusal
from resolveiq.rag.retrieve import RetrievalResult, Retriever
from resolveiq.rag.types import RetrievedChunk

MAX_QUESTION_CHARS = 4000


@dataclass(frozen=True)
class Source:
    file: str
    page: int
    score: float | None


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    refused: bool
    sources: list[Source]
    timings_ms: dict[str, float] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    """One source per (file, page), in retrieval order, with the best similarity."""
    best: dict[tuple[str, int], float | None] = {}
    for item in chunks:
        key = (item.chunk.source, item.chunk.page)
        current = best.get(key)
        similarity = item.similarity
        if key not in best or (
            similarity is not None and (current is None or similarity > current)
        ):
            best[key] = similarity
    return [
        Source(file, page, None if score is None else round(score, 4))
        for (file, page), score in best.items()
    ]


class AnswerEngine:
    """Retrieve, apply the refusal rule, then ask the language model for a cited answer."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient | None,
        company_name: str,
        refusal_threshold: float,
        settings_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.company_name = company_name
        self.refusal_threshold = refusal_threshold
        self.settings_snapshot = settings_snapshot or {}

    @classmethod
    def from_settings(cls, settings: Settings, with_llm: bool = True) -> AnswerEngine:
        rag, llm = settings.rag, settings.llm
        snapshot = {
            "embedding_model": rag.embedding_model,
            "chunk_tokens": rag.chunk_tokens,
            "chunk_overlap": rag.chunk_overlap,
            "top_k": rag.top_k,
            "hybrid": rag.hybrid,
            "rerank": rag.rerank,
            "refusal_threshold": rag.refusal_threshold,
            "llm_provider": llm.provider if with_llm else None,
            "llm_model": llm.active.model if with_llm else None,
        }
        return cls(
            retriever=Retriever.from_settings(settings),
            llm=create_llm_client(llm) if with_llm else None,
            company_name=settings.project.company_name,
            refusal_threshold=rag.refusal_threshold,
            settings_snapshot=snapshot,
        )

    def _retrieve(self, question: str) -> tuple[RetrievalResult, float]:
        question = question.strip()
        if not question:
            raise ValueError("The question is empty")
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"The question is longer than {MAX_QUESTION_CHARS} characters")
        started = time.perf_counter()
        result = self.retriever.retrieve(question)
        return result, _elapsed_ms(started)

    def _is_out_of_scope(self, retrieval: RetrievalResult) -> bool:
        return not retrieval.chunks or retrieval.best_similarity < self.refusal_threshold

    def retrieve_only(self, question: str) -> AnswerResult:
        """Run retrieval and the refusal rule without calling the language model.

        Used to tune retrieval settings quickly. ``answer`` is left empty.
        """
        retrieval, retrieval_ms = self._retrieve(question)
        refused = self._is_out_of_scope(retrieval)
        return AnswerResult(
            answer="",
            refused=refused,
            sources=[] if refused else collect_sources(retrieval.chunks),
            timings_ms={"retrieval": retrieval_ms, **retrieval.timings_ms},
            settings={**self.settings_snapshot, "best_similarity": retrieval.best_similarity},
        )

    def stream(self, question: str) -> Iterator[tuple[str, Any]]:
        """Answer while the language model writes.

        Yields ``("sources", (sources, timings))`` once retrieval passes the refusal
        rule, then ``("token", text)`` for each piece of the answer, and finally
        ``("result", AnswerResult)``. Out-of-scope questions yield only the result.
        """
        if self.llm is None:
            raise RuntimeError("AnswerEngine was created without a language model client")
        retrieval, retrieval_ms = self._retrieve(question)
        timings = {"retrieval": retrieval_ms}
        settings = {**self.settings_snapshot, "best_similarity": retrieval.best_similarity}

        if self._is_out_of_scope(retrieval):
            yield "result", AnswerResult(REFUSAL_MESSAGE, True, [], timings, settings)
            return

        sources = collect_sources(retrieval.chunks)
        yield "sources", (sources, dict(timings))
        messages = build_messages(question, retrieval.chunks, self.company_name)
        started = time.perf_counter()
        pieces = []
        for piece in stream_text(self.llm, messages):
            pieces.append(piece)
            yield "token", piece
        timings["generation"] = _elapsed_ms(started)
        text = "".join(pieces).strip()

        if is_refusal(text):
            yield "result", AnswerResult(REFUSAL_MESSAGE, True, [], timings, settings)
            return
        yield "result", AnswerResult(text, False, sources, timings, settings)

    def answer(self, question: str) -> AnswerResult:
        result: AnswerResult | None = None
        for kind, value in self.stream(question):
            if kind == "result":
                result = value
        assert result is not None
        return result


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
