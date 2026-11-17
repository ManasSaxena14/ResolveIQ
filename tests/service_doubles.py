"""Small stand-ins for the trained models, used by the service, API and UI tests."""

from __future__ import annotations

from resolveiq.classifier.predict import Prediction
from resolveiq.monitoring.logging import RequestLog
from resolveiq.rag.generate import AnswerEngine
from resolveiq.rag.llm import LLMError
from resolveiq.rag.retrieve import Retriever
from resolveiq.rag.types import Chunk
from resolveiq.risk.predict import RiskScore
from resolveiq.service import Copilot

KB = [
    (
        "billing/double-charges.md",
        "If you were charged twice this month, the second charge is "
        "usually a pending authorization that disappears within three business days.",
    ),
    ("shipping/options.md", "Express shipping takes two business days."),
    ("account/password.md", "The password reset link is valid for thirty minutes."),
]


class StubLLM:
    provider = "stub"
    model = "stub-model"

    def __init__(
        self,
        reply: str = "The second charge is a pending authorization [double-charges.md, page 1].",
    ) -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        return self.reply


class BrokenLLM(StubLLM):
    def complete(self, messages):
        raise LLMError("connection refused")


class KeywordClassifier:
    """Billing for money words, Technical Support otherwise."""

    def predict_ticket(self, body: str, subject: str = "") -> Prediction:
        text = f"{subject} {body}".lower()
        billing = any(word in text for word in ("charge", "refund", "invoice"))
        label = "Billing and Payments" if billing else "Technical Support"
        other = "Technical Support" if billing else "Billing and Payments"
        confidence = 0.9 if billing else 0.55
        return Prediction(
            label=label,
            confidence=confidence,
            needs_review=confidence < 0.7,
            probabilities={
                label: confidence,
                other: round(1 - confidence - 0.05, 2),
                "General Inquiry": 0.05,
            },
        )


class KeywordRiskModel:
    """Escalation probability rises with urgent words."""

    threshold = 0.4

    def _score(self, text: str) -> RiskScore:
        text = text.lower()
        hits = [word for word in ("urgent", "down", "twice", "again") if word in text]
        probability = round(min(0.2 + 0.25 * len(hits), 0.95), 4)
        return RiskScore(
            probability,
            probability >= self.threshold,
            [f'Message mentions "{w}"' for w in hits[:3]],
        )

    def score(self, subject: str, body: str, ticket_type: str = "") -> RiskScore:
        return self._score(f"{subject} {body}")

    def score_texts(self, texts, ticket_types, k: int = 3):
        return [self._score(text) for text in texts]


def make_engine(embedder, vector_store, llm) -> AnswerEngine:
    chunks = [Chunk(f"{src}::p1::c{i}", src, 1, i, text) for i, (src, text) in enumerate(KB)]
    vector_store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
    retriever = Retriever(embedder, vector_store, top_k=2, candidate_k=3)
    return AnswerEngine(retriever, llm, "Altura", 0.2, {"top_k": 2})


def make_copilot(
    embedder,
    vector_store,
    llm=None,
    with_engine: bool = True,
    log: RequestLog | None = None,
    **limits,
) -> Copilot:
    engine = make_engine(embedder, vector_store, llm or StubLLM()) if with_engine else None
    return Copilot(
        answer_engine=engine,
        classifier=KeywordClassifier(),
        risk_model=KeywordRiskModel(),
        versions={"embedder": "hashing", "llm": "stub", "classifier": "keyword", "risk": "keyword"},
        request_log=log if log is not None else RequestLog(":memory:"),
        index_info={"files": 3, "chunks": 3},
        **limits,
    )
