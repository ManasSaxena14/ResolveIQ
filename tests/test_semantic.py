from types import SimpleNamespace

import numpy as np
import pytest

from resolveiq.risk.predict import RiskScore
from resolveiq.semantic import (
    SemanticClassifier,
    SemanticModel,
    SemanticParams,
    SemanticRisk,
    TextEncoder,
    content_words,
    semantic_ready,
    soft_share,
    without_word,
)

LABELS = ["Billing", "Technical"]
AXES = {"refund": 0, "server": 1, "fire": 2}


class WordEmbedder:
    """Unit vectors along one axis per known word, so similarities are easy to reason about."""

    def __init__(self):
        self.seen = []

    def embed_documents(self, texts):
        self.seen.extend(texts)
        rows = np.full((len(texts), 4), 0.0, dtype=np.float32)
        for row, text in zip(rows, texts, strict=True):
            for word, axis in AXES.items():
                row[axis] += text.lower().count(word)
            if not row.any():
                row[3] = 1.0
            row /= np.linalg.norm(row)
        return rows


def make_model(**changes):
    params = SemanticParams(
        labels=LABELS,
        neighbours=3,
        familiar_from=0.8,
        familiar_full=0.9,
        review_familiar=0.5,
        review_unfamiliar=0.95,
        risk_stack=(0.5, 1.0, 0.0),
        risk_threshold=0.2,
        examples=[
            {"text": "fire in the kitchen", "queue": "Technical", "urgent": True},
            {"text": "a calm question", "queue": "Billing", "urgent": False},
        ],
        version="semantic-test",
        **changes,
    )
    return SemanticModel(
        params,
        past=np.array([[1, 0, 0, 0], [0.99, 0.14, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
        past_queue=np.array([0, 0, 1]),
        past_high=np.array([0, 0, 1]),
        examples=np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
        queue_weights=np.array([[0, 0, -2, 2, 0], [0, 0, 2, -2, 0]], dtype=np.float32),
        urgency_weights=np.array([0, 0, 6, -6, 0], dtype=np.float32),
    )


class StubRisk:
    version = "xgboost-test"

    def score_texts(self, texts, ticket_types, k=3):
        return [RiskScore(0.6, True, ["From the risk model"]) for _ in texts]

    def word_weights(self, subject, body, k=8):
        return [("stub", 1.0)]


def pair(**changes):
    encoder = TextEncoder(WordEmbedder())
    model = make_model(**changes)
    return SemanticClassifier(model, encoder), SemanticRisk(model, encoder, StubRisk())


def test_familiar_tickets_follow_the_similar_past_tickets():
    classifier, risk = pair()

    prediction = classifier.predict_ticket("Please refund me")
    assert prediction.label == "Billing"
    assert prediction.confidence > 0.95
    assert not prediction.needs_review
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=0.01)

    server = risk.score("", "server is slow")
    assert server.alert
    assert server.reasons[0].endswith("most similar past tickets were high priority")
    assert "From the risk model" in server.reasons
    calm = risk.score("", "refund")
    assert calm.escalation_probability < server.escalation_probability
    assert calm.reasons == ["From the risk model"]


def test_unfamiliar_tickets_use_the_retail_models_and_quote_an_urgent_example():
    classifier, risk = pair()

    fire = classifier.predict_ticket("fire")
    assert fire.label == "Technical"
    assert fire.confidence == pytest.approx(0.982, abs=0.01)
    score = risk.score("", "fire")
    assert score.alert and score.escalation_probability > 0.9
    assert score.reasons == ['Close in meaning to an urgent example: "fire in the kitchen"']
    assert score.details == []

    calm = risk.score("", "opening hours")
    assert not calm.alert and calm.reasons == []


def test_review_depends_on_how_familiar_the_ticket_is():
    classifier, _ = pair(queue_scale=0.2)

    assert classifier.predict_ticket("fire").needs_review
    assert not classifier.predict_ticket("refund").needs_review


def test_the_gate_blends_between_the_limits():
    model = make_model()

    assert model.familiarity_gate(np.array([0.7, 0.85, 0.95])).tolist() == pytest.approx(
        [0.0, 0.5, 1.0]
    )
    share = soft_share(np.array([0.0, 0.5, 1.0]))
    assert np.isfinite(share).all() and share[1] == pytest.approx(0)


def test_words_are_explained_by_taking_them_out():
    classifier, risk = pair()

    assert classifier.word_weights("there is a fire here")[0] == ("fire", 1.0)
    effects = dict(risk.word_weights("", "there is a fire here"))
    assert effects["fire"] > 1
    assert risk.word_weights("", "refund please") == [("stub", 1.0)]
    assert classifier.word_weights("12 ?") == []


def test_content_words_and_removal():
    assert content_words("The refund, the REFUND and a fee", 5) == ["refund", "fee"]
    assert content_words("tiny enormous middle", 1) == ["enormous"]
    assert without_word("Fire and fireplace, fire!", "fire") == "  and fireplace,  !"


def test_predictions_reject_empty_text_and_batches_accept_nothing():
    classifier, risk = pair()

    with pytest.raises(ValueError, match="must not be empty"):
        classifier.predict(["  "])
    assert risk.score_texts([], []) == []


def test_the_encoder_remembers_recent_texts():
    embedder = WordEmbedder()
    encoder = TextEncoder(embedder, size=2)

    encoder.encode(["a", "b", "a"])
    encoder.encode(["a"])
    assert embedder.seen == ["a", "b"]
    encoder.encode(["c", "d", "e"])
    encoder.encode(["a"])
    assert embedder.seen[-1] == "a"


def test_saving_and_loading_keeps_every_decision(tmp_path):
    assert not semantic_ready(tmp_path)
    make_model().save(tmp_path, {"note": "kept"})
    assert semantic_ready(tmp_path)

    before, after = make_model(), SemanticModel.load(tmp_path)
    vectors = WordEmbedder().embed_documents(["fire", "refund server", "refund"])
    assert after.params == before.params
    assert np.allclose(before.evidence(vectors).queue, after.evidence(vectors).queue, atol=1e-2)


def test_the_service_uses_the_semantic_model_only_when_asked_and_trained(tmp_path, caplog):
    from resolveiq.service import _semantic_pair

    settings = SimpleNamespace(service=SimpleNamespace(decisions="transformer"))
    assert _semantic_pair(settings, tmp_path, WordEmbedder(), StubRisk()) is None
    settings.service.decisions = "semantic"
    assert _semantic_pair(settings, tmp_path, WordEmbedder(), StubRisk()) is None
    assert "No semantic model" in caplog.text
    make_model().save(tmp_path)
    classifier, risk = _semantic_pair(settings, tmp_path, WordEmbedder(), StubRisk(), 16)
    assert classifier.max_words == risk.max_words == 16
    assert risk.version == "semantic-test + xgboost-test"


def test_retail_data_is_complete_and_the_test_set_is_not_in_it():
    import csv

    from resolveiq.config import load_config

    settings = load_config()
    with settings.resolve(settings.semantic.seed_path).open(encoding="utf-8") as handle:
        seed = list(csv.DictReader(handle))
    with settings.resolve(settings.semantic.testset_path).open(encoding="utf-8") as handle:
        testset = list(csv.DictReader(handle))
    queues = {row["queue"] for row in seed}
    assert len(queues) == 10
    assert all(q in queues for row in testset for q in row["accepted_queues"].split(";"))
    assert not {r["body"] for r in seed} & {r["body"] for r in testset}
    assert len({r["id"] for r in seed}) == len(seed)
    assert {r["split"] for r in testset} == {"dev", "holdout"}
