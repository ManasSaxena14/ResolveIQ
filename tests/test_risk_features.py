import numpy as np
import pytest

from resolveiq.risk.features import (
    KEYWORD_GROUPS,
    LEAKAGE_COLUMNS,
    TEXT_FEATURES,
    TEXT_SCORE_FEATURE,
    TICKET_TYPES,
    FeatureSchema,
    feature_matrix,
    keyword_matches,
    label_from_priority,
    load_feature_schema,
    load_feature_split,
    out_of_fold_probabilities,
    queue_slug,
    text_statistics,
    ticket_features,
)

SCHEMA = FeatureSchema(queues=("Billing and Payments", "Technical Support"), text_score=False)


def test_label_uses_configured_values_only():
    assert label_from_priority("high", ["high"]) == 1
    assert label_from_priority(" High ", ["high"]) == 1
    assert label_from_priority("medium", ["high"]) == 0
    assert label_from_priority("critical", ["high", "critical"]) == 1


def test_no_leakage_column_is_a_feature():
    names = set(SCHEMA.names)
    for column in LEAKAGE_COLUMNS:
        assert column not in names
        assert not any(name == f"kw_{column}" for name in names)


def test_schema_order_and_size():
    names = SCHEMA.names

    assert names[: len(TEXT_FEATURES)] == list(TEXT_FEATURES)
    assert names[-2:] == ["queue_billing_and_payments", "queue_technical_support"]
    assert len(names) == len(TEXT_FEATURES) + len(KEYWORD_GROUPS) + len(TICKET_TYPES) + 2
    assert SCHEMA.queue_for_feature("queue_technical_support") == "Technical Support"
    assert SCHEMA.queue_for_feature("char_count") is None


def test_keywords_match_whole_words_once():
    matches = keyword_matches("URGENT: the server is down, down again. Please refund me.")

    assert matches["urgency"] == ["urgent"]
    assert matches["outage"] == ["down"]
    assert matches["frustration"] == ["again"]
    assert matches["money"] == ["refund"]
    assert keyword_matches("Download the stillness guide")["outage"] == []
    assert keyword_matches("Download the stillness guide")["frustration"] == []


def test_multi_word_phrases_win_over_parts():
    assert keyword_matches("It has stopped working")["outage"] == ["stopped working"]


def test_text_statistics():
    stats = text_statistics("HELP now! Why?")

    assert stats["char_count"] == 14
    assert stats["word_count"] == 3
    assert stats["exclamation_count"] == 1
    assert stats["question_count"] == 1
    assert stats["caps_ratio"] == pytest.approx(5 / 10)
    assert -1 <= stats["sentiment_compound"] <= 1


def test_negative_text_has_negative_sentiment():
    angry = text_statistics("This is terrible and unacceptable, I am very angry")
    happy = text_statistics("Thank you, this is great and works perfectly")

    assert angry["sentiment_compound"] < 0 < happy["sentiment_compound"]


def test_ticket_features_follow_schema():
    row = ticket_features("Urgent outage!", "Incident", [0.2, 0.8], SCHEMA)
    named = dict(zip(SCHEMA.names, row, strict=True))

    assert named["kw_urgency"] == 1
    assert named["kw_outage"] == 1
    assert named["type_incident"] == 1
    assert named["type_request"] == 0
    assert named["queue_technical_support"] == pytest.approx(0.8)


def test_text_score_is_the_last_feature():
    schema = FeatureSchema(queues=SCHEMA.queues)
    row = ticket_features("Server down", "Incident", [0.5, 0.5], schema, text_score=0.9)

    assert schema.names[-1] == TEXT_SCORE_FEATURE
    assert row[-1] == pytest.approx(0.9)
    with pytest.raises(ValueError, match="text score"):
        ticket_features("Server down", "Incident", [0.5, 0.5], schema)


def test_ticket_features_reject_wrong_queue_count():
    with pytest.raises(ValueError, match="queue probabilities"):
        ticket_features("text", "Incident", [1.0], SCHEMA)


def test_feature_matrix_shape_and_unknown_type():
    matrix = feature_matrix(
        ["a", "b", "c"], ["Incident", "", "Unknown"], np.full((3, 2), 0.5), SCHEMA
    )

    assert matrix.shape == (3, len(SCHEMA.names))
    assert matrix.dtype == np.float32
    type_columns = [SCHEMA.names.index(f"type_{t.lower()}") for t in TICKET_TYPES]
    assert matrix[1, type_columns].sum() == 0


def test_queue_slug():
    assert queue_slug("Sales and Pre-Sales") == "sales_and_pre_sales"


def test_out_of_fold_probabilities_cover_every_ticket():
    texts = [f"invoice payment refund {i}" for i in range(12)] + [
        f"server crash error {i}" for i in range(12)
    ]
    labels = ["Billing"] * 12 + ["Technical"] * 12

    probabilities, queues = out_of_fold_probabilities(texts, labels, folds=3, seed=0)

    assert queues == ["Billing", "Technical"]
    assert probabilities.shape == (24, 2)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert probabilities[:12, 0].mean() > 0.5


def test_feature_files_round_trip(tmp_path):
    import json

    features = np.ones((2, len(SCHEMA.names)), dtype=np.float32)
    np.savez_compressed(
        tmp_path / "train.npz",
        features=features,
        labels=np.array([0, 1], dtype=np.int8),
        ticket_ids=np.array(["a", "b"]),
    )
    (tmp_path / "schema.json").write_text(json.dumps(SCHEMA.as_dict()))

    loaded, labels, ids = load_feature_split(tmp_path, "train")
    assert loaded.shape == features.shape
    assert labels.tolist() == [0, 1]
    assert ids == ["a", "b"]
    assert load_feature_schema(tmp_path) == SCHEMA


def test_missing_feature_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_risk_features"):
        load_feature_split(tmp_path, "train")
    with pytest.raises(FileNotFoundError, match="build_risk_features"):
        load_feature_schema(tmp_path)


def test_stale_schema_is_rejected(tmp_path):
    import json

    (tmp_path / "schema.json").write_text(
        json.dumps({"queues": list(SCHEMA.queues), "feature_names": ["old"]})
    )
    with pytest.raises(ValueError, match="rebuild"):
        load_feature_schema(tmp_path)
