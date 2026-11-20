import csv
import json

import numpy as np
import pytest

from resolveiq.classifier.baseline import train_baseline
from resolveiq.risk.explain import (
    describe_feature,
    global_importance,
    plot_shap_summary,
    shap_values,
    text_model_terms,
    top_reasons,
)
from resolveiq.risk.features import FeatureSchema, feature_matrix, out_of_fold_probabilities
from resolveiq.risk.predict import RiskModel, save_risk_model, score_csv
from resolveiq.risk.train import (
    ProbabilityCalibrator,
    build_logistic_baseline,
    dmatrix,
    grid_candidates,
    scale_pos_weight,
    train_xgboost,
    tune_xgboost,
    xgboost_threads,
)

PARAMS = {
    "n_estimators": 40,
    "learning_rate": 0.3,
    "max_depth": 3,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "early_stopping_rounds": 10,
}


def _tickets(n: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    texts, types, queues, labels = [], [], [], []
    for i in range(n):
        urgent = i % 2 == 0
        # Random filler keeps message length uninformative for both classes.
        filler = " details" * int(rng.integers(0, 8))
        if urgent:
            text = f"URGENT: server outage, system down again!{filler}"
        else:
            text = f"Question about my invoice and payment plan.{filler}"
        texts.append(text)
        types.append("Incident" if urgent else "Request")
        queues.append("Technical" if urgent else "Billing")
        # Mostly follows the wording, with some label noise.
        labels.append(int(urgent) if rng.random() > 0.05 else int(not urgent))
    return texts, types, queues, np.array(labels)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    texts, types, queues, labels = _tickets(120)
    queue_model = train_baseline(texts, queues)
    text_model = train_baseline(texts, labels.tolist())
    schema = FeatureSchema(queues=tuple(str(c) for c in queue_model.classes_))
    # Out-of-fold text scores, as in scripts/build_risk_features.py.
    text_scores, _ = out_of_fold_probabilities(texts, labels.tolist(), folds=3, seed=0)
    features = feature_matrix(
        texts, types, queue_model.predict_proba(texts), schema, text_scores[:, 1]
    )
    tree = train_xgboost(
        features[:80], labels[:80], features[80:], labels[80:], schema.names, PARAMS
    )
    calibrator = ProbabilityCalibrator("sigmoid").fit(tree.predict(features[80:]), labels[80:])
    out_dir = tmp_path_factory.mktemp("risk")
    meta = {"schema": schema.as_dict(), "threshold": 0.5, "version": "test"}
    save_risk_model(out_dir, tree, calibrator, queue_model, text_model, meta)
    return {
        "dir": out_dir,
        "tree": tree,
        "schema": schema,
        "features": features,
        "labels": labels,
        "texts": texts,
        "types": types,
        "text_model": text_model,
    }


def test_scale_pos_weight():
    assert scale_pos_weight([0, 0, 0, 1]) == 3.0
    with pytest.raises(ValueError, match="no escalations"):
        scale_pos_weight([0, 0])


def test_grid_candidates_cover_every_combination():
    candidates = grid_candidates({"max_depth": 5, "eta": 0.1}, {"max_depth": [3, 4], "x": [1, 2]})

    assert len(candidates) == 4
    assert {c["max_depth"] for c in candidates} == {3, 4}
    assert all(c["eta"] == 0.1 for c in candidates)


def test_xgboost_learns_and_is_cut_to_best_iteration(trained):
    tree = trained["tree"]

    assert tree.val_pr_auc > 0.8
    assert tree.booster.num_boosted_rounds() == tree.best_iteration + 1
    assert tree.params["n_estimators"] == tree.best_iteration + 1


def test_tuning_keeps_the_best_trial(trained):
    features, labels, schema = trained["features"], trained["labels"], trained["schema"]

    result = tune_xgboost(
        features[:80],
        labels[:80],
        features[80:],
        labels[80:],
        schema.names,
        PARAMS,
        {"max_depth": [2, 3]},
        on_trial=lambda number, trial: None,
    )

    assert len(result.trials) == 2
    assert result.best.val_pr_auc == pytest.approx(
        max(t["val_pr_auc"] for t in result.trials), abs=1e-4
    )
    with pytest.raises(ValueError, match="empty"):
        tune_xgboost(
            features[:80], labels[:80], features[80:], labels[80:], schema.names, PARAMS, {"a": []}
        )


def test_logistic_baseline_scores(trained):
    model = build_logistic_baseline().fit(trained["features"], trained["labels"])

    assert model.predict_proba(trained["features"]).shape == (120, 2)


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibrators_are_monotone_probabilities(method):
    scores = np.linspace(0, 1, 50)
    labels = (scores + np.random.default_rng(0).normal(0, 0.2, 50) > 0.5).astype(int)

    calibrated = ProbabilityCalibrator(method).fit(scores, labels).transform(scores)

    assert np.all((calibrated >= 0) & (calibrated <= 1))
    assert np.all(np.diff(calibrated) >= -1e-9)


def test_calibrator_errors():
    with pytest.raises(ValueError, match="Unknown"):
        ProbabilityCalibrator("histogram")
    with pytest.raises(RuntimeError, match="not fitted"):
        ProbabilityCalibrator().transform(np.array([0.5]))


def test_shap_values_add_up_to_the_margin(trained):
    tree, features, schema = trained["tree"], trained["features"], trained["schema"]
    contributions = shap_values(tree.booster, features, schema.names)
    margin = tree.booster.predict(dmatrix(features, schema.names), output_margin=True)

    assert contributions.shape == (120, len(schema.names) + 1)
    assert np.allclose(contributions.sum(axis=1), margin, atol=1e-4)


def test_top_reasons_are_positive_and_readable(trained):
    tree, features, schema = trained["tree"], trained["features"], trained["schema"]
    contributions = shap_values(tree.booster, features, schema.names)
    terms = text_model_terms(trained["text_model"], trained["texts"][0])

    reasons = top_reasons(
        contributions[0], features[0], schema, trained["texts"][0], "Incident", text_terms=terms
    )

    assert 1 <= len(reasons) <= 3
    assert all(r.contribution >= 0.05 for r in reasons)
    assert len({r.text for r in reasons}) == len(reasons)
    words = trained["texts"][0].lower()
    assert terms and all(word in words for term in terms for word in term.split())


def test_describe_feature_sentences():
    schema = FeatureSchema(queues=("Billing and Payments",))
    text = "Please refund and cancel now!"

    assert describe_feature("kw_money", 1, text, "", schema) == 'Message mentions "refund"'
    assert describe_feature("kw_legal", 0, text, "", schema).startswith("Message does not")
    assert describe_feature("type_incident", 1, text, "Incident", schema) == (
        "Ticket type is Incident"
    )
    assert describe_feature("queue_billing_and_payments", 0.9, text, "", schema) == (
        "Predicted queue is Billing and Payments (90%)"
    )
    assert "Low chance" in describe_feature("queue_billing_and_payments", 0.1, text, "", schema)
    assert describe_feature("text_risk_score", 0.8, text, "", schema, ["refund"]).endswith(
        '"refund"'
    )
    assert "text score 0.80" in describe_feature("text_risk_score", 0.8, text, "", schema)
    assert describe_feature("exclamation_count", 1, text, "", schema) == "1 exclamation mark"
    assert describe_feature("question_count", 2, text, "", schema) == "2 question marks"
    assert describe_feature("char_count", 900, text, "", schema).startswith("Long message")
    assert describe_feature("word_count", 12, text, "", schema) == "Message has 12 words"
    assert describe_feature("caps_ratio", 0.5, text, "", schema) == "50% of letters are capitals"
    assert describe_feature("sentiment_compound", -0.6, text, "", schema).startswith("Negative")
    assert describe_feature("sentiment_negative", 0.25, text, "", schema).startswith("25%")


def test_global_importance_and_summary_plot(trained, tmp_path):
    tree, features, schema = trained["tree"], trained["features"], trained["schema"]
    contributions = shap_values(tree.booster, features, schema.names)

    importance = global_importance(contributions, schema.names)
    plot_shap_summary(contributions, features, schema.names, tmp_path / "shap.png")

    assert [v for _, v in importance] == sorted((v for _, v in importance), reverse=True)
    assert (tmp_path / "shap.png").stat().st_size > 0


def test_saved_model_scores_raw_tickets(trained):
    model = RiskModel.load(trained["dir"])

    urgent = model.score("URGENT: server outage", "The system is down again!", "Incident")
    calm = model.score("Invoice question", "A question about my payment plan, thank you")

    assert urgent.escalation_probability > calm.escalation_probability
    assert urgent.alert is True
    assert urgent.reasons and len(urgent.reasons) <= 3
    assert set(urgent.as_dict()) == {"escalation_probability", "alert", "reasons"}
    assert model.score_texts([], []) == []


def test_load_rejects_missing_or_stale_models(trained, tmp_path):
    with pytest.raises(FileNotFoundError, match="Train the risk model"):
        RiskModel.load(tmp_path)

    meta = json.loads((trained["dir"] / "risk_meta.json").read_text())
    meta["schema"]["feature_names"] = ["old"]
    stale = tmp_path / "stale"
    stale.mkdir()
    for path in trained["dir"].iterdir():
        (stale / path.name).write_bytes(path.read_bytes())
    (stale / "risk_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="rebuild"):
        RiskModel.load(stale)


def test_text_model_is_required_when_schema_uses_it(trained):
    with pytest.raises(ValueError, match="text model"):
        RiskModel(None, None, None, trained["schema"], 0.5, text_model=None)


def test_score_csv_ranks_by_risk(trained, tmp_path):
    model = RiskModel.load(trained["dir"])
    source = tmp_path / "tickets.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "subject", "body", "type"])
        writer.writeheader()
        writer.writerow(
            {"id": "calm", "subject": "Invoice", "body": "Question about my payment plan, thanks"}
        )
        writer.writerow(
            {"id": "hot", "subject": "URGENT outage", "body": "Server down!", "type": "Incident"}
        )
    output = tmp_path / "ranked.csv"

    assert score_csv(model, source, output) == 2
    rows = list(csv.DictReader(output.open()))
    assert [row["id"] for row in rows] == ["hot", "calm"]
    assert rows[0]["rank"] == "1"
    assert "body" not in rows[0]


def test_score_csv_requires_text_columns(trained, tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text("id,message\n1,hello\n")

    with pytest.raises(ValueError, match="subject"):
        score_csv(RiskModel.load(trained["dir"]), source, tmp_path / "out.csv")


def test_xgboost_uses_one_thread_next_to_pytorch_on_macos(monkeypatch):
    monkeypatch.setattr("resolveiq.risk.train.sys.platform", "darwin")
    monkeypatch.setitem(__import__("sys").modules, "torch", object())
    assert xgboost_threads() == 1

    monkeypatch.setattr("resolveiq.risk.train.sys.platform", "linux")
    assert xgboost_threads() is None


def test_word_weights_explain_the_text_score_in_both_directions(trained):
    model = RiskModel.load(trained["dir"])

    urgent = dict(model.word_weights("URGENT outage", "The server is down again!"))
    calm = dict(model.word_weights("Invoice", "A question about my payment plan"))

    assert urgent and max(urgent.values()) > 0
    assert calm and min(calm.values()) < 0, "calm wording lowers the score"
    magnitudes = [abs(v) for v in urgent.values()]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_word_weights_need_a_text_model(trained):
    model = RiskModel.load(trained["dir"])
    model.text_model = None

    assert model.word_weights("a", "b") == []
