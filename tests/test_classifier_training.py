"""Training and prediction tests with a tiny, randomly initialized DistilBERT.

The model has one layer and a 32-dimensional hidden size, and the tokenizer is
built from a handful of words, so these tests run on CPU in seconds and never
download pre-trained weights.
"""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from resolveiq.classifier.predict import TicketClassifier  # noqa: E402
from resolveiq.classifier.train import (  # noqa: E402
    TrainConfig,
    balanced_class_weights,
    length_bucketed_batches,
    read_meta,
    select_device,
    train_classifier,
    update_meta,
)

LABELS = ["billing", "outage"]
WORDS = [
    "invoice",
    "charge",
    "payment",
    "refund",
    "server",
    "down",
    "offline",
    "outage",
    "please",
    "help",
]


@pytest.fixture
def tiny_model_and_tokenizer(tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = transformers.DistilBertTokenizerFast(vocab_file=str(vocab_file))
    config = transformers.DistilBertConfig(
        vocab_size=len(vocab),
        dim=32,
        n_layers=1,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=64,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    torch.manual_seed(0)
    return transformers.DistilBertForSequenceClassification(config), tokenizer


def _data(n: int = 40):
    rng = random.Random(0)
    texts, labels = [], []
    for i in range(n):
        label = LABELS[i % 2]
        pool = WORDS[:4] if label == "billing" else WORDS[4:8]
        texts.append(" ".join(rng.choice(pool) for _ in range(6)) + " please help")
        labels.append(label)
    return texts, labels


def _config(**overrides) -> TrainConfig:
    values = dict(
        model_name="tiny",
        max_length=32,
        learning_rate=5e-3,
        batch_size=8,
        epochs=3,
        weight_decay=0.0,
        warmup_ratio=0.1,
        class_weights=True,
        seed=1,
        patience=5,
    )
    values.update(overrides)
    return TrainConfig(**values)


def test_balanced_class_weights():
    weights = balanced_class_weights([0, 0, 0, 1], num_labels=3)

    assert weights.tolist() == pytest.approx([4 / 9, 4 / 3, 4 / 3])


def test_length_bucketed_batches_cover_every_index_once():
    lengths = [random.Random(i).randint(1, 50) for i in range(103)]

    batches = length_bucketed_batches(lengths, batch_size=8, rng=random.Random(0), bucket_factor=2)

    flat = [i for batch in batches for i in batch]
    assert sorted(flat) == list(range(103))
    assert all(len(batch) <= 8 for batch in batches)


def test_select_device_honours_preference():
    assert str(select_device("cpu")) == "cpu"


def test_training_saves_best_checkpoint_and_predicts(tmp_path, tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    texts, labels = _data()
    epochs_seen = []

    result = train_classifier(
        texts,
        labels,
        texts[:10],
        labels[:10],
        LABELS,
        _config(),
        tmp_path / "model",
        device=torch.device("cpu"),
        model=model,
        tokenizer=tokenizer,
        on_epoch=epochs_seen.append,
    )

    assert [record["epoch"] for record in result.history] == [1, 2, 3]
    assert len(epochs_seen) == 3
    assert result.best_val_macro_f1 == max(r["val_macro_f1"] for r in result.history)
    meta = read_meta(tmp_path / "model")
    assert meta["labels"] == LABELS
    assert meta["best_epoch"] == result.best_epoch

    update_meta(tmp_path / "model", review_threshold=0.99)
    classifier = TicketClassifier.load(tmp_path / "model", device="cpu")
    predictions = classifier.predict(["server down offline outage", "invoice refund charge"])

    assert [p.label in LABELS for p in predictions] == [True, True]
    for prediction in predictions:
        assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
        assert prediction.needs_review == (prediction.confidence < 0.99)
    single = classifier.predict_ticket("refund my invoice", subject="Billing question")
    assert single.label in LABELS


def test_training_stops_early_without_improvement(tmp_path, tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    texts, labels = _data(16)

    result = train_classifier(
        texts,
        labels,
        texts[:4],
        labels[:4],
        LABELS,
        _config(learning_rate=0.0, epochs=5, patience=1),
        tmp_path / "model",
        device=torch.device("cpu"),
        model=model,
        tokenizer=tokenizer,
    )

    assert len(result.history) == 3
    assert result.best_epoch == 1


def test_empty_ticket_is_rejected(tmp_path, tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    classifier = TicketClassifier(model, tokenizer, LABELS, 32, 0.5, torch.device("cpu"))

    with pytest.raises(ValueError):
        classifier.predict(["   "])


def test_missing_meta_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train the classifier"):
        read_meta(tmp_path)


def test_class_weights_can_be_disabled(tmp_path, tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    texts, labels = _data(8)

    result = train_classifier(
        texts, labels, texts, labels, LABELS, _config(epochs=1, class_weights=False),
        tmp_path / "model", device=torch.device("cpu"), model=model, tokenizer=tokenizer,
    )  # fmt: skip

    assert np.isfinite(result.history[0]["train_loss"])


def test_word_weights_are_whole_words_scaled_to_one(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    classifier = TicketClassifier(model, tokenizer, LABELS, 32, 0.5, torch.device("cpu"))

    weights = classifier.word_weights("server down, please refund the charge", "outage")

    assert weights, "a ticket with known words gets highlights"
    assert weights[0][1] == 1.0
    assert all(0 < weight <= 1.0 for _, weight in weights)
    assert all(word.isalpha() and word not in {"please", "the"} for word, _ in weights)
    assert all(p.grad is None for p in model.parameters()), "gradients are cleared"
    assert classifier.word_weights("", "") == []


def test_word_scores_merge_pieces_and_skip_noise():
    from resolveiq.classifier.predict import word_scores

    text = "Refunding the charge, refunding"
    # [CLS] refund ##ing the charge , refund ##ing [SEP]
    offsets = [(0, 0), (0, 6), (6, 9), (10, 13), (14, 20), (20, 21), (22, 28), (28, 31), (0, 0)]
    scores = [9.0, 0.5, 0.25, 5.0, 2.0, 3.0, 0.25, 0.25, 9.0]

    weights = dict(word_scores(text, offsets, scores))

    # charge 2.0 is the strongest; both "refunding" add up to 1.25; "the" and "," drop.
    assert weights == {"charge": 1.0, "refunding": 0.625}
    assert word_scores(text, offsets, [-1.0] * len(scores)) == []
