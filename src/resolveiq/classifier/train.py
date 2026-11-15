"""Fine-tune a transformer to classify tickets into queues.

A plain PyTorch training loop is used instead of the Hugging Face Trainer so
every step (class-weighted loss, warmup schedule, length bucketing, early
stopping on validation macro F1) is visible and easy to explain.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from resolveiq.classifier.metrics import classification_metrics

logger = logging.getLogger(__name__)

META_FILENAME = "classifier_meta.json"


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    max_length: int
    learning_rate: float
    batch_size: int
    epochs: int
    weight_decay: float
    warmup_ratio: float
    class_weights: bool
    seed: int
    patience: int = 1
    eval_batch_size: int = 64


@dataclass
class TrainingResult:
    output_dir: Path
    best_epoch: int
    best_val_macro_f1: float
    history: list[dict[str, float]] = field(default_factory=list)
    train_seconds: float = 0.0
    device: str = "cpu"


def select_device(preferred: str | None = None):
    """CUDA if available, then Apple Silicon (MPS), then CPU."""
    import torch

    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def balanced_class_weights(label_ids: Sequence[int], num_labels: int) -> np.ndarray:
    """Weights inversely proportional to class frequency (sklearn's 'balanced')."""
    counts = np.bincount(np.asarray(label_ids), minlength=num_labels).astype(float)
    counts[counts == 0] = 1.0
    return len(label_ids) / (num_labels * counts)


def length_bucketed_batches(
    lengths: Sequence[int], batch_size: int, rng: random.Random, bucket_factor: int = 50
) -> list[list[int]]:
    """Shuffled batches of similar-length examples, to minimize padding.

    Indices are shuffled, cut into large buckets, sorted by length inside each
    bucket and split into batches. The batch order is shuffled again so the
    model does not see lengths in a fixed order.
    """
    indices = list(range(len(lengths)))
    rng.shuffle(indices)
    bucket = batch_size * bucket_factor
    batches: list[list[int]] = []
    for start in range(0, len(indices), bucket):
        chunk = sorted(indices[start : start + bucket], key=lambda i: lengths[i])
        batches.extend(chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size))
    rng.shuffle(batches)
    return batches


def encode(tokenizer: Any, texts: Sequence[str], max_length: int) -> list[list[int]]:
    """Token ids without padding; padding is applied per batch."""
    return tokenizer(list(texts), truncation=True, max_length=max_length)["input_ids"]


def _collate(tokenizer: Any, encoded: Sequence[list[int]], device: Any) -> dict[str, Any]:
    batch = tokenizer.pad({"input_ids": list(encoded)}, return_tensors="pt")
    return {key: value.to(device) for key, value in batch.items()}


def predict_proba(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int,
    device: Any,
    batch_size: int = 64,
) -> np.ndarray:
    """Class probabilities for ``texts``, in the model's label order."""
    import torch

    encoded = encode(tokenizer, texts, max_length)
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    probabilities = np.zeros((len(encoded), model.config.num_labels), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch = _collate(tokenizer, [encoded[i] for i in idx], device)
            logits = model(**batch).logits.float()
            probabilities[idx] = torch.softmax(logits, dim=-1).cpu().numpy()
    return probabilities


def train_classifier(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    val_texts: Sequence[str],
    val_labels: Sequence[str],
    labels: Sequence[str],
    config: TrainConfig,
    output_dir: Path,
    device: Any = None,
    model: Any = None,
    tokenizer: Any = None,
    on_epoch: Callable[[dict[str, float]], None] | None = None,
) -> TrainingResult:
    """Fine-tune and save the checkpoint with the best validation macro F1.

    ``model`` and ``tokenizer`` default to the pre-trained ``config.model_name``;
    tests pass small randomly initialized ones instead.
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    set_seed(config.seed)
    device = device or select_device()
    label_to_id = {label: i for i, label in enumerate(labels)}
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.model_name)
    model = model or AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=label_to_id,
    )
    model.to(device)

    train_ids = encode(tokenizer, train_texts, config.max_length)
    train_y = [label_to_id[label] for label in train_labels]
    rng = random.Random(config.seed)
    steps_per_epoch = -(-len(train_ids) // config.batch_size)
    total_steps = steps_per_epoch * config.epochs

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    parameter_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(parameter_groups, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(config.warmup_ratio * total_steps), total_steps
    )
    weights = (
        torch.tensor(balanced_class_weights(train_y, len(labels)), dtype=torch.float32)
        if config.class_weights
        else None
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)

    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_f1, best_epoch, epochs_without_gain = -1.0, 0, 0
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_started, running_loss = time.perf_counter(), 0.0
        batches = length_bucketed_batches([len(x) for x in train_ids], config.batch_size, rng)
        for step, idx in enumerate(batches, start=1):
            batch = _collate(tokenizer, [train_ids[i] for i in idx], device)
            targets = torch.tensor([train_y[i] for i in idx], device=device)
            loss = loss_fn(model(**batch).logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            if step % 100 == 0:
                logger.info("epoch %d step %d/%d loss %.4f", epoch, step, len(batches), loss.item())

        probabilities = predict_proba(
            model, tokenizer, val_texts, config.max_length, device, config.eval_batch_size
        )
        predictions = [labels[i] for i in probabilities.argmax(axis=1)]
        metrics = classification_metrics(val_labels, predictions, labels)
        record = {
            "epoch": epoch,
            "train_loss": round(running_loss / max(1, len(batches)), 4),
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "epoch_seconds": round(time.perf_counter() - epoch_started, 1),
        }
        history.append(record)
        logger.info("epoch %d: %s", epoch, record)
        if on_epoch:
            on_epoch(record)

        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch, epochs_without_gain = metrics["macro_f1"], epoch, 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            epochs_without_gain += 1
            if epochs_without_gain > config.patience:
                logger.info("Stopping early: no improvement for %d epochs", epochs_without_gain)
                break

    result = TrainingResult(
        output_dir=output_dir,
        best_epoch=best_epoch,
        best_val_macro_f1=best_f1,
        history=history,
        train_seconds=round(time.perf_counter() - started, 1),
        device=str(device),
    )
    write_meta(
        output_dir,
        {
            "labels": list(labels),
            "max_length": config.max_length,
            "train_config": asdict(config),
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_f1,
            "history": history,
            "train_seconds": result.train_seconds,
            "device": result.device,
        },
    )
    return result


def read_meta(model_dir: Path) -> dict[str, Any]:
    path = model_dir / META_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Train the classifier first.")
    return json.loads(path.read_text(encoding="utf-8"))


def write_meta(model_dir: Path, meta: dict[str, Any]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def update_meta(model_dir: Path, **values: Any) -> None:
    meta = read_meta(model_dir)
    meta.update(values)
    write_meta(model_dir, meta)
