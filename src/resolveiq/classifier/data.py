"""Loading, cleaning and splitting the labeled ticket dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from resolveiq.privacy import mask_pii

SPLIT_NAMES = ("train", "validation", "test")
LABELS_FILENAME = "labels.json"

_ESCAPED_BREAK = re.compile(r"\\n|<br\s*/?>", re.IGNORECASE)
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_NON_WORD = re.compile(r"\W+")


@dataclass(frozen=True)
class Ticket:
    """One ticket after cleaning. ``label`` is the routing queue."""

    ticket_id: str
    text: str
    label: str
    priority: str
    ticket_type: str


@dataclass(frozen=True)
class CleaningReport:
    rows_read: int
    rows_other_language: int
    rows_empty: int
    duplicates_removed: int
    conflicting_labels_removed: int
    tickets_kept: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def clean_text(text: str) -> str:
    """Normalize line breaks and whitespace, then mask personal data."""
    text = _ESCAPED_BREAK.sub("\n", text or "").replace("\r\n", "\n")
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    text = _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()
    return mask_pii(text)


def build_text(subject: str, body: str) -> str:
    """Combine subject and body into the model input."""
    subject, body = clean_text(subject), clean_text(body)
    return f"{subject}\n\n{body}" if subject else body


def _normalized(text: str) -> str:
    return _NON_WORD.sub(" ", text.lower()).strip()


def ticket_id(text: str) -> str:
    """Stable id derived from the normalized text."""
    return hashlib.sha1(_normalized(text).encode("utf-8")).hexdigest()[:12]


def load_tickets(
    path: Path, language: str, label_column: str
) -> tuple[list[Ticket], CleaningReport]:
    """Read the raw CSV, keep one language, clean and deduplicate.

    Exact duplicates after normalization are dropped. When the same text
    appears with different labels, every copy is dropped, because no single
    label can be defended for it.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"subject", "body", "language", "priority", "type", label_column}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
        raw = list(reader)

    other_language = empty = 0
    candidates: list[Ticket] = []
    for row in raw:
        if row["language"].strip().lower() != language:
            other_language += 1
            continue
        text = build_text(row["subject"], row["body"])
        label = (row[label_column] or "").strip()
        if not text or not label:
            empty += 1
            continue
        candidates.append(
            Ticket(
                ticket_id=ticket_id(text),
                text=text,
                label=label,
                priority=(row["priority"] or "").strip().lower(),
                ticket_type=(row["type"] or "").strip(),
            )
        )

    labels_by_id: dict[str, set[str]] = {}
    for ticket in candidates:
        labels_by_id.setdefault(ticket.ticket_id, set()).add(ticket.label)
    conflicting = {tid for tid, labels in labels_by_id.items() if len(labels) > 1}

    kept: list[Ticket] = []
    seen: set[str] = set()
    duplicates = conflicts = 0
    for ticket in candidates:
        if ticket.ticket_id in conflicting:
            conflicts += 1
        elif ticket.ticket_id in seen:
            duplicates += 1
        else:
            seen.add(ticket.ticket_id)
            kept.append(ticket)

    report = CleaningReport(
        rows_read=len(raw),
        rows_other_language=other_language,
        rows_empty=empty,
        duplicates_removed=duplicates,
        conflicting_labels_removed=conflicts,
        tickets_kept=len(kept),
    )
    return kept, report


def split_tickets(
    tickets: Sequence[Ticket], validation: float, test: float, seed: int
) -> dict[str, list[Ticket]]:
    """Stratified train, validation and test split with a fixed seed."""
    from sklearn.model_selection import train_test_split

    labels = [t.label for t in tickets]
    train_val, test_set = train_test_split(
        list(tickets), test_size=test, stratify=labels, random_state=seed
    )
    relative_val = validation / (1.0 - test)
    train_set, val_set = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=[t.label for t in train_val],
        random_state=seed,
    )
    return {"train": train_set, "validation": val_set, "test": test_set}


def label_names(tickets: Iterable[Ticket]) -> list[str]:
    return sorted({t.label for t in tickets})


def save_splits(splits: dict[str, list[Ticket]], labels: Sequence[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [f.name for f in fields(Ticket)]
    for name, tickets in splits.items():
        with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(asdict(t) for t in tickets)
    (out_dir / LABELS_FILENAME).write_text(json.dumps(list(labels), indent=2), encoding="utf-8")


def load_split(out_dir: Path, name: str) -> list[Ticket]:
    path = out_dir / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run scripts/prepare_tickets.py first.")
    with path.open(newline="", encoding="utf-8") as handle:
        return [Ticket(**row) for row in csv.DictReader(handle)]


def load_labels(out_dir: Path) -> list[str]:
    return json.loads((out_dir / LABELS_FILENAME).read_text(encoding="utf-8"))


def class_counts(tickets: Iterable[Ticket]) -> dict[str, int]:
    return dict(sorted(Counter(t.label for t in tickets).items(), key=lambda kv: -kv[1]))
