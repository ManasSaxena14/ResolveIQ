import csv
from collections import Counter

import pytest

from resolveiq.classifier.data import (
    Ticket,
    build_text,
    class_counts,
    clean_text,
    label_names,
    load_labels,
    load_split,
    load_tickets,
    save_splits,
    split_tickets,
    ticket_id,
)

COLUMNS = ["subject", "body", "answer", "type", "queue", "priority", "language"]


def _write_raw(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({"answer": "", "type": "Incident", "priority": "high", **row})


def _row(subject, body, queue, language="en"):
    return {"subject": subject, "body": body, "queue": queue, "language": language}


def test_clean_text_converts_escaped_breaks_and_masks_pii():
    text = "Hello,\\n\\nMy email is a@b.com<br>Thanks   a lot\\n\\n\\n\\nBye"

    assert clean_text(text) == "Hello,\n\nMy email is [EMAIL]\nThanks a lot\n\nBye"


def test_build_text_handles_missing_subject():
    assert build_text("Login issue", "Cannot sign in") == "Login issue\n\nCannot sign in"
    assert build_text("", "Cannot sign in") == "Cannot sign in"


def test_ticket_id_ignores_case_and_punctuation():
    assert ticket_id("Cannot sign in!") == ticket_id("cannot   sign in")


def test_load_tickets_filters_language_and_deduplicates(tmp_path):
    path = tmp_path / "raw.csv"
    _write_raw(
        path,
        [
            _row("Refund", "Please refund my order", "Billing"),
            _row("refund", "please refund my order!", "Billing"),  # duplicate
            _row("Hilfe", "Bitte helfen", "Billing", language="de"),
            _row("Server", "The server is down", "Outages"),
            _row("Server", "The server is down", "IT Support"),  # conflicting label
            _row("", "   ", "Billing"),  # empty
        ],
    )

    tickets, report = load_tickets(path, language="en", label_column="queue")

    assert [t.label for t in tickets] == ["Billing"]
    assert report.rows_read == 6
    assert report.rows_other_language == 1
    assert report.rows_empty == 1
    assert report.duplicates_removed == 1
    assert report.conflicting_labels_removed == 2
    assert report.tickets_kept == 1
    assert tickets[0].priority == "high"


def test_load_tickets_requires_columns(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("subject,body\nA,B\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing columns"):
        load_tickets(path, "en", "queue")


def _tickets(n_per_class: int = 40) -> list[Ticket]:
    return [
        Ticket(f"{label}-{i}", f"{label} ticket number {i}", label, "low", "Request")
        for label in ("A", "B", "C")
        for i in range(n_per_class)
    ]


def test_split_is_stratified_disjoint_and_reproducible():
    tickets = _tickets()

    splits = split_tickets(tickets, validation=0.15, test=0.15, seed=42)
    again = split_tickets(tickets, validation=0.15, test=0.15, seed=42)

    ids = {name: {t.ticket_id for t in items} for name, items in splits.items()}
    assert sum(len(v) for v in ids.values()) == len(tickets)
    assert not ids["train"] & ids["test"]
    assert not ids["validation"] & ids["test"]
    assert Counter(t.label for t in splits["test"]) == {"A": 6, "B": 6, "C": 6}
    assert [t.ticket_id for t in again["test"]] == [t.ticket_id for t in splits["test"]]


def test_save_and_load_round_trip(tmp_path):
    tickets = _tickets(5)
    splits = {"train": tickets[:10], "validation": tickets[10:12], "test": tickets[12:]}

    save_splits(splits, label_names(tickets), tmp_path)

    assert load_split(tmp_path, "train") == tickets[:10]
    assert load_labels(tmp_path) == ["A", "B", "C"]
    assert class_counts(tickets) == {"A": 5, "B": 5, "C": 5}


def test_missing_split_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_tickets"):
        load_split(tmp_path, "train")
