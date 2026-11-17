"""Request log and feedback store backed by SQLite.

Every analyzed ticket writes one row with the numbers needed for monitoring:
time, channel, predicted category and confidence, risk score, text length,
latency and model versions. The ticket text itself is never stored. Only its
length and a truncated SHA-256 hash are kept, so a repeated ticket can be
recognized without keeping personal data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import statistics
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from resolveiq.privacy import mask_pii

logger = logging.getLogger("resolveiq.requests")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id   TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    channel      TEXT NOT NULL DEFAULT '',
    category     TEXT NOT NULL,
    confidence   REAL NOT NULL,
    needs_review INTEGER NOT NULL,
    risk_score   REAL NOT NULL,
    alert        INTEGER NOT NULL,
    text_length  INTEGER NOT NULL,
    text_hash    TEXT NOT NULL,
    draft_status TEXT NOT NULL,
    latency_ms   REAL NOT NULL,
    versions     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_requests_created ON requests (created_at);
CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL REFERENCES requests (request_id),
    rating     TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment    TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
"""

HASH_CHARS = 16


def text_fingerprint(text: str) -> tuple[int, str]:
    """Length and truncated SHA-256 hash of a text, safe to log."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_CHARS]
    return len(text), digest


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    channel: str
    category: str
    confidence: float
    needs_review: bool
    risk_score: float
    alert: bool
    text_length: int
    text_hash: str
    draft_status: str
    latency_ms: float
    versions: dict[str, str]
    created_at: str = ""


@dataclass(frozen=True)
class FeedbackStats:
    total: int
    up: int
    down: int

    @property
    def down_rate(self) -> float:
        return self.down / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "down_rate": round(self.down_rate, 4)}


class RequestLog:
    """Thread-safe SQLite store for request rows and feedback votes."""

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._transaction() as cursor:
            cursor.executescript(_SCHEMA)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cursor = self._connection.cursor()
            try:
                yield cursor
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
            finally:
                cursor.close()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def record(self, record: RequestRecord) -> None:
        values = asdict(record)
        values["created_at"] = record.created_at or utc_now()
        values["needs_review"] = int(record.needs_review)
        values["alert"] = int(record.alert)
        values["versions"] = json.dumps(record.versions, sort_keys=True)
        columns = ", ".join(values)
        placeholders = ", ".join(f":{name}" for name in values)
        with self._transaction() as cursor:
            cursor.execute(f"INSERT INTO requests ({columns}) VALUES ({placeholders})", values)

    def exists(self, request_id: str) -> bool:
        with self._transaction() as cursor:
            cursor.execute("SELECT 1 FROM requests WHERE request_id = ?", (request_id,))
            return cursor.fetchone() is not None

    def add_feedback(self, request_id: str, rating: str, comment: str = "") -> bool:
        """Store a vote. Returns False when the request id is unknown."""
        if rating not in ("up", "down"):
            raise ValueError("rating must be 'up' or 'down'")
        if not self.exists(request_id):
            return False
        with self._transaction() as cursor:
            cursor.execute(
                "INSERT INTO feedback (request_id, rating, comment, created_at) "
                "VALUES (?, ?, ?, ?)",
                (request_id, rating, mask_pii(comment.strip())[:500], utc_now()),
            )
        return True

    def recent(self, limit: int) -> list[dict[str, Any]]:
        """The newest ``limit`` request rows, oldest first."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM (SELECT rowid AS seq, * FROM requests "
                "ORDER BY created_at DESC, seq DESC LIMIT ?) ORDER BY created_at, seq",
                (limit,),
            )
            rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            row.pop("seq", None)
            row["versions"] = json.loads(row["versions"])
            row["needs_review"] = bool(row["needs_review"])
            row["alert"] = bool(row["alert"])
        return rows

    def count(self) -> int:
        with self._transaction() as cursor:
            cursor.execute("SELECT COUNT(*) FROM requests")
            return int(cursor.fetchone()[0])

    def feedback_stats(self) -> FeedbackStats:
        with self._transaction() as cursor:
            cursor.execute("SELECT rating, COUNT(*) FROM feedback GROUP BY rating")
            counts = {rating: int(n) for rating, n in cursor.fetchall()}
        up, down = counts.get("up", 0), counts.get("down", 0)
        return FeedbackStats(total=up + down, up=up, down=down)

    def ratings(self, request_ids: list[str]) -> dict[str, str]:
        """The latest vote for each of ``request_ids`` that has one."""
        if not request_ids:
            return {}
        placeholders = ", ".join("?" for _ in request_ids)
        with self._transaction() as cursor:
            cursor.execute(
                f"SELECT request_id, rating FROM feedback WHERE request_id IN ({placeholders}) "
                "ORDER BY id",
                request_ids,
            )
            return {request_id: rating for request_id, rating in cursor.fetchall()}

    def summary(self) -> dict[str, Any]:
        """Totals over the whole log for the monitoring dashboard."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT COUNT(*), COALESCE(SUM(alert), 0), COALESCE(SUM(needs_review), 0), "
                "AVG(risk_score), AVG(confidence) FROM requests"
            )
            total, alerts, reviews, mean_risk, mean_confidence = cursor.fetchone()
            cursor.execute("SELECT category, COUNT(*) FROM requests GROUP BY category")
            categories = {label: int(n) for label, n in cursor.fetchall()}
            cursor.execute("SELECT draft_status, COUNT(*) FROM requests GROUP BY draft_status")
            drafts = {status: int(n) for status, n in cursor.fetchall()}
            cursor.execute("SELECT latency_ms FROM requests ORDER BY latency_ms")
            latencies = [float(row[0]) for row in cursor.fetchall()]
        median = round(statistics.median(latencies), 1) if latencies else None
        p95 = (
            latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))]
            if latencies
            else None
        )
        return {
            "requests": int(total),
            "alerts": int(alerts),
            "needs_review": int(reviews),
            "mean_risk": round(float(mean_risk), 4) if mean_risk is not None else None,
            "mean_confidence": (
                round(float(mean_confidence), 4) if mean_confidence is not None else None
            ),
            "categories": dict(sorted(categories.items(), key=lambda item: -item[1])),
            "draft_status": drafts,
            "latency_median_ms": median,
            "latency_p95_ms": p95,
        }


def log_event(event: str, **fields: Any) -> None:
    """Write one structured log line. Callers must never pass ticket text."""
    logger.info(json.dumps({"event": event, **fields}, sort_keys=True, default=str))
