"""Experiment tracking helpers built on MLflow."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from resolveiq.config import PROJECT_ROOT, Settings


def git_commit() -> str:
    """Short hash of the current commit, or ``"unknown"`` outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def resolve_tracking_uri(uri: str) -> str:
    """Make relative file and SQLite tracking URIs absolute from the project root.

    MLflow resolves relative paths from the current working directory, which
    would scatter runs across folders when scripts are started from elsewhere.
    """
    sqlite_prefix = "sqlite:///"
    if uri.startswith(sqlite_prefix) and not uri[len(sqlite_prefix) :].startswith("/"):
        return sqlite_prefix + str(PROJECT_ROOT / uri[len(sqlite_prefix) :])
    if "://" not in uri and not uri.startswith("/"):
        return (PROJECT_ROOT / uri).as_uri()
    return uri


@contextmanager
def tracked_run(
    settings: Settings,
    experiment: str,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    """Open an MLflow run tagged with the git commit, or do nothing if disabled.

    Yields the ``mlflow`` module when tracking is enabled and ``None`` otherwise,
    so callers can write ``if mlflow: mlflow.log_metric(...)``.
    """
    if not settings.tracking.enabled:
        yield None
        return

    import mlflow

    mlflow.set_tracking_uri(resolve_tracking_uri(settings.tracking.tracking_uri))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"git_commit": git_commit(), **(tags or {})})
        yield mlflow
