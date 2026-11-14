import re

from resolveiq.config import PROJECT_ROOT
from resolveiq.tracking import git_commit, resolve_tracking_uri, tracked_run


def test_relative_sqlite_uri_is_anchored_to_project_root():
    assert resolve_tracking_uri("sqlite:///mlflow.db") == f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


def test_absolute_and_remote_uris_are_unchanged():
    assert resolve_tracking_uri("sqlite:////tmp/mlflow.db") == "sqlite:////tmp/mlflow.db"
    assert resolve_tracking_uri("http://localhost:5001") == "http://localhost:5001"


def test_relative_folder_becomes_file_uri():
    uri = resolve_tracking_uri("mlruns")

    assert uri.startswith("file://")
    assert uri.endswith("/mlruns")


def test_git_commit_returns_a_short_hash_or_unknown():
    commit = git_commit()

    assert commit == "unknown" or re.fullmatch(r"[0-9a-f]{7,12}", commit)


def test_tracking_disabled_yields_none(settings):
    with tracked_run(settings, experiment="rag") as mlflow:
        assert mlflow is None
