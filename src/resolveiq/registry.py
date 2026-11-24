"""Register trained models in the local MLflow model registry and load them back.

Each model folder is logged as an MLflow ``pyfunc`` model whose artifact is the
folder itself, then registered under a name and tagged with the ``production``
alias. The service downloads the folder behind that alias once, caches it by
version, and loads it with the project's own loaders.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from resolveiq.config import Settings
from resolveiq.rag.loaders import is_sync_duplicate
from resolveiq.tracking import resolve_tracking_uri

logger = logging.getLogger(__name__)

ARTIFACT_KEY = "model_dir"


@dataclass(frozen=True)
class ResolvedModel:
    path: Path
    source: str  # "registry" or "local"
    version: str


def clean_copy(source: Path, destination: Path) -> Path:
    """Copy a model folder without sync-tool duplicates or hidden files."""
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and not path.name.startswith(".") and not is_sync_duplicate(path.name):
            shutil.copy2(path, destination / path.name)
    return destination


def _python_model(kind: str) -> Any:
    """A pyfunc wrapper so the registered model can also be served by MLflow."""
    import mlflow.pyfunc

    class ResolveIQModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context: Any) -> None:
            model_dir = Path(context.artifacts[ARTIFACT_KEY])
            if kind == "classifier":
                from resolveiq.classifier.predict import TicketClassifier

                self.model = TicketClassifier.load(model_dir)
            elif kind == "semantic":
                from resolveiq.rag.embeddings import SentenceTransformerEmbedder
                from resolveiq.semantic import SemanticClassifier, SemanticModel, TextEncoder

                model = SemanticModel.load(model_dir)
                encoder = TextEncoder(SentenceTransformerEmbedder(model.params.embedding_model))
                self.model = SemanticClassifier(model, encoder)
            else:
                from resolveiq.risk.predict import RiskModel

                self.model = RiskModel.load(model_dir)

        def predict(self, context: Any, model_input: Any, params: Any = None) -> Any:
            subjects = model_input.get("subject", [""] * len(model_input))
            bodies = model_input["body"]
            if kind in ("classifier", "semantic"):
                return [
                    self.model.predict_ticket(body, subject).label
                    for subject, body in zip(subjects, bodies, strict=True)
                ]
            types = model_input.get("type", [""] * len(model_input))
            return [
                self.model.score(subject, body, ticket_type).escalation_probability
                for subject, body, ticket_type in zip(subjects, bodies, types, strict=True)
            ]

    return ResolveIQModel()


def registered_name(settings: Settings, kind: str) -> str:
    return getattr(settings.service.registry, f"{kind}_name")


def register_model(
    settings: Settings, kind: str, model_dir: Path, experiment: str, tags: dict[str, str]
) -> str:
    """Log ``model_dir`` as a model, register it and point the alias at it.

    Returns the new registry version.
    """
    import mlflow
    from mlflow import MlflowClient

    from resolveiq.tracking import tracked_run

    registry = settings.service.registry
    name = registered_name(settings, kind)
    with tempfile.TemporaryDirectory() as tmp:
        staged = clean_copy(model_dir, Path(tmp) / model_dir.name)
        with tracked_run(settings, experiment, run_name=f"register-{kind}", tags=tags):
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=_python_model(kind),
                artifacts={ARTIFACT_KEY: str(staged)},
                registered_model_name=name,
                pip_requirements=["-r requirements.txt"],
            )
    client = MlflowClient()
    version = str(info.registered_model_version)
    client.set_registered_model_alias(name, registry.alias, version)
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, value)
    return version


HUB_ENV_VARS = {
    "classifier": "RESOLVEIQ_CLASSIFIER_REPO",
    "risk": "RESOLVEIQ_RISK_REPO",
    "semantic": "RESOLVEIQ_SEMANTIC_REPO",
}
LITE_REPO_ENV_VAR = "RESOLVEIQ_LITE_REPO"
META_FILES = {
    "classifier": "classifier_meta.json",
    "risk": "risk_meta.json",
    "semantic": "semantic_meta.json",
}


def fetch_from_hub(settings: Settings, environ: Any = None) -> list[str]:
    """Download published models into the local folders when they are missing.

    Used by the hosted demo, where ``models/`` is not part of the repository.
    The Hugging Face repository of each model is read from the environment
    (``RESOLVEIQ_CLASSIFIER_REPO``, ``RESOLVEIQ_RISK_REPO``,
    ``RESOLVEIQ_SEMANTIC_REPO``). Returns the kinds that were downloaded.
    """
    import os

    environ = os.environ if environ is None else environ
    downloaded = []
    if settings.service.runtime == "lite":
        # The lite bundle holds both models and the index; nothing else is needed.
        from resolveiq.lite import lite_ready

        repo_id = environ.get(LITE_REPO_ENV_VAR, "").strip()
        local = settings.resolve(settings.service.lite_dir)
        if repo_id and not lite_ready(local):
            from huggingface_hub import snapshot_download

            logger.info("Downloading the lite bundle from %s", repo_id)
            snapshot_download(repo_id=repo_id, local_dir=str(local))
            downloaded.append("lite")
        return downloaded
    for kind, variable in HUB_ENV_VARS.items():
        repo_id = environ.get(variable, "").strip()
        local = settings.resolve(getattr(settings.service, f"{kind}_dir"))
        if not repo_id or (local / META_FILES[kind]).is_file():
            continue
        from huggingface_hub import snapshot_download

        logger.info("Downloading the %s from %s", kind, repo_id)
        snapshot_download(repo_id=repo_id, local_dir=str(local))
        downloaded.append(kind)
    return downloaded


def resolve_model_dir(settings: Settings, kind: str) -> ResolvedModel:
    """Folder to load a model from, following ``service.model_source``.

    With ``registry`` the folder behind the alias is downloaded once and cached
    by version. Any registry problem (no database, no alias, missing files)
    falls back to the local folder with a warning, so a hosted demo without the
    registry still starts.
    """
    service = settings.service
    local = settings.resolve(getattr(service, f"{kind}_dir"))
    if service.model_source == "registry":
        try:
            return _download_from_registry(settings, kind)
        except Exception as exc:  # the registry is optional at run time
            logger.warning("Registry load of the %s failed (%s); using %s", kind, exc, local)
    return ResolvedModel(local, "local", "local")


def _download_from_registry(settings: Settings, kind: str) -> ResolvedModel:
    import mlflow
    from mlflow import MlflowClient

    registry = settings.service.registry
    name = registered_name(settings, kind)
    mlflow.set_tracking_uri(resolve_tracking_uri(settings.tracking.tracking_uri))
    version = MlflowClient().get_model_version_by_alias(name, registry.alias).version
    cache = settings.resolve(registry.cache_dir) / name / str(version)
    if not cache.is_dir():
        with tempfile.TemporaryDirectory() as tmp:
            downloaded = Path(
                mlflow.artifacts.download_artifacts(
                    artifact_uri=f"models:/{name}/{version}", dst_path=tmp
                )
            )
            artifact_root = downloaded / "artifacts"
            folders = [p for p in artifact_root.iterdir() if p.is_dir()]
            if len(folders) != 1:
                raise FileNotFoundError(f"Unexpected artifact layout in {artifact_root}")
            cache.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(folders[0], cache)
    return ResolvedModel(cache, "registry", f"{name}/v{version}")
