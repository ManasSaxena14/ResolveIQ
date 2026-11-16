from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from resolveiq.config import CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH, load_config


def _write_config(tmp_path: Path, **overrides: dict) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for section, values in overrides.items():
        raw[section].update(values)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_default_config_is_valid():
    settings = load_config(DEFAULT_CONFIG_PATH)

    assert settings.project.company_name
    assert settings.rag.chunk_overlap < settings.rag.chunk_tokens
    assert settings.llm.active.model


def test_config_path_can_come_from_environment(tmp_path, monkeypatch):
    path = _write_config(tmp_path, project={"company_name": "Testco"})
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))

    assert load_config().project.company_name == "Testco"


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_overlap_must_be_smaller_than_chunk(tmp_path):
    path = _write_config(tmp_path, rag={"chunk_tokens": 100, "chunk_overlap": 100})

    with pytest.raises(ValidationError, match="chunk_overlap"):
        load_config(path)


def test_candidate_pool_must_cover_top_k(tmp_path):
    path = _write_config(tmp_path, rag={"top_k": 10, "candidate_k": 5})

    with pytest.raises(ValidationError, match="candidate_k"):
        load_config(path)


def test_unknown_keys_are_rejected(tmp_path):
    path = _write_config(tmp_path, rag={"chunk_tokenz": 300})

    with pytest.raises(ValidationError):
        load_config(path)


def test_active_provider_must_be_configured(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["llm"]["provider"] = "groq"
    del raw["llm"]["providers"]["groq"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValidationError, match="groq"):
        load_config(path)


def test_relative_paths_resolve_from_project_root():
    settings = load_config(DEFAULT_CONFIG_PATH)

    resolved = settings.resolve(settings.paths.kb_dir)

    assert resolved.is_absolute()
    assert resolved.parts[-2:] == ("data", "kb")


def test_risk_tuning_grid_rejects_unknown_parameters(tmp_path):
    path = _write_config(tmp_path, risk={"tuning_grid": {"max_leaves": [8]}})

    with pytest.raises(ValidationError, match="unknown parameters"):
        load_config(path)


def test_risk_defaults_are_valid():
    risk = load_config(DEFAULT_CONFIG_PATH).risk

    assert risk.label.source_column == "priority"
    assert risk.alert_cost_ratio in risk.cost_ratios
    assert risk.calibration in {"isotonic", "sigmoid"}
