from resolveiq.registry import clean_copy, fetch_from_hub, resolve_model_dir


def test_clean_copy_skips_sync_duplicates_and_hidden_files(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    for name in ["model.safetensors", "model 2.safetensors", "config.json", ".DS_Store"]:
        (source / name).write_text(name)
    (source / "nested").mkdir()

    copied = clean_copy(source, tmp_path / "out")

    assert sorted(p.name for p in copied.iterdir()) == ["config.json", "model.safetensors"]


def test_local_source_uses_the_configured_folder(settings):
    settings.service.model_source = "local"

    resolved = resolve_model_dir(settings, "risk")

    assert resolved.source == "local"
    assert resolved.path == settings.resolve(settings.service.risk_dir)


def test_registry_problems_fall_back_to_the_local_folder(settings, tmp_path, caplog):
    settings.service.model_source = "registry"
    settings.tracking.tracking_uri = f"sqlite:///{tmp_path / 'empty.db'}"

    with caplog.at_level("WARNING"):
        resolved = resolve_model_dir(settings, "classifier")

    assert resolved.source == "local"
    assert resolved.path == settings.resolve(settings.service.classifier_dir)
    assert "Registry load of the classifier failed" in caplog.text


def test_hub_fetch_skips_without_a_repo_or_with_local_models(settings, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: calls.append(kwargs))
    settings.service.risk_dir = tmp_path / "risk"
    settings.service.risk_dir.mkdir()
    (settings.service.risk_dir / "risk_meta.json").write_text("{}")

    assert fetch_from_hub(settings, environ={}) == []
    assert fetch_from_hub(settings, environ={"RESOLVEIQ_RISK_REPO": "u/risk"}) == []
    assert calls == []


def test_hub_fetch_downloads_missing_models(settings, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: calls.append(kwargs))
    settings.service.classifier_dir = tmp_path / "classifier"
    settings.service.risk_dir = tmp_path / "risk"

    downloaded = fetch_from_hub(
        settings, environ={"RESOLVEIQ_CLASSIFIER_REPO": "u/clf", "RESOLVEIQ_RISK_REPO": " "}
    )

    assert downloaded == ["classifier"]
    assert calls == [{"repo_id": "u/clf", "local_dir": str(tmp_path / "classifier")}]
