from resolveiq.registry import clean_copy, resolve_model_dir


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
