import importlib.util
from pathlib import Path

from resolveiq.config import PROJECT_ROOT

_SPEC = importlib.util.spec_from_file_location(
    "deploy_space", PROJECT_ROOT / "scripts" / "deploy_space.py"
)
deploy_space = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(deploy_space)


def _tree(root: Path) -> None:
    files = {
        "space.py": "",
        "src/pkg/core.py": "",
        "src/pkg/__pycache__/core.cpython-312.pyc": "",
        "src/pkg/core 2.py": "",
        "src/pkg/.env": "SECRET=1",
        "web/index.html": "",
        "web/assets/app.js": "",
        "web/assets/app.js.map": "",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def test_stage_copies_only_what_the_space_needs(tmp_path):
    root, out = tmp_path / "repo", tmp_path / "out"
    _tree(root)

    copied = deploy_space.stage(out, root, ["space.py", "src/pkg", "web"])

    assert sorted(copied) == ["space.py", "src/pkg/core.py", "web/assets/app.js", "web/index.html"]
    assert not (out / "src/pkg/.env").exists()


def test_stage_requires_the_built_frontend(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="frontend/dist"):
        deploy_space.stage(tmp_path / "out", tmp_path, ["frontend/dist"])


def test_space_readme_points_at_the_launcher(tmp_path):
    deploy_space.write_readme(tmp_path, "https://example.org/repo")

    text = (tmp_path / "README.md").read_text()
    header = text.split("---")[1]
    assert "app_file: space.py" in header
    assert "sdk: gradio" in header
    assert f"sdk_version: {deploy_space.gradio_version()}" in header
    assert "https://example.org/repo" in text


def test_default_include_list_never_ships_secrets_or_models():
    assert not any(
        entry.startswith((".env", "models", "logs", "data/raw", "data/processed", "mlruns"))
        for entry in deploy_space.INCLUDE
    )
