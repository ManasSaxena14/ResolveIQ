from pathlib import Path
from types import SimpleNamespace

import pytest

from resolveiq.rag import loaders
from resolveiq.rag.loaders import (
    discover_files,
    html_to_text,
    is_sync_duplicate,
    load_documents,
    load_file,
)


def test_discover_files_skips_unsupported_formats(kb_dir):
    names = [path.name for path in discover_files(kb_dir)]

    assert names == ["account.html", "refunds.md", "shipping.txt"]


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_files(tmp_path / "missing")


def test_sources_are_relative_to_kb_dir(kb_dir):
    sources = {page.source for page in load_documents(kb_dir)}

    assert sources == {"account.html", "billing/refunds.md", "shipping.txt"}


def test_html_drops_scripts_and_head():
    text = html_to_text(
        "<html><head><title>T</title><style>p{}</style></head>"
        "<body><nav>Menu</nav><h1>Title</h1><p>First &amp; second.</p>"
        "<ul><li>One</li><li>Two</li></ul><script>alert(1)</script></body></html>"
    )

    assert text.splitlines() == ["Title", "First & second.", "- One", "- Two"]


def test_text_formats_produce_a_single_page(tmp_path):
    path = tmp_path / "article.md"
    path.write_text("# Heading\n\nBody", encoding="utf-8")

    pages = load_file(path)

    assert len(pages) == 1
    assert pages[0].page == 1
    assert pages[0].source == "article.md"


def test_unsupported_file_type_raises(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported"):
        load_file(path)


def test_pdf_keeps_page_numbers(tmp_path, monkeypatch):
    class BrokenPage:
        def extract_text(self):
            raise RuntimeError("corrupt page")

    fake_reader = SimpleNamespace(
        pages=[
            SimpleNamespace(extract_text=lambda: "Page one text"),
            BrokenPage(),
            SimpleNamespace(extract_text=lambda: None),
        ]
    )
    monkeypatch.setattr("pypdf.PdfReader", lambda _path: fake_reader)
    path = tmp_path / "guide.pdf"
    path.write_bytes(b"%PDF-1.4")

    pages = loaders.load_file(Path(path))

    assert [(p.page, p.text) for p in pages] == [(1, "Page one text"), (2, ""), (3, "")]


def test_sync_duplicates_and_hidden_files_are_skipped(kb_dir, caplog):
    (kb_dir / "billing" / "refunds 2.md").write_text("Copy made by a sync tool", encoding="utf-8")
    (kb_dir / ".draft.md").write_text("Hidden draft", encoding="utf-8")

    with caplog.at_level("WARNING"):
        names = [path.name for path in discover_files(kb_dir)]

    assert names == ["account.html", "refunds.md", "shipping.txt"]
    assert "Skipped 2 hidden or duplicate files" in caplog.text


def test_is_sync_duplicate():
    assert is_sync_duplicate("refunds 2.md")
    assert is_sync_duplicate("notes 12")
    assert not is_sync_duplicate("chapter 2 overview.md")
    assert not is_sync_duplicate("refunds.md")
