"""Read knowledge base files into pages of plain text."""

from __future__ import annotations

import logging
from html.parser import HTMLParser
from pathlib import Path

from resolveiq.rag.types import Page

logger = logging.getLogger(__name__)

PDF_SUFFIXES = {".pdf"}
HTML_SUFFIXES = {".html", ".htm"}
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | HTML_SUFFIXES | TEXT_SUFFIXES


def discover_files(kb_dir: Path) -> list[Path]:
    """Return supported files under ``kb_dir`` in a stable order."""
    if not kb_dir.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")
    return sorted(
        path
        for path in kb_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def load_documents(kb_dir: Path) -> list[Page]:
    """Load every supported file under ``kb_dir`` into pages."""
    pages: list[Page] = []
    for path in discover_files(kb_dir):
        source = path.relative_to(kb_dir).as_posix()
        pages.extend(load_file(path, source=source))
    return pages


def load_file(path: Path, source: str | None = None) -> list[Page]:
    """Load one file. ``source`` defaults to the file name."""
    source = source or path.name
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return _load_pdf(path, source)
    if suffix in HTML_SUFFIXES:
        text = html_to_text(path.read_text(encoding="utf-8", errors="replace"))
        return [Page(source=source, page=1, text=text)]
    if suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [Page(source=source, page=1, text=text)]
    raise ValueError(f"Unsupported file type: {path}")


def _load_pdf(path: Path, source: str) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for number, pdf_page in enumerate(reader.pages, start=1):
        try:
            text = pdf_page.extract_text() or ""
        except Exception:  # pypdf raises several error types on malformed pages
            logger.warning("Could not extract text from %s page %d", source, number)
            text = ""
        pages.append(Page(source=source, page=number, text=text))
    return pages


# Elements whose text is never shown to a reader of the article.
_SKIPPED_TAGS = frozenset({"script", "style", "noscript", "template", "head", "nav", "footer"})
# Elements that start a new line in the extracted text.
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article",
        "main", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre",
    }
)  # fmt: skip


class _TextExtractor(HTMLParser):
    """Collect visible text from HTML, keeping block elements on separate lines."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")
        if tag == "li" and not self._skip_depth:
            self._parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self._parts).splitlines())
        return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> str:
    """Convert an HTML document to readable plain text."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()
