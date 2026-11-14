from resolveiq.rag.cleaning import clean_pages, normalize_text
from resolveiq.rag.types import Page


def test_normalize_joins_hyphenated_and_soft_breaks():
    text = "The refund is pro-\ncessed within three\nbusiness days.\n\n\n\nNext   paragraph."

    assert normalize_text(text) == (
        "The refund is processed within three business days.\n\nNext paragraph."
    )


def test_normalize_keeps_list_structure():
    text = "Accepted cards:\n- Visa\n- Mastercard"

    assert normalize_text(text) == text


def test_repeated_headers_and_footers_are_removed():
    pages = [
        Page("guide.pdf", n, f"Altura Help Center\nContent of page {n} with details.\nPage footer")
        for n in range(1, 5)
    ]

    cleaned = clean_pages(pages)

    assert [p.page for p in cleaned] == [1, 2, 3, 4]
    for page in cleaned:
        assert "Altura Help Center" not in page.text
        assert "Page footer" not in page.text
        assert page.text.startswith("Content of page")


def test_headers_are_not_detected_in_short_documents():
    pages = [Page("a.pdf", n, "Title line\nEnough body text to keep this page.") for n in (1, 2)]

    cleaned = clean_pages(pages)

    assert all(p.text.startswith("Title line") for p in cleaned)


def test_empty_pages_are_dropped_but_numbering_is_kept():
    pages = [
        Page("a.pdf", 1, "A page with enough text to keep."),
        Page("a.pdf", 2, "   \n  "),
        Page("a.pdf", 3, "Another page with enough text."),
    ]

    assert [p.page for p in clean_pages(pages)] == [1, 3]
