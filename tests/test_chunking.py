import itertools

import pytest

from resolveiq.rag.chunking import chunk_pages, make_chunk_id, split_text
from resolveiq.rag.types import Page


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_is_one_chunk():
    assert split_text("one two three", chunk_tokens=10, chunk_overlap=2) == ["one two three"]


def test_empty_text_has_no_chunks():
    assert split_text("   \n ", chunk_tokens=10, chunk_overlap=2) == []


def test_chunks_respect_the_token_limit():
    chunks = split_text(_words(95), chunk_tokens=20, chunk_overlap=5)

    assert all(len(chunk.split()) <= 20 for chunk in chunks)


def test_consecutive_chunks_overlap():
    chunks = split_text(_words(50), chunk_tokens=20, chunk_overlap=5)

    for previous, current in itertools.pairwise(chunks):
        assert previous.split()[-5:] == current.split()[:5]


def test_every_word_is_covered():
    text = _words(73)
    chunks = split_text(text, chunk_tokens=20, chunk_overlap=5)

    covered = set()
    for chunk in chunks:
        covered.update(chunk.split())
    assert covered == set(text.split())
    assert chunks[-1].split()[-1] == "w72"


def test_zero_overlap_produces_disjoint_chunks():
    chunks = split_text(_words(40), chunk_tokens=10, chunk_overlap=0)

    assert len(chunks) == 4
    assert sum(len(c.split()) for c in chunks) == 40


def test_word_longer_than_limit_becomes_its_own_chunk():
    chunks = split_text("short huge short", 3, 1, count_tokens=lambda w: 5 if w == "huge" else 1)

    assert chunks == ["short", "huge", "short"]


def test_custom_token_counter_is_used():
    chunks = split_text("aa bb cc dd", 4, 0, count_tokens=lambda w: 2)

    assert chunks == ["aa bb", "cc dd"]


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_sizes_raise(size, overlap):
    with pytest.raises(ValueError):
        split_text("text", size, overlap)


def test_chunk_pages_keeps_source_metadata():
    pages = [Page("billing/refunds.md", 1, _words(30)), Page("guide.pdf", 4, _words(5))]

    chunks = chunk_pages(pages, chunk_tokens=20, chunk_overlap=5)

    assert [(c.source, c.page, c.chunk_index) for c in chunks] == [
        ("billing/refunds.md", 1, 0),
        ("billing/refunds.md", 1, 1),
        ("guide.pdf", 4, 0),
    ]
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert chunks[2].chunk_id == make_chunk_id("guide.pdf", 4, 0)
