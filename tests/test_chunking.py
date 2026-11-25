import pytest

from app.services.chunking import chunk_text


def test_empty_text_produces_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_produces_one_chunk():
    text = "This is a short document with just a few words."
    chunks = chunk_text(text, chunk_size_words=220, overlap_words=30)
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].content == text
    assert chunks[0].word_count == len(text.split())


def test_long_text_splits_into_multiple_overlapping_chunks():
    words = [f"word{i}" for i in range(500)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size_words=100, overlap_words=20)

    assert len(chunks) > 1
    # Every chunk respects the size cap.
    assert all(c.word_count <= 100 for c in chunks)
    # Consecutive chunks actually overlap: the tail of one chunk reappears
    # at the head of the next.
    first_words = chunks[0].content.split()
    second_words = chunks[1].content.split()
    assert first_words[-20:] == second_words[:20]
    # Indices are sequential starting at 0.
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunks_cover_entire_text_with_no_gaps():
    words = [f"w{i}" for i in range(300)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size_words=80, overlap_words=10)

    all_words_seen = set()
    for c in chunks:
        all_words_seen.update(c.content.split())
    assert all_words_seen == set(words)  # nothing dropped between windows


def test_invalid_chunk_size_rejected():
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size_words=0)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size_words=10, overlap_words=10)
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size_words=10, overlap_words=-1)
