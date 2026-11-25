"""
Chunking service.

Splits extracted document text into overlapping windows suitable for
embedding. Word-count-based rather than character-based, since embedding
models have *token* limits and word count is a much closer proxy for token
count than character count is — good enough without pulling in a tokenizer
dependency just for chunk sizing.

Overlap matters: without it, a sentence that straddles a chunk boundary gets
split and neither half retrieves well on its own. A ~15% overlap is a
reasonable default that doesn't bloat storage/embedding cost too much.
"""
from dataclasses import dataclass


@dataclass
class TextChunk:
    index: int
    content: str
    word_count: int


def chunk_text(
    text: str,
    *,
    chunk_size_words: int = 220,
    overlap_words: int = 30,
) -> list[TextChunk]:
    """
    Splits `text` into overlapping chunks of roughly `chunk_size_words` words.

    ~220 words is a reasonable approximation of ~300 tokens for English
    prose, a common sweet spot for retrieval chunk size: big enough to carry
    context, small enough that irrelevant content doesn't dilute a match.
    """
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be >= 0 and < chunk_size_words")

    words = text.split()
    if not words:
        return []

    stride = chunk_size_words - overlap_words
    chunks: list[TextChunk] = []
    start = 0
    index = 0
    while start < len(words):
        window = words[start : start + chunk_size_words]
        content = " ".join(window)
        chunks.append(TextChunk(index=index, content=content, word_count=len(window)))
        index += 1
        if start + chunk_size_words >= len(words):
            break
        start += stride

    return chunks
