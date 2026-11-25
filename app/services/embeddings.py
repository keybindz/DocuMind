"""
Embedding backend abstraction.

Two implementations:

- SentenceTransformerBackend: real production backend. Loads
  `all-MiniLM-L6-v2` (384 dims, CPU-friendly, no API key needed) on first
  use. This is what docker-compose wires up.

- DeterministicFakeBackend: a hash-based pseudo-embedding used only in
  tests. It has no ML content whatsoever, but it IS a valid unit-length
  vector space where identical text always maps to the identical vector and
  different text maps to different vectors — which is all the chunking ->
  embedding -> vector-store -> retrieval pipeline needs in order to be
  tested deterministically, offline, and in milliseconds. It is never used
  outside of tests (see app/config.py EMBEDDING_BACKEND).
"""
import abc
import hashlib
import struct

from app.config import get_settings

settings = get_settings()


class EmbeddingBackend(abc.ABC):
    dimension: int

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Returns one unit-length vector of length `self.dimension` per input text."""


class SentenceTransformerBackend(EmbeddingBackend):
    """
    Wraps sentence-transformers/all-MiniLM-L6-v2.

    The model is lazy-loaded (not at import time) so importing this module
    never triggers a network call or a multi-hundred-MB load — only the
    first real `embed()` call does, and only when this backend is actually
    selected (production/docker, not tests).
    """

    dimension = 384

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # local import: heavy, optional dep

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


class DeterministicFakeBackend(EmbeddingBackend):
    """
    Deterministic, dependency-free stand-in used in tests. Maps each text to
    a reproducible unit vector derived from a SHA-256 hash of its content —
    same text always -> same vector, different text -> a different
    (pseudo-random, but fixed) vector. Good enough to exercise nearest
    neighbor search logic without any ML model or network access.
    """

    dimension = 32

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand the 32-byte digest into `dimension` floats in [-1, 1].
        raw = (digest * ((self.dimension // len(digest)) + 1))[: self.dimension]
        floats = [(b / 127.5) - 1.0 for b in struct.unpack(f"{self.dimension}B", raw)]
        norm = sum(f * f for f in floats) ** 0.5 or 1.0
        return [f / norm for f in floats]


_backend: EmbeddingBackend | None = None


def get_embedding_backend() -> EmbeddingBackend:
    global _backend
    if _backend is None:
        if settings.EMBEDDING_BACKEND == "sentence-transformers":
            _backend = SentenceTransformerBackend()
        elif settings.EMBEDDING_BACKEND == "fake":
            _backend = DeterministicFakeBackend()
        else:
            raise ValueError(f"Unknown EMBEDDING_BACKEND: {settings.EMBEDDING_BACKEND}")
    return _backend
