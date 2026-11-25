"""
Vector store abstraction.

Same pattern as storage.py and embeddings.py: a real backend (Qdrant, via
docker-compose) and an in-memory fake used only in tests, so the retrieval
pipeline is fully testable without a running Qdrant instance.

Payloads are deliberately minimal — tenant_id, document_id, chunk_id — never
chunk text. Chunk content lives in Postgres (see models.Chunk) and is
fetched by id after a vector search returns its matches. This keeps the
vector store from becoming a second source of truth for content, and keeps
payloads small regardless of chunk size.

Tenant isolation: every search() call requires a tenant_id and it is always
applied as a hard filter, never left optional — there's no code path that
can run a cross-tenant vector search by accident.
"""
import abc
import math
import uuid
from dataclasses import dataclass

from app.config import get_settings

settings = get_settings()


@dataclass
class VectorRecord:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    tenant_id: uuid.UUID
    vector: list[float]


@dataclass
class SearchResult:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    score: float


class VectorStore(abc.ABC):
    @abc.abstractmethod
    def upsert(self, records: list[VectorRecord]) -> None:
        ...

    @abc.abstractmethod
    def search(
        self,
        *,
        tenant_id: uuid.UUID,
        query_vector: list[float],
        top_k: int = 5,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[SearchResult]:
        ...

    @abc.abstractmethod
    def delete_by_document(self, *, tenant_id: uuid.UUID, document_id: uuid.UUID) -> None:
        ...


class QdrantVectorStore(VectorStore):
    """Real backend. Lazy-imports qdrant_client so it's only required when actually used."""

    def __init__(self, collection_name: str = "chunks"):
        self.collection_name = collection_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels

            self._client = QdrantClient(url=settings.QDRANT_URL)
            self._qmodels = qmodels
            existing = [c.name for c in self._client.get_collections().collections]
            if self.collection_name not in existing:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=settings.EMBEDDING_DIMENSION, distance=qmodels.Distance.COSINE
                    ),
                )
            # Every search filters on tenant_id (and often document_id), so
            # index both; without payload indexes Qdrant filters by scanning.
            for field in ("tenant_id", "document_id"):
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
        return self._client

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        client = self._get_client()
        points = [
            self._qmodels.PointStruct(
                id=str(r.chunk_id),
                vector=r.vector,
                payload={
                    "tenant_id": str(r.tenant_id),
                    "document_id": str(r.document_id),
                },
            )
            for r in records
        ]
        client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        *,
        tenant_id: uuid.UUID,
        query_vector: list[float],
        top_k: int = 5,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[SearchResult]:
        client = self._get_client()
        must = [self._qmodels.FieldCondition(key="tenant_id", match=self._qmodels.MatchValue(value=str(tenant_id)))]
        if document_ids:
            must.append(
                self._qmodels.FieldCondition(
                    key="document_id",
                    match=self._qmodels.MatchAny(any=[str(d) for d in document_ids]),
                )
            )
        hits = client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=self._qmodels.Filter(must=must),
            limit=top_k,
        )
        return [
            SearchResult(
                chunk_id=uuid.UUID(str(h.id)),
                document_id=uuid.UUID(h.payload["document_id"]),
                score=h.score,
            )
            for h in hits
        ]

    def delete_by_document(self, *, tenant_id: uuid.UUID, document_id: uuid.UUID) -> None:
        client = self._get_client()
        client.delete(
            collection_name=self.collection_name,
            points_selector=self._qmodels.FilterSelector(
                filter=self._qmodels.Filter(
                    must=[
                        self._qmodels.FieldCondition(
                            key="tenant_id", match=self._qmodels.MatchValue(value=str(tenant_id))
                        ),
                        self._qmodels.FieldCondition(
                            key="document_id", match=self._qmodels.MatchValue(value=str(document_id))
                        ),
                    ]
                )
            ),
        )


class InMemoryVectorStore(VectorStore):
    """
    Brute-force cosine-similarity search over a plain Python list. O(n) per
    query, which is exactly fine for a test suite with dozens of vectors and
    completely wrong for production — that trade-off is the point: it needs
    zero infrastructure and zero network access, so tests can fully exercise
    tenant-filtered retrieval without a real Qdrant instance.
    """

    def __init__(self):
        self._records: list[VectorRecord] = []

    def upsert(self, records: list[VectorRecord]) -> None:
        existing_ids = {r.chunk_id for r in records}
        self._records = [r for r in self._records if r.chunk_id not in existing_ids] + list(records)

    def search(
        self,
        *,
        tenant_id: uuid.UUID,
        query_vector: list[float],
        top_k: int = 5,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[SearchResult]:
        candidates = [r for r in self._records if r.tenant_id == tenant_id]
        if document_ids:
            doc_id_set = set(document_ids)
            candidates = [r for r in candidates if r.document_id in doc_id_set]

        scored = [
            SearchResult(chunk_id=r.chunk_id, document_id=r.document_id, score=_cosine(query_vector, r.vector))
            for r in candidates
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def delete_by_document(self, *, tenant_id: uuid.UUID, document_id: uuid.UUID) -> None:
        self._records = [
            r for r in self._records if not (r.tenant_id == tenant_id and r.document_id == document_id)
        ]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        if settings.VECTOR_STORE_BACKEND == "qdrant":
            _store = QdrantVectorStore()
        elif settings.VECTOR_STORE_BACKEND == "memory":
            _store = InMemoryVectorStore()
        else:
            raise ValueError(f"Unknown VECTOR_STORE_BACKEND: {settings.VECTOR_STORE_BACKEND}")
    return _store
