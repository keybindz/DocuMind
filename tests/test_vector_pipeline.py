import uuid

from app.services.embeddings import DeterministicFakeBackend
from app.services.vector_store import InMemoryVectorStore, VectorRecord


def test_fake_embedding_is_deterministic():
    backend = DeterministicFakeBackend()
    v1 = backend.embed(["hello world"])[0]
    v2 = backend.embed(["hello world"])[0]
    assert v1 == v2


def test_fake_embedding_differs_for_different_text():
    backend = DeterministicFakeBackend()
    v1 = backend.embed(["cats are great"])[0]
    v2 = backend.embed(["quarterly revenue report"])[0]
    assert v1 != v2


def test_fake_embedding_vectors_are_unit_length():
    backend = DeterministicFakeBackend()
    vectors = backend.embed(["some text", "other text", ""])
    for v in vectors:
        norm = sum(x * x for x in v) ** 0.5
        assert abs(norm - 1.0) < 1e-6


def test_vector_store_returns_most_similar_first():
    store = InMemoryVectorStore()
    tenant = uuid.uuid4()
    doc = uuid.uuid4()
    backend = DeterministicFakeBackend()

    texts = ["semantic search with vector databases", "a recipe for chocolate cake", "quarterly earnings report"]
    vectors = backend.embed(texts)
    chunk_ids = [uuid.uuid4() for _ in texts]
    store.upsert(
        [
            VectorRecord(chunk_id=cid, document_id=doc, tenant_id=tenant, vector=v)
            for cid, v in zip(chunk_ids, vectors)
        ]
    )

    # Querying with the exact embedding of the first text should rank it first.
    [query_vector] = backend.embed([texts[0]])
    results = store.search(tenant_id=tenant, query_vector=query_vector, top_k=3)

    assert results[0].chunk_id == chunk_ids[0]
    assert results[0].score >= results[1].score >= results[2].score


def test_vector_store_tenant_isolation():
    store = InMemoryVectorStore()
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
    backend = DeterministicFakeBackend()

    [vec_a] = backend.embed(["tenant A confidential content"])
    [vec_b] = backend.embed(["tenant B confidential content"])
    store.upsert(
        [
            VectorRecord(chunk_id=uuid.uuid4(), document_id=doc_a, tenant_id=tenant_a, vector=vec_a),
            VectorRecord(chunk_id=uuid.uuid4(), document_id=doc_b, tenant_id=tenant_b, vector=vec_b),
        ]
    )

    results_for_a = store.search(tenant_id=tenant_a, query_vector=vec_b, top_k=10)
    assert all(True for _ in results_for_a)  # sanity
    assert len(results_for_a) == 1
    assert results_for_a[0].document_id == doc_a  # tenant A only ever sees tenant A's vectors,
    # even when searching with a vector that's actually closer to tenant B's content


def test_vector_store_document_id_filter():
    store = InMemoryVectorStore()
    tenant = uuid.uuid4()
    doc_1, doc_2 = uuid.uuid4(), uuid.uuid4()
    backend = DeterministicFakeBackend()

    [v1] = backend.embed(["content in document one"])
    [v2] = backend.embed(["content in document two"])
    store.upsert(
        [
            VectorRecord(chunk_id=uuid.uuid4(), document_id=doc_1, tenant_id=tenant, vector=v1),
            VectorRecord(chunk_id=uuid.uuid4(), document_id=doc_2, tenant_id=tenant, vector=v2),
        ]
    )

    results = store.search(tenant_id=tenant, query_vector=v2, top_k=10, document_ids=[doc_1])
    assert all(r.document_id == doc_1 for r in results)


def test_vector_store_delete_by_document_removes_only_that_documents_vectors():
    store = InMemoryVectorStore()
    tenant = uuid.uuid4()
    doc_1, doc_2 = uuid.uuid4(), uuid.uuid4()
    backend = DeterministicFakeBackend()

    [v1] = backend.embed(["doc one"])
    [v2] = backend.embed(["doc two"])
    store.upsert(
        [
            VectorRecord(chunk_id=uuid.uuid4(), document_id=doc_1, tenant_id=tenant, vector=v1),
            VectorRecord(chunk_id=uuid.uuid4(), document_id=doc_2, tenant_id=tenant, vector=v2),
        ]
    )

    store.delete_by_document(tenant_id=tenant, document_id=doc_1)
    results = store.search(tenant_id=tenant, query_vector=v1, top_k=10)
    assert all(r.document_id != doc_1 for r in results)
    assert any(r.document_id == doc_2 for r in results)
