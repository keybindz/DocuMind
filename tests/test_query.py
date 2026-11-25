def _upload_text(client, headers, filename: str, content: str):
    resp = client.post(
        "/v1/documents",
        headers=headers,
        files={"file": (filename, content.encode(), "text/plain")},
    )
    assert resp.status_code == 202, resp.text
    return resp.json()


def test_upload_indexes_document_into_chunks(client, auth_headers):
    long_text = " ".join([f"sentence{i} about backend systems" for i in range(400)])
    doc = _upload_text(client, auth_headers, "big.txt", long_text)
    assert doc["status"] == "ready"

    chunks_resp = client.get(f"/v1/documents/{doc['id']}/chunks", headers=auth_headers)
    assert chunks_resp.status_code == 200
    body = chunks_resp.json()
    assert body["total"] >= 1
    assert all(item["embedding_status"] == "embedded" for item in body["items"])
    # Chunk indices are sequential starting at 0.
    assert [item["chunk_index"] for item in body["items"]] == list(range(len(body["items"])))


def test_short_document_still_gets_one_chunk(client, auth_headers):
    doc = _upload_text(client, auth_headers, "small.txt", "A very short note about Qdrant.")
    assert doc["status"] == "ready"
    chunks = client.get(f"/v1/documents/{doc['id']}/chunks", headers=auth_headers).json()
    assert chunks["total"] == 1


def test_query_retrieves_relevant_chunk(client, auth_headers):
    """
    Verifies the full /v1/query pipeline end-to-end: embed question -> vector
    search -> fetch chunk content from Postgres -> rank by score.

    The test fixtures use DeterministicFakeBackend (see services/embeddings.py),
    which is hash-based and intentionally has NO semantic understanding —
    it guarantees identical text maps to identical vectors, nothing more.
    So rather than asserting that "database" content outranks "cooking"
    content for a database *question* (which would only pass by fake-backend
    coincidence and prove nothing), we query with the exact text of one
    chunk. That guarantees a cosine-similarity-1.0 match against its own
    vector, deterministically exercising every stage of the pipeline — end
    to end ranking correctness against real semantic similarity is exercised
    separately using the real sentence-transformers backend (see README).
    """
    cooking_text = "This document is about baking sourdough bread at home, including starter maintenance."
    db_text = "This document explains vector databases, embeddings, and semantic search for retrieval."
    _upload_text(client, auth_headers, "cooking.txt", cooking_text)
    _upload_text(client, auth_headers, "databases.txt", db_text)

    resp = client.post("/v1/query", headers=auth_headers, json={"question": db_text, "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieved_chunks"], "expected at least one retrieved chunk"
    top = body["retrieved_chunks"][0]
    assert top["document_filename"] == "databases.txt"
    assert top["score"] > 0.99  # near-exact match since the fake backend hashes identical text identically


def test_query_without_llm_key_returns_no_synthesized_answer(client, auth_headers):
    # The test app never sets ANTHROPIC_API_KEY, so synthesis should be
    # skipped gracefully rather than erroring, while retrieval still works.
    _upload_text(client, auth_headers, "doc.txt", "Some content about distributed systems and consensus.")
    resp = client.post("/v1/query", headers=auth_headers, json={"question": "distributed systems consensus"})
    body = resp.json()
    assert resp.status_code == 200
    assert body["answer_synthesized"] is False
    assert body["answer"] is None
    assert len(body["retrieved_chunks"]) >= 1


def test_query_with_no_documents_returns_empty_gracefully(client, auth_headers):
    resp = client.post("/v1/query", headers=auth_headers, json={"question": "anything at all"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieved_chunks"] == []
    assert body["answer_synthesized"] is False


def test_query_document_ids_filter_restricts_retrieval(client, auth_headers):
    doc_a = _upload_text(client, auth_headers, "a.txt", "Alpha document about kubernetes autoscaling policies.")
    doc_b = _upload_text(client, auth_headers, "b.txt", "Beta document about kubernetes autoscaling policies too.")

    resp = client.post(
        "/v1/query",
        headers=auth_headers,
        json={"question": "kubernetes autoscaling", "document_ids": [doc_a["id"]], "top_k": 5},
    )
    body = resp.json()
    assert all(c["document_id"] == doc_a["id"] for c in body["retrieved_chunks"])
    assert not any(c["document_id"] == doc_b["id"] for c in body["retrieved_chunks"])


def test_query_requires_auth(client):
    resp = client.post("/v1/query", json={"question": "anything"})
    assert resp.status_code == 401


def test_query_rejects_empty_question(client, auth_headers):
    resp = client.post("/v1/query", headers=auth_headers, json={"question": ""})
    assert resp.status_code == 422


def test_query_tenant_isolation(client):
    signup_a = client.post(
        "/v1/auth/signup",
        json={"tenant_name": "Tenant A", "email": "a@tenanta.io", "password": "supersecret123"},
    )
    headers_a = {"Authorization": f"Bearer {signup_a.json()['token']['access_token']}"}
    _upload_text(client, headers_a, "secret.txt", "Tenant A's confidential roadmap and financial projections.")

    signup_b = client.post(
        "/v1/auth/signup",
        json={"tenant_name": "Tenant B", "email": "b@tenantb.io", "password": "supersecret123"},
    )
    headers_b = {"Authorization": f"Bearer {signup_b.json()['token']['access_token']}"}

    # Tenant B queries for content that only exists in tenant A's corpus —
    # must get zero results, never a cross-tenant leak.
    resp = client.post(
        "/v1/query",
        headers=headers_b,
        json={"question": "confidential roadmap and financial projections", "top_k": 5},
    )
    assert resp.json()["retrieved_chunks"] == []


def test_delete_document_removes_its_chunks(client, auth_headers):
    doc = _upload_text(client, auth_headers, "todelete.txt", "Content that will be deleted shortly after upload.")
    doc_id = doc["id"]

    chunks_before = client.get(f"/v1/documents/{doc_id}/chunks", headers=auth_headers).json()
    assert chunks_before["total"] >= 1

    delete_resp = client.delete(f"/v1/documents/{doc_id}", headers=auth_headers)
    assert delete_resp.status_code == 204

    # Document (and thus its chunks) is gone — route now 404s.
    chunks_after = client.get(f"/v1/documents/{doc_id}/chunks", headers=auth_headers)
    assert chunks_after.status_code == 404
