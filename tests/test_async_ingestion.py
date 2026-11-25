"""
The asynchronous ingestion contract. Every test here uses a LAZY queue: jobs
only run when the test calls `task_queue.drain()`, so we can assert exactly
what the API says BEFORE the worker has done anything.
"""
import uuid

import pytest

from app.models import Chunk, Document, DocumentStatus
from tests.helpers import FlakyEmbedder, HookEmbedder

pytestmark = pytest.mark.async_queue


def _signup(client, name="Alpha"):
    r = client.post(
        "/v1/auth/signup",
        json={"tenant_name": name, "email": f"o@{name.lower()}.io", "password": "supersecret123"},
    )
    return {"Authorization": f"Bearer {r.json()['token']['access_token']}"}


def _upload(client, headers, name="a.txt", text="some document content about queues", mime="text/plain", **kw):
    return client.post("/v1/documents", headers=headers, files={"file": (name, text.encode(), mime)}, **kw)


def _get(client, headers, doc_id):
    return client.get(f"/v1/documents/{doc_id}", headers=headers).json()


# ------------------------------------------------------- the async contract

def test_upload_returns_202_pending_before_any_work_happens(client, task_queue, db_session):
    headers = _signup(client)
    resp = _upload(client, headers)

    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "pending"
    assert body["attempts"] == 0
    assert body["char_count"] is None
    assert len(task_queue.jobs) == 1  # exactly one job was handed to the queue
    assert db_session.query(Chunk).count() == 0  # and nothing has been processed


def test_results_are_unavailable_until_the_worker_finishes(client, task_queue, backends):
    headers = _signup(client)
    doc_id = _upload(client, headers, text="alpha roadmap and secret plans").json()["id"]

    assert client.get(f"/v1/documents/{doc_id}/content", headers=headers).status_code == 409
    assert client.get(f"/v1/documents/{doc_id}/chunks", headers=headers).json()["total"] == 0
    q = client.post("/v1/query", headers=headers, json={"question": "alpha roadmap and secret plans"})
    assert q.json()["retrieved_chunks"] == []  # not searchable yet

    task_queue.drain()

    assert _get(client, headers, doc_id)["status"] == "ready"
    assert client.get(f"/v1/documents/{doc_id}/content", headers=headers).status_code == 200
    q = client.post("/v1/query", headers=headers, json={"question": "alpha roadmap and secret plans"})
    assert len(q.json()["retrieved_chunks"]) >= 1  # searchable now


def test_each_stage_is_committed_so_polling_sees_real_progress(client, task_queue, backends, session_factory):
    headers = _signup(client)
    seen = []

    def observe_from_another_session():
        # While the worker is inside the embedding stage, a separate session
        # (i.e. a polling API request) must already see its committed status.
        with session_factory() as other:
            seen.append(other.query(Document).one().status)

    backends.embedder.backend = HookEmbedder(observe_from_another_session)
    _upload(client, headers)
    task_queue.drain()

    assert seen == [DocumentStatus.embedding]


# ------------------------------------------------------------- idempotency

def test_redelivering_a_finished_job_is_a_noop(client, task_queue, db_session, backends):
    counter = FlakyEmbedder(failures=0)  # behaves normally; just counts embed() calls
    backends.embedder.backend = counter
    headers = _signup(client)
    doc = _upload(client, headers).json()
    task_queue.drain()
    assert counter.calls == 1
    tenant_id = db_session.query(Document).one().tenant_id

    task_queue.enqueue_ingestion(uuid.UUID(doc["id"]), tenant_id)  # at-least-once redelivery of a finished job
    task_queue.drain()

    # The guard must skip the job entirely -- not redo the work and merely
    # happen to land in the same end state.
    assert counter.calls == 1
    assert _get(client, headers, doc["id"])["status"] == "ready"


def test_rerunning_a_job_that_died_midway_never_duplicates_chunks_or_vectors(
    client, task_queue, db_session, backends
):
    """
    Worker crashed after writing chunks + vectors but before the final commit,
    then the broker redelivered the job. Simulate exactly that state and re-run.
    """
    headers = _signup(client)
    text = " ".join(f"word{i}" for i in range(500))  # several chunks
    doc = _upload(client, headers, text=text).json()
    task_queue.drain()
    tenant_id = db_session.query(Document).one().tenant_id
    chunks_before = db_session.query(Chunk).count()
    vectors_before = len(backends.vector_store._records)
    assert chunks_before > 1

    # Rewind to "crashed mid-embedding".
    db_session.query(Document).update({"status": DocumentStatus.embedding})
    db_session.commit()

    task_queue.enqueue_ingestion(uuid.UUID(doc["id"]), tenant_id)
    task_queue.drain()

    db_session.expire_all()
    assert db_session.query(Chunk).count() == chunks_before
    assert len(backends.vector_store._records) == vectors_before
    assert _get(client, headers, doc["id"])["status"] == "ready"


# ------------------------------------------------------ failure classification

def test_corrupt_file_fails_permanently_without_retrying(client, task_queue):
    headers = _signup(client)
    doc = _upload(client, headers, name="bad.pdf", text="definitely not a pdf", mime="application/pdf").json()
    task_queue.drain()

    body = _get(client, headers, doc["id"])
    assert body["status"] == "failed"
    assert body["error_message"]
    assert body["attempts"] == 1
    assert len([j for j in task_queue.executed if j[0] == "ingest"]) == 1  # ran once, never retried
    assert task_queue.retry_delays == []  # a bad file is not worth retrying


def test_transient_failure_is_retried_with_backoff_then_succeeds(client, task_queue, backends):
    headers = _signup(client)
    backends.embedder.backend = FlakyEmbedder(failures=2)
    doc = _upload(client, headers).json()
    task_queue.drain()

    body = _get(client, headers, doc["id"])
    assert body["status"] == "ready"
    assert body["attempts"] == 3
    assert body["error_message"] is None  # the transient error was cleared on success
    assert task_queue.retry_delays == [5, 10]  # exponential backoff: 5s, 10s


def test_transient_failure_gives_up_after_max_attempts_and_cleans_up(client, task_queue, backends):
    headers = _signup(client)
    backends.embedder.backend = FlakyEmbedder(failures=99)
    doc = _upload(client, headers).json()
    task_queue.drain()

    body = _get(client, headers, doc["id"])
    assert body["status"] == "failed"
    assert body["attempts"] == 4  # 1 try + 3 retries
    assert "Gave up after 4 attempts" in body["error_message"]
    assert task_queue.retry_delays == [5, 10, 20]
    assert backends.vector_store._records == []  # no half-indexed leftovers


# --------------------------------------------------- broker outage + retry API

def test_broker_outage_returns_503_and_marks_document_failed(client, task_queue):
    headers = _signup(client)
    task_queue.fail_enqueue = True

    resp = _upload(client, headers)
    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "30"

    listing = client.get("/v1/documents", headers=headers).json()
    assert listing["total"] == 1  # the upload itself was saved...
    assert listing["items"][0]["status"] == "failed"  # ...but is honestly marked failed, not stuck 'pending'


def test_retry_endpoint_recovers_a_document_after_an_outage(client, task_queue):
    headers = _signup(client)
    task_queue.fail_enqueue = True
    _upload(client, headers)
    doc_id = client.get("/v1/documents", headers=headers).json()["items"][0]["id"]

    task_queue.fail_enqueue = False
    retry = client.post(f"/v1/documents/{doc_id}/retry", headers=headers)
    assert retry.status_code == 202
    assert retry.json()["status"] == "pending"
    task_queue.drain()
    assert _get(client, headers, doc_id)["status"] == "ready"


def test_reuploading_the_same_file_after_an_outage_requeues_it(client, task_queue):
    headers = _signup(client)
    task_queue.fail_enqueue = True
    assert _upload(client, headers).status_code == 503
    task_queue.fail_enqueue = False

    resp = _upload(client, headers)  # same bytes
    assert resp.status_code == 202
    task_queue.drain()
    listing = client.get("/v1/documents", headers=headers).json()
    assert listing["total"] == 1  # no duplicate document
    assert listing["items"][0]["status"] == "ready"


def test_retry_is_only_allowed_for_failed_documents(client, task_queue):
    headers = _signup(client)
    doc_id = _upload(client, headers).json()["id"]
    assert client.post(f"/v1/documents/{doc_id}/retry", headers=headers).status_code == 409  # pending
    task_queue.drain()
    assert client.post(f"/v1/documents/{doc_id}/retry", headers=headers).status_code == 409  # ready


def test_retry_cannot_touch_another_tenants_document(client, task_queue):
    a, b = _signup(client, "Alpha"), _signup(client, "Beta")
    task_queue.fail_enqueue = True
    _upload(client, a)
    doc_id = client.get("/v1/documents", headers=a).json()["items"][0]["id"]
    task_queue.fail_enqueue = False
    assert client.post(f"/v1/documents/{doc_id}/retry", headers=b).status_code == 404


# ------------------------------------------------- deletes racing the worker

def test_deleting_a_queued_document_turns_its_job_into_a_noop(client, task_queue, backends, db_session):
    headers = _signup(client)
    doc_id = _upload(client, headers).json()["id"]
    assert client.delete(f"/v1/documents/{doc_id}", headers=headers).status_code == 204

    task_queue.drain()  # must not raise

    assert db_session.query(Chunk).count() == 0
    assert backends.vector_store._records == []


def test_document_deleted_mid_embedding_leaves_no_orphan_vectors(client, task_queue, backends, session_factory):
    headers = _signup(client)

    def delete_while_embedding():
        with session_factory() as other:  # the user deletes the document while the worker is busy
            other.query(Chunk).delete()
            other.query(Document).delete()
            other.commit()

    backends.embedder.backend = HookEmbedder(delete_while_embedding)
    _upload(client, headers)
    task_queue.drain()  # must not raise

    assert backends.vector_store._records == []
    with session_factory() as s:
        assert s.query(Document).count() == 0


# ------------------------------------------------------------ tenant scoping

def test_worker_never_mixes_tenants(client, task_queue, backends):
    a, b = _signup(client, "Alpha"), _signup(client, "Beta")
    doc_a = _upload(client, a, text="alpha only content").json()["id"]
    doc_b = _upload(client, b, text="beta only content").json()["id"]
    task_queue.drain()

    assert _get(client, a, doc_a)["status"] == "ready"
    assert client.get(f"/v1/documents/{doc_b}", headers=a).status_code == 404
    leak = client.post("/v1/query", headers=b, json={"question": "alpha only content", "top_k": 10})
    assert all(c["document_id"] == doc_b for c in leak.json()["retrieved_chunks"])
