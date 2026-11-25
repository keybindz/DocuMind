"""Idempotency-Key semantics on POST /v1/documents."""
from datetime import datetime, timedelta, timezone

import pytest

from app.core import plans
from app.core.plans import BucketSpec, PlanLimits
from app.models import Document, IdempotencyKey


def _signup(client, name="Alpha"):
    r = client.post(
        "/v1/auth/signup",
        json={"tenant_name": name, "email": f"o@{name.lower()}.io", "password": "supersecret123"},
    )
    return {"Authorization": f"Bearer {r.json()['token']['access_token']}"}


def _upload(client, headers, key=None, name="a.txt", text="idempotent content"):
    h = dict(headers)
    if key is not None:
        h["Idempotency-Key"] = key
    return client.post("/v1/documents", headers=h, files={"file": (name, text.encode(), "text/plain")})


def _ingest_jobs(task_queue):
    return [j for j in task_queue.executed if j[0] == "ingest"]


def test_retry_with_same_key_returns_the_original_and_does_no_new_work(client, task_queue, db_session):
    headers = _signup(client)
    first = _upload(client, headers, key="req-1")
    second = _upload(client, headers, key="req-1")  # the client's retry after a lost response

    assert first.status_code == second.status_code == 202
    assert first.json()["id"] == second.json()["id"]
    assert "Idempotent-Replay" not in first.headers
    assert second.headers["Idempotent-Replay"] == "true"
    assert db_session.query(Document).count() == 1
    assert len(_ingest_jobs(task_queue)) == 1  # the retry did not enqueue a second job


def test_same_key_with_different_content_is_rejected(client, db_session):
    headers = _signup(client)
    _upload(client, headers, key="req-1", text="original content")
    clash = _upload(client, headers, key="req-1", text="DIFFERENT content")
    assert clash.status_code == 422
    assert "different request" in clash.json()["detail"]
    assert db_session.query(Document).count() == 1


def test_same_key_with_different_filename_is_rejected(client):
    headers = _signup(client)
    _upload(client, headers, key="req-1", name="a.txt")
    assert _upload(client, headers, key="req-1", name="renamed.txt").status_code == 422


def test_keys_are_scoped_per_tenant(client, db_session):
    a, b = _signup(client, "Alpha"), _signup(client, "Beta")
    doc_a = _upload(client, a, key="shared-key", text="alpha content")
    doc_b = _upload(client, b, key="shared-key", text="beta content")

    assert doc_a.status_code == doc_b.status_code == 202
    assert doc_a.json()["id"] != doc_b.json()["id"]
    assert "Idempotent-Replay" not in doc_b.headers  # B must never be handed A's result
    assert db_session.query(Document).count() == 2


def test_requests_without_a_key_are_independent(client, db_session):
    headers = _signup(client)
    _upload(client, headers, name="1.txt", text="one")
    _upload(client, headers, name="2.txt", text="two")
    assert db_session.query(Document).count() == 2
    assert db_session.query(IdempotencyKey).count() == 0


@pytest.mark.parametrize("bad_key", ["has space", "x" * 256, "tab\there", ""])
def test_malformed_keys_are_rejected(client, bad_key):
    headers = _signup(client)
    resp = _upload(client, headers, key=bad_key)
    assert resp.status_code == 400


def test_replay_after_the_document_was_deleted_is_treated_as_fresh(client, db_session):
    headers = _signup(client)
    first = _upload(client, headers, key="req-1").json()
    client.delete(f"/v1/documents/{first['id']}", headers=headers)

    again = _upload(client, headers, key="req-1")
    assert again.status_code == 202
    assert again.json()["id"] != first["id"]
    assert "Idempotent-Replay" not in again.headers


def test_expired_keys_are_forgotten(client, db_session):
    headers = _signup(client)
    _upload(client, headers, key="req-1", text="original content")
    db_session.query(IdempotencyKey).update({"expires_at": datetime.now(timezone.utc) - timedelta(seconds=1)})
    db_session.commit()

    # Past its TTL the key means nothing: reusing it with a different payload is fine, not a 422.
    resp = _upload(client, headers, key="req-1", text="completely different content")
    assert resp.status_code == 202
    assert db_session.query(Document).count() == 2


def test_replay_is_not_blocked_by_quota(client, monkeypatch):
    monkeypatch.setitem(
        plans.PLANS, "free", PlanLimits(1, BucketSpec(100, 100), BucketSpec(100, 100))
    )
    headers = _signup(client)
    assert _upload(client, headers, key="req-1").status_code == 202
    # The tenant is now AT its 1-document quota; a retry of the same request must still succeed.
    retry = _upload(client, headers, key="req-1")
    assert retry.status_code == 202 and retry.headers["Idempotent-Replay"] == "true"
    # ...while a genuinely new upload is blocked.
    assert _upload(client, headers, key="req-2", name="b.txt", text="new").status_code == 403


def test_key_on_a_content_dedupe_hit_still_replays_consistently(client, db_session):
    headers = _signup(client)
    first = _upload(client, headers, text="same bytes")  # no key
    keyed = _upload(client, headers, key="req-9", text="same bytes")  # dedupe hit, now keyed
    assert keyed.status_code == 200
    assert keyed.json()["id"] == first.json()["id"]

    replay = _upload(client, headers, key="req-9", text="same bytes")
    assert replay.json()["id"] == first.json()["id"]
    assert replay.headers["Idempotent-Replay"] == "true"
    assert db_session.query(Document).count() == 1


@pytest.mark.async_queue
def test_key_is_released_when_the_broker_is_down_so_the_same_key_can_retry(client, task_queue, db_session):
    headers = _signup(client)
    task_queue.fail_enqueue = True
    assert _upload(client, headers, key="req-1").status_code == 503
    assert db_session.query(IdempotencyKey).count() == 0  # not poisoned with a failed outcome

    task_queue.fail_enqueue = False
    retry = _upload(client, headers, key="req-1")  # same key, same bytes
    assert retry.status_code == 202
    task_queue.drain()
    db_session.expire_all()
    assert db_session.query(Document).one().status.value == "ready"
