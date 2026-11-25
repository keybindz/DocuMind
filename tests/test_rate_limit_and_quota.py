import pytest

from app.core import plans
from app.core.plans import BucketSpec, PlanLimits, limits_for
from app.models import Tenant
from app.services.rate_limit import InMemoryTokenBucket


# ---------- limiter unit tests (fake clock, no sleeping) ----------

class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_bucket_allows_burst_then_blocks():
    clock = FakeClock()
    limiter = InMemoryTokenBucket(clock=clock)
    results = [limiter.check("k", capacity=3, refill_per_second=1) for _ in range(4)]
    assert [r.allowed for r in results] == [True, True, True, False]
    assert results[3].retry_after_seconds == pytest.approx(1.0)


def test_bucket_refills_over_time():
    clock = FakeClock()
    limiter = InMemoryTokenBucket(clock=clock)
    for _ in range(3):
        limiter.check("k", capacity=3, refill_per_second=1)
    assert limiter.check("k", capacity=3, refill_per_second=1).allowed is False

    clock.now += 2.0  # two tokens refill
    assert limiter.check("k", capacity=3, refill_per_second=1).allowed is True
    assert limiter.check("k", capacity=3, refill_per_second=1).allowed is True
    assert limiter.check("k", capacity=3, refill_per_second=1).allowed is False


def test_bucket_never_exceeds_capacity():
    clock = FakeClock()
    limiter = InMemoryTokenBucket(clock=clock)
    limiter.check("k", capacity=2, refill_per_second=1)
    clock.now += 10_000  # long idle must not bank unlimited tokens
    allowed = [limiter.check("k", capacity=2, refill_per_second=1).allowed for _ in range(3)]
    assert allowed == [True, True, False]


def test_bucket_keys_are_independent():
    limiter = InMemoryTokenBucket(clock=FakeClock())
    assert limiter.check("tenant-a", capacity=1, refill_per_second=1).allowed
    assert not limiter.check("tenant-a", capacity=1, refill_per_second=1).allowed
    assert limiter.check("tenant-b", capacity=1, refill_per_second=1).allowed


def test_unknown_plan_falls_back_to_most_restrictive():
    assert limits_for("definitely-not-a-plan") == plans.PLANS["free"]


# ---------- API-level rate limiting ----------

def _signup(client, name):
    resp = client.post(
        "/v1/auth/signup",
        json={"tenant_name": name, "email": f"o@{name.lower()}.io", "password": "supersecret123"},
    )
    return {"Authorization": f"Bearer {resp.json()['token']['access_token']}"}


def _query(client, headers):
    return client.post("/v1/query", headers=headers, json={"question": "anything"})


def test_free_plan_query_rate_limit_returns_429_with_retry_after(client):
    headers = _signup(client, "Alpha")
    burst = plans.PLANS["free"].query_bucket.capacity
    assert all(_query(client, headers).status_code == 200 for _ in range(burst))

    blocked = _query(client, headers)
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1
    assert "Rate limit exceeded" in blocked.json()["detail"]


def test_rate_limits_are_per_tenant(client):
    headers_a = _signup(client, "Alpha")
    headers_b = _signup(client, "Beta")
    burst = plans.PLANS["free"].query_bucket.capacity
    for _ in range(burst):
        _query(client, headers_a)
    assert _query(client, headers_a).status_code == 429
    # A noisy neighbor must not consume Beta's budget.
    assert _query(client, headers_b).status_code == 200


def test_rate_limit_is_shared_across_users_of_a_tenant_not_per_user(client):
    # Keyed by tenant: a tenant can't multiply its budget by adding users.
    # (Only one user per tenant exists in this phase, so assert the key shape
    # indirectly: same tenant, same bucket across separate token issuances.)
    headers = _signup(client, "Alpha")
    login = client.post("/v1/auth/login", data={"username": "o@alpha.io", "password": "supersecret123"})
    headers2 = {"Authorization": f"Bearer {login.json()['access_token']}"}
    burst = plans.PLANS["free"].query_bucket.capacity
    for _ in range(burst):
        _query(client, headers)
    assert _query(client, headers2).status_code == 429


def test_upload_rate_limit(client):
    headers = _signup(client, "Alpha")
    burst = plans.PLANS["free"].upload_bucket.capacity
    for i in range(burst):
        r = client.post("/v1/documents", headers=headers, files={"file": (f"{i}.txt", f"doc {i}".encode(), "text/plain")})
        assert r.status_code == 202
    r = client.post("/v1/documents", headers=headers, files={"file": ("x.txt", b"one too many", "text/plain")})
    assert r.status_code == 429


def test_pro_plan_gets_higher_limits(client, db_session):
    headers = _signup(client, "Alpha")
    db_session.query(Tenant).update({"plan": "pro"})
    db_session.commit()
    free_burst = plans.PLANS["free"].query_bucket.capacity
    assert all(_query(client, headers).status_code == 200 for _ in range(free_burst + 5))


# ---------- quotas ----------

@pytest.fixture()
def tiny_free_plan(monkeypatch):
    monkeypatch.setitem(
        plans.PLANS, "free", PlanLimits(max_documents=2, query_bucket=BucketSpec(100, 100), upload_bucket=BucketSpec(100, 100))
    )


def _upload(client, headers, name, text):
    return client.post("/v1/documents", headers=headers, files={"file": (name, text.encode(), "text/plain")})


def test_document_quota_blocks_upload_over_limit(client, tiny_free_plan):
    headers = _signup(client, "Alpha")
    assert _upload(client, headers, "1.txt", "one").status_code == 202
    assert _upload(client, headers, "2.txt", "two").status_code == 202
    blocked = _upload(client, headers, "3.txt", "three")
    assert blocked.status_code == 403
    assert "quota" in blocked.json()["detail"].lower()


def test_dedupe_hit_does_not_count_against_or_hit_quota(client, tiny_free_plan):
    headers = _signup(client, "Alpha")
    _upload(client, headers, "1.txt", "one")
    _upload(client, headers, "2.txt", "two")
    # At the limit, re-uploading an existing file still succeeds (it's free).
    again = _upload(client, headers, "1-again.txt", "one")
    assert again.status_code == 200


def test_deleting_a_document_frees_quota(client, tiny_free_plan):
    headers = _signup(client, "Alpha")
    first = _upload(client, headers, "1.txt", "one").json()
    _upload(client, headers, "2.txt", "two")
    assert _upload(client, headers, "3.txt", "three").status_code == 403

    assert client.delete(f"/v1/documents/{first['id']}", headers=headers).status_code == 204
    assert _upload(client, headers, "3.txt", "three").status_code == 202


def test_quotas_are_per_tenant(client, tiny_free_plan):
    headers_a = _signup(client, "Alpha")
    headers_b = _signup(client, "Beta")
    _upload(client, headers_a, "1.txt", "one")
    _upload(client, headers_a, "2.txt", "two")
    assert _upload(client, headers_a, "3.txt", "three").status_code == 403
    assert _upload(client, headers_b, "1.txt", "beta one").status_code == 202


def test_tenant_endpoint_reports_plan_and_limits(client):
    headers = _signup(client, "Alpha")
    body = client.get("/v1/tenant", headers=headers).json()
    assert body["plan"] == "free"
    assert body["max_documents"] == plans.PLANS["free"].max_documents
    assert body["query_rate_per_minute"] == plans.PLANS["free"].query_bucket.refill_per_minute
