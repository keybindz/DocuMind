"""
Adversarial isolation tests: each one *attempts* a cross-tenant breach and
asserts it fails. These run on SQLite, so they exercise the application-layer
defenses. tests/test_rls_postgres.py re-proves the database-layer defense
(Row-Level Security) against a real Postgres.
"""
import uuid

from jose import jwt

from app.config import get_settings
from app.core.security import create_access_token
from app.database import get_system_db
from app.main import app


def _signup(client, name: str):
    resp = client.post(
        "/v1/auth/signup",
        json={"tenant_name": name, "email": f"owner@{name.lower().replace(' ', '')}.io", "password": "supersecret123"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    headers = {"Authorization": f"Bearer {body['token']['access_token']}"}
    return headers, body["user"]


def _upload(client, headers, name, text):
    resp = client.post("/v1/documents", headers=headers, files={"file": (name, text.encode(), "text/plain")})
    assert resp.status_code == 202, resp.text
    return resp.json()


def test_every_document_endpoint_is_closed_to_other_tenants(client):
    headers_a, _ = _signup(client, "Alpha Corp")
    headers_b, _ = _signup(client, "Beta Corp")
    doc = _upload(client, headers_a, "secret.txt", "alpha's private strategy memo")
    doc_id = doc["id"]

    # Tenant B probes every route that takes a document id. All must look
    # identical to "does not exist" (404), never 403 and never data.
    assert client.get(f"/v1/documents/{doc_id}", headers=headers_b).status_code == 404
    assert client.get(f"/v1/documents/{doc_id}/content", headers=headers_b).status_code == 404
    assert client.get(f"/v1/documents/{doc_id}/chunks", headers=headers_b).status_code == 404
    assert client.delete(f"/v1/documents/{doc_id}", headers=headers_b).status_code == 404

    # ...and the attempted delete must not have deleted anything.
    still_there = client.get(f"/v1/documents/{doc_id}", headers=headers_a)
    assert still_there.status_code == 200
    assert client.get(f"/v1/documents/{doc_id}/chunks", headers=headers_a).json()["total"] >= 1


def test_query_cannot_target_another_tenants_documents(client):
    headers_a, _ = _signup(client, "Alpha Corp")
    headers_b, _ = _signup(client, "Beta Corp")
    doc = _upload(client, headers_a, "secret.txt", "alpha's private strategy memo")

    resp = client.post(
        "/v1/query",
        headers=headers_b,
        json={"question": "alpha's private strategy memo", "document_ids": [doc["id"]], "top_k": 10},
    )
    assert resp.status_code == 200
    assert resp.json()["retrieved_chunks"] == []


def test_dedupe_never_returns_another_tenants_document(client):
    headers_a, _ = _signup(client, "Alpha Corp")
    headers_b, _ = _signup(client, "Beta Corp")
    same_text = "identical bytes uploaded by two different companies"

    doc_a = _upload(client, headers_a, "same.txt", same_text)
    doc_b = _upload(client, headers_b, "same.txt", same_text)

    # If dedupe were global, B would be handed A's document row.
    assert doc_a["id"] != doc_b["id"]
    assert client.get(f"/v1/documents/{doc_b['id']}", headers=headers_a).status_code == 404


def test_token_with_mismatched_tenant_claim_is_rejected(client):
    """
    Simulates a bug (or compromised issuer) that mints a validly-signed token
    pairing user A with tenant B. The user lookup must not resolve.
    """
    headers_a, user_a = _signup(client, "Alpha Corp")
    _, user_b = _signup(client, "Beta Corp")
    _upload(client, headers_a, "a.txt", "alpha content")

    forged = create_access_token(user_id=uuid.UUID(user_a["id"]), tenant_id=uuid.UUID(user_b["tenant_id"]))
    resp = client.get("/v1/documents", headers={"Authorization": f"Bearer {forged}"})
    assert resp.status_code == 401


def test_token_without_tenant_claim_is_rejected(client):
    _, user_a = _signup(client, "Alpha Corp")
    settings = get_settings()
    legacy_token = jwt.encode({"sub": user_a["id"]}, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    resp = client.get("/v1/documents", headers={"Authorization": f"Bearer {legacy_token}"})
    assert resp.status_code == 401


def test_token_for_nonexistent_user_is_rejected(client):
    token = create_access_token(user_id=uuid.uuid4(), tenant_id=uuid.uuid4())
    resp = client.get("/v1/documents", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_tenant_usage_endpoint_only_reports_own_data(client):
    headers_a, _ = _signup(client, "Alpha Corp")
    headers_b, _ = _signup(client, "Beta Corp")
    _upload(client, headers_a, "1.txt", "one")
    _upload(client, headers_a, "2.txt", "two")

    a = client.get("/v1/tenant", headers=headers_a).json()
    b = client.get("/v1/tenant", headers=headers_b).json()
    assert a["name"] == "Alpha Corp" and a["document_count"] == 2
    assert b["name"] == "Beta Corp" and b["document_count"] == 0
    assert a["tenant_id"] != b["tenant_id"]


def _all_dependency_callables(dependant):
    for dep in dependant.dependencies:
        yield dep.call
        yield from _all_dependency_callables(dep)


def test_system_context_only_on_auth_routes(client):
    """
    The RLS bypass ("system context") exists solely so signup/login can find
    users before a tenant is known. If any route that serves tenant data ever
    depends on get_system_db, isolation is silently void for that route --
    so this test makes that a build failure.
    """
    offenders = []
    for route in app.routes:
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            continue
        uses_system_db = get_system_db in set(_all_dependency_callables(dependant))
        if uses_system_db and not route.path.startswith("/v1/auth/"):
            offenders.append(route.path)
    assert offenders == [], f"Routes outside /v1/auth must not use the RLS bypass: {offenders}"
