def test_signup_creates_tenant_and_owner(client):
    resp = client.post(
        "/v1/auth/signup",
        json={"tenant_name": "Acme Corp", "email": "owner@acme.io", "password": "supersecret123"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["user"]["email"] == "owner@acme.io"
    assert body["user"]["role"] == "owner"
    assert body["token"]["access_token"]
    assert body["token"]["token_type"] == "bearer"


def test_signup_duplicate_email_rejected(client):
    payload = {"tenant_name": "Acme Corp", "email": "dup@acme.io", "password": "supersecret123"}
    first = client.post("/v1/auth/signup", json=payload)
    assert first.status_code == 201

    second = client.post("/v1/auth/signup", json={**payload, "tenant_name": "Other Co"})
    assert second.status_code == 409


def test_signup_rejects_short_password(client):
    resp = client.post(
        "/v1/auth/signup",
        json={"tenant_name": "Acme Corp", "email": "short@acme.io", "password": "short"},
    )
    assert resp.status_code == 422


def test_login_success(client):
    client.post(
        "/v1/auth/signup",
        json={"tenant_name": "Acme Corp", "email": "login@acme.io", "password": "supersecret123"},
    )
    resp = client.post(
        "/v1/auth/login",
        data={"username": "login@acme.io", "password": "supersecret123"},
    )
    assert resp.status_code == 200
    assert resp.json()["access_token"]


def test_login_wrong_password(client):
    client.post(
        "/v1/auth/signup",
        json={"tenant_name": "Acme Corp", "email": "wrongpw@acme.io", "password": "supersecret123"},
    )
    resp = client.post(
        "/v1/auth/login",
        data={"username": "wrongpw@acme.io", "password": "not-the-password"},
    )
    assert resp.status_code == 401


def test_login_unknown_user(client):
    resp = client.post(
        "/v1/auth/login",
        data={"username": "ghost@acme.io", "password": "whatever123"},
    )
    assert resp.status_code == 401


def test_protected_endpoint_requires_token(client):
    resp = client.get("/v1/documents")
    assert resp.status_code == 401


def test_protected_endpoint_rejects_garbage_token(client):
    resp = client.get("/v1/documents", headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401
