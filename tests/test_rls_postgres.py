"""
Database-layer isolation tests against a REAL Postgres with RLS enabled.

Skipped automatically unless both env vars are set (see README):

    RLS_TEST_ADMIN_URL  owner/superuser URL for a scratch database. The
                        schema in this DB is WIPED and rebuilt by the tests.
    RLS_TEST_APP_URL    URL of the restricted `documind_app` role for the
                        same database -- the role the real API connects as.

What these prove that the SQLite suite cannot: even code that forgets every
`WHERE tenant_id = ...` cannot read or write across tenants, because the
database itself refuses.
"""
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

from tests.pg_support import (  # noqa: F401 - fixtures are discovered by name in this module
    ADMIN_URL,
    APP_URL,
    TABLES,
    _alembic,
    _ctx,
    _grant_app_role,
    admin_engine,
    app_engine,
    clean_tables,
    execute,
    rows,
)

pytestmark = pytest.mark.skipif(
    not (ADMIN_URL and APP_URL), reason="set RLS_TEST_ADMIN_URL and RLS_TEST_APP_URL to run Postgres RLS tests"
)


def _insert_doc(engine, tenant, user, name, *, as_tenant=None, bypass=False):
    return execute(
        engine,
        """INSERT INTO documents (id, tenant_id, uploaded_by, filename, storage_path, mime_type,
                                  file_hash, size_bytes, status)
           VALUES (:id, :tenant, :user, :name, 'p', 'text/plain', :hash, 1, 'ready')""",
        {"id": uuid.uuid4(), "tenant": tenant, "user": user, "name": name, "hash": uuid.uuid4().hex},
        tenant=as_tenant,
        bypass=bypass,
    )


@pytest.fixture()
def seeded(app_engine):
    """Two tenants, one user + one document each, created via the system context."""
    a, b = uuid.uuid4(), uuid.uuid4()
    ua, ub = uuid.uuid4(), uuid.uuid4()
    for tid, name in ((a, "Alpha"), (b, "Beta")):
        execute(
            app_engine,
            "INSERT INTO tenants (id, name, plan, is_active) VALUES (:id, :n, 'free', true)",
            {"id": tid, "n": name},
            bypass=True,
        )
    for uid, tid, email in ((ua, a, "a@alpha.io"), (ub, b, "b@beta.io")):
        execute(
            app_engine,
            "INSERT INTO users (id, tenant_id, email, hashed_password, role) VALUES (:id, :t, :e, 'x', 'owner')",
            {"id": uid, "t": tid, "e": email},
            bypass=True,
        )
    _insert_doc(app_engine, a, ua, "alpha.txt", bypass=True)
    _insert_doc(app_engine, b, ub, "beta.txt", bypass=True)
    return {"a": a, "b": b, "ua": ua, "ub": ub}


# ---------- role sanity ----------

def test_app_role_is_not_exempt_from_rls(app_engine):
    """If the API ever connects as a superuser/BYPASSRLS role, RLS is a no-op. Guard against it."""
    (row,) = rows(app_engine, "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
    assert row.rolsuper is False
    assert row.rolbypassrls is False


def test_all_tenant_tables_have_forced_rls_and_a_policy(admin_engine):
    flags = {
        r.relname: (r.relrowsecurity, r.relforcerowsecurity)
        for r in rows(admin_engine, "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = ANY(:t)", {"t": TABLES})
    }
    assert flags == {t: (True, True) for t in TABLES}
    policies = {r.tablename for r in rows(admin_engine, "SELECT tablename FROM pg_policies")}
    assert policies == set(TABLES)


# ---------- reads ----------

def test_no_tenant_context_sees_nothing(app_engine, seeded):
    for table in TABLES:
        assert rows(app_engine, f"SELECT 1 FROM {table}") == [], f"{table} leaked rows with no tenant context"


def test_tenant_sees_only_its_own_rows(app_engine, seeded):
    docs_a = rows(app_engine, "SELECT filename FROM documents", tenant=seeded["a"])
    docs_b = rows(app_engine, "SELECT filename FROM documents", tenant=seeded["b"])
    assert [r.filename for r in docs_a] == ["alpha.txt"]
    assert [r.filename for r in docs_b] == ["beta.txt"]
    assert [r.name for r in rows(app_engine, "SELECT name FROM tenants", tenant=seeded["a"])] == ["Alpha"]
    assert [r.email for r in rows(app_engine, "SELECT email FROM users", tenant=seeded["a"])] == ["a@alpha.io"]


def test_unfiltered_select_by_known_foreign_id_returns_nothing(app_engine, seeded):
    foreign = rows(app_engine, "SELECT id FROM documents", tenant=seeded["b"])[0].id
    assert rows(app_engine, "SELECT * FROM documents WHERE id = :i", {"i": foreign}, tenant=seeded["a"]) == []


def test_garbage_tenant_setting_fails_closed(app_engine, seeded):
    with pytest.raises(DBAPIError):  # not a uuid -> cast error, never "all rows"
        rows(app_engine, "SELECT 1 FROM documents", tenant="not-a-uuid")


# ---------- writes ----------

def test_cannot_insert_rows_for_another_tenant(app_engine, seeded):
    with pytest.raises(DBAPIError, match="row-level security"):
        _insert_doc(app_engine, seeded["b"], seeded["ub"], "planted.txt", as_tenant=seeded["a"])


def test_cannot_insert_with_no_context(app_engine, seeded):
    with pytest.raises(DBAPIError, match="row-level security"):
        _insert_doc(app_engine, seeded["a"], seeded["ua"], "x.txt")


def test_cannot_update_or_delete_another_tenants_rows(app_engine, seeded):
    assert execute(app_engine, "UPDATE documents SET filename = 'pwned'", tenant=seeded["a"]) == 1  # own row only
    assert execute(app_engine, "DELETE FROM documents", tenant=seeded["a"]) == 1  # own row only
    survivors = rows(app_engine, "SELECT filename FROM documents", tenant=seeded["b"])
    assert [r.filename for r in survivors] == ["beta.txt"]  # B untouched by A's blanket UPDATE/DELETE


def test_cannot_reassign_own_row_to_another_tenant(app_engine, seeded):
    with pytest.raises(DBAPIError, match="row-level security"):
        execute(app_engine, "UPDATE documents SET tenant_id = :b", {"b": seeded["b"]}, tenant=seeded["a"])


# ---------- the ORM path the app actually uses ----------

def test_orm_query_with_no_filter_is_still_scoped(app_engine, seeded):
    """The headline guarantee: an ORM query that forgot its tenant filter cannot leak."""
    from app.core.tenancy import set_tenant
    from app.models import Document

    with Session(app_engine) as session:
        set_tenant(session, seeded["a"])
        names = [d.filename for d in session.query(Document).all()]  # deliberately NO tenant filter
    assert names == ["alpha.txt"]


def test_context_is_reapplied_after_commit_and_never_leaks_on_pooled_connection(seeded):
    """
    pool_size=1 forces the same physical connection to be reused. (1) After a
    mid-request COMMIT the tenant must still apply to the next transaction.
    (2) A later session with NO tenant must not inherit the previous one's.
    """
    from app.core.tenancy import set_tenant
    from app.models import Document

    engine = create_engine(APP_URL, pool_size=1, max_overflow=0)
    try:
        with Session(engine) as s1:
            set_tenant(s1, seeded["a"])
            assert len(s1.query(Document).all()) == 1
            s1.commit()  # ends the transaction the setting was scoped to
            assert [d.filename for d in s1.query(Document).all()] == ["alpha.txt"]  # new txn, still scoped
        with Session(engine) as s2:  # same pooled connection, no tenant set
            assert s2.query(Document).all() == []
    finally:
        engine.dispose()


# ---------- migrations on real Postgres ----------

def test_rls_migrations_downgrade_and_upgrade_roundtrip(admin_engine):
    # Two migrations add policies: the Phase 3 one (4 tables) and Phase 4's (3 new tables).
    _alembic("downgrade", "-1")
    assert len(rows(admin_engine, "SELECT 1 FROM pg_policies")) == 4
    _alembic("downgrade", "-1")
    assert len(rows(admin_engine, "SELECT 1 FROM pg_policies")) == 0
    _alembic("upgrade", "head")
    _grant_app_role(admin_engine)
    assert len(rows(admin_engine, "SELECT 1 FROM pg_policies")) == len(TABLES)


# ---------- the whole API on real Postgres with RLS active ----------

@pytest.fixture()
def pg_client(app_engine, tmp_path):
    from app.database import get_db
    from app.main import app
    from app.services.embeddings import DeterministicFakeBackend, get_embedding_backend
    from app.services.rate_limit import InMemoryTokenBucket, get_rate_limiter
    from app.services.storage import LocalDiskStorage, get_storage_backend
    from app.services.task_queue import get_task_queue
    from app.services.vector_store import InMemoryVectorStore, get_vector_store
    from types import SimpleNamespace
    from tests.helpers import FakeSender, build_queue

    SessionFactory = sessionmaker(bind=app_engine, autoflush=False)

    def override_get_db():
        session = SessionFactory()
        try:
            yield session
        finally:
            session.close()

    embedder, store, limiter = DeterministicFakeBackend(), InMemoryVectorStore(), InMemoryTokenBucket()
    storage = LocalDiskStorage(base_dir=str(tmp_path))
    queue = build_queue(
        session_factory=SessionFactory,  # the worker's sessions are tenant-scoped + RLS-enforced too
        storage=storage,
        embedder=SimpleNamespace(backend=embedder),
        vector_store=store,
        sender_holder=SimpleNamespace(sender=FakeSender()),
        eager=True,
    )
    app.dependency_overrides[get_task_queue] = lambda: queue
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_storage_backend] = lambda: storage
    app.dependency_overrides[get_embedding_backend] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_rate_limiter] = lambda: limiter
    with TestClient(app) as client:
        client.task_queue = queue
        yield client
    app.dependency_overrides.clear()


def _signup(client, name):
    r = client.post(
        "/v1/auth/signup",
        json={"tenant_name": name, "email": f"o@{name.lower()}.io", "password": "supersecret123"},
    )
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['token']['access_token']}"}


def _upload(client, headers, name, text_):
    return client.post("/v1/documents", headers=headers, files={"file": (name, text_.encode(), "text/plain")})


def test_full_api_flow_on_postgres_with_rls_is_isolated(pg_client):
    a = _signup(pg_client, "Alpha")  # signup/login run in the system context
    b = _signup(pg_client, "Beta")
    doc = _upload(pg_client, a, "a.txt", "alpha confidential roadmap")
    assert doc.status_code == 202 and doc.json()["status"] == "ready"
    doc_id = doc.json()["id"]

    assert pg_client.get(f"/v1/documents/{doc_id}", headers=a).status_code == 200
    assert pg_client.get(f"/v1/documents/{doc_id}", headers=b).status_code == 404
    assert pg_client.get("/v1/documents", headers=b).json()["total"] == 0
    leak = pg_client.post("/v1/query", headers=b, json={"question": "alpha confidential roadmap"})
    assert leak.json()["retrieved_chunks"] == []
    assert pg_client.post(
        "/v1/auth/login", data={"username": "o@alpha.io", "password": "supersecret123"}
    ).status_code == 200


def test_mid_request_commit_path_works_under_rls(pg_client):
    # A corrupt PDF commits mid-handler (status=failed) then refreshes the row;
    # that second transaction must still be tenant-scoped or it would 500/lose the row.
    a = _signup(pg_client, "Alpha")
    r = pg_client.post("/v1/documents", headers=a, files={"file": ("bad.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 202 and r.json()["status"] == "failed"


def test_rls_catches_a_route_that_forgot_its_tenant_filter(pg_client, monkeypatch):
    """
    Simulate the exact bug RLS exists for: a developer ships a lookup with no
    tenant filter. Application-layer isolation is now gone -- the database
    must still refuse.
    """
    from fastapi import HTTPException

    from app.models import Document
    from app.routers import documents as documents_router

    def buggy_lookup(db, document_id, tenant_id):  # NOTE: ignores tenant_id entirely
        doc = db.get(Document, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc

    a = _signup(pg_client, "Alpha")
    b = _signup(pg_client, "Beta")
    doc_id = _upload(pg_client, a, "a.txt", "alpha only").json()["id"]

    monkeypatch.setattr(documents_router, "_get_owned_document", buggy_lookup)
    assert pg_client.get(f"/v1/documents/{doc_id}", headers=a).status_code == 200  # owner still works
    assert pg_client.get(f"/v1/documents/{doc_id}", headers=b).status_code == 404  # RLS saved us


# ---------- Phase 4: new tables, worker sessions, and concurrency ----------

NEW_TABLES = ["webhooks", "webhook_deliveries", "idempotency_keys"]


def _seed_phase4_rows(app_engine, seeded):
    """One webhook, delivery and idempotency key per tenant, created via the system context."""
    for tenant, user_key in ((seeded["a"], "a"), (seeded["b"], "b")):
        hook_id = uuid.uuid4()
        execute(
            app_engine,
            "INSERT INTO webhooks (id, tenant_id, url, secret_encrypted, events, is_active) "
            "VALUES (:id, :t, 'https://x.example/hook', 'enc', '[]', true)",
            {"id": hook_id, "t": tenant},
            bypass=True,
        )
        execute(
            app_engine,
            "INSERT INTO webhook_deliveries (id, tenant_id, webhook_id, event_type, payload, status, attempts) "
            "VALUES (:id, :t, :h, 'document.ready', '{}', 'pending', 0)",
            {"id": uuid.uuid4(), "t": tenant, "h": hook_id},
            bypass=True,
        )
        execute(
            app_engine,
            "INSERT INTO idempotency_keys (id, tenant_id, key, fingerprint, expires_at) "
            "VALUES (:id, :t, :k, 'fp', now() + interval '1 day')",
            {"id": uuid.uuid4(), "t": tenant, "k": f"key-{user_key}"},
            bypass=True,
        )


@pytest.mark.parametrize("table", NEW_TABLES)
def test_phase4_tables_are_isolated_too(app_engine, seeded, table):
    _seed_phase4_rows(app_engine, seeded)

    assert rows(app_engine, f"SELECT 1 FROM {table}") == []  # no context -> nothing
    assert len(rows(app_engine, f"SELECT 1 FROM {table}", tenant=seeded["a"])) == 1
    assert len(rows(app_engine, f"SELECT 1 FROM {table}", tenant=seeded["b"])) == 1
    if table == "webhooks":  # deliveries reference webhooks (FK), so remove them first
        execute(app_engine, "DELETE FROM webhook_deliveries", tenant=seeded["a"])
    # A blanket DELETE as tenant A removes only A's row.
    assert execute(app_engine, f"DELETE FROM {table}", tenant=seeded["a"]) == 1
    assert len(rows(app_engine, f"SELECT 1 FROM {table}", tenant=seeded["b"])) == 1  # B untouched


def test_cannot_plant_a_webhook_in_another_tenant(app_engine, seeded):
    with pytest.raises(DBAPIError, match="row-level security"):
        execute(
            app_engine,
            "INSERT INTO webhooks (id, tenant_id, url, secret_encrypted, events, is_active) "
            "VALUES (:id, :t, 'https://evil.example/hook', 'enc', '[]', true)",
            {"id": uuid.uuid4(), "t": seeded["b"]},
            tenant=seeded["a"],  # tenant A tries to register a hook that would receive B's events
        )


def test_idempotency_keys_with_the_same_value_coexist_across_tenants(app_engine, seeded):
    for tenant in (seeded["a"], seeded["b"]):
        execute(
            app_engine,
            "INSERT INTO idempotency_keys (id, tenant_id, key, fingerprint, expires_at) "
            "VALUES (:id, :t, 'same-key', 'fp', now() + interval '1 day')",
            {"id": uuid.uuid4(), "t": tenant},
            tenant=tenant,
        )
    with pytest.raises(DBAPIError, match="uq_idempotency_tenant_key"):  # but never twice inside one tenant
        execute(
            app_engine,
            "INSERT INTO idempotency_keys (id, tenant_id, key, fingerprint, expires_at) "
            "VALUES (:id, :t, 'same-key', 'fp', now() + interval '1 day')",
            {"id": uuid.uuid4(), "t": seeded["a"]},
            tenant=seeded["a"],
        )


def test_worker_sessions_are_tenant_scoped_under_rls(app_engine, seeded):
    """The background worker has no JWT; its tenant comes from the job args. It must still be bound by RLS."""
    from app.core.tenancy import tenant_session
    from app.models import Document

    factory = sessionmaker(bind=app_engine)
    with tenant_session(factory, seeded["a"]) as db:
        assert [d.filename for d in db.query(Document).all()] == ["alpha.txt"]  # note: no tenant filter
    with tenant_session(factory, seeded["b"]) as db:
        assert [d.filename for d in db.query(Document).all()] == ["beta.txt"]
    with factory() as db:  # a worker session that forgot to set a tenant sees nothing
        assert db.query(Document).all() == []


def test_concurrent_requests_with_one_idempotency_key_create_exactly_one_document(
    pg_client, admin_engine
):
    """
    8 simultaneous retries of the same request. The unique index serializes them:
    one commits, the rest replay its result. Needs a real Postgres -- SQLite
    can't exercise this race.
    """
    from concurrent.futures import ThreadPoolExecutor

    from app.main import app

    headers = _signup(pg_client, "Alpha")
    with admin_engine.begin() as conn:  # pro plan: the default free upload burst (5) would 429 some of the 8
        conn.execute(text("UPDATE tenants SET plan = 'pro'"))
    keyed = {**headers, "Idempotency-Key": "double-click-key"}

    def attempt(_):
        with TestClient(app) as c:  # one client per thread
            return c.post("/v1/documents", headers=keyed, files={"file": ("a.txt", b"racing upload", "text/plain")})

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(attempt, range(8)))

    assert [r.status_code for r in responses] == [202] * 8, [(r.status_code, r.text) for r in responses]
    assert len({r.json()["id"] for r in responses}) == 1  # everyone got the same document
    with admin_engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM documents")).scalar() == 1
        assert conn.execute(text("SELECT count(*) FROM idempotency_keys")).scalar() == 1
    ingest_jobs = [j for j in pg_client.task_queue.executed if j[0] == "ingest"]
    assert len(ingest_jobs) == 1  # the document was queued for processing exactly once
