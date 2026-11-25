"""
Test fixtures.

Everything external is replaced by an in-process stand-in so `pytest` needs no
services: SQLite (in-memory), a deterministic fake embedder, an in-memory
vector store, an in-memory token bucket, an in-memory task queue, and a fake
webhook sender. The stand-ins implement the SAME interfaces as the production
backends, and the task queue runs the SAME worker handlers production runs.

Queue modes:
  * default  -> EAGER: jobs run the moment they're enqueued, so tests about
                other features (auth, search, isolation...) can upload and
                immediately query, like Celery's task_always_eager.
  * @pytest.mark.async_queue -> LAZY: jobs wait until `task_queue.drain()`,
                so a test can assert what the API returns BEFORE the worker ran.
"""
import shutil
import tempfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.database import Base, get_db
from app.main import app
from app.services.embeddings import DeterministicFakeBackend, get_embedding_backend
from app.services.rate_limit import InMemoryTokenBucket, get_rate_limiter
from app.services.storage import LocalDiskStorage, get_storage_backend
from app.services.task_queue import get_task_queue
from app.services.vector_store import InMemoryVectorStore, get_vector_store
from tests.helpers import FakeSender, build_queue


@pytest.fixture(autouse=True)
def allow_private_webhook_urls(monkeypatch):
    # Most tests register http://localhost-style receivers; SSRF-specific tests
    # flip this back to False explicitly.
    monkeypatch.setattr(get_settings(), "WEBHOOK_ALLOW_PRIVATE_URLS", True)


@pytest.fixture()
def tmp_storage_dir():
    path = tempfile.mkdtemp(prefix="documind_test_storage_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)
    eng.dispose()


@pytest.fixture()
def session_factory(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def db_session(session_factory):
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def backends(tmp_storage_dir):
    """The fake infrastructure shared by the API and the (in-memory) worker."""
    return SimpleNamespace(
        storage=LocalDiskStorage(base_dir=tmp_storage_dir),
        embedder=SimpleNamespace(backend=DeterministicFakeBackend()),  # holder: tests may swap in a FlakyEmbedder
        vector_store=InMemoryVectorStore(),
        limiter=InMemoryTokenBucket(),
        sender_holder=SimpleNamespace(sender=FakeSender()),
    )


@pytest.fixture()
def webhook_sender(backends):
    return backends.sender_holder.sender


@pytest.fixture()
def task_queue(request, session_factory, backends):
    eager = request.node.get_closest_marker("async_queue") is None
    return build_queue(
        session_factory=session_factory,
        storage=backends.storage,
        embedder=backends.embedder,
        vector_store=backends.vector_store,
        sender_holder=backends.sender_holder,
        eager=eager,
    )


@pytest.fixture()
def client(db_session, backends, task_queue):
    def override_get_db():
        # The worker commits through its own sessions; make sure this shared
        # session never serves stale identity-mapped rows from an earlier request.
        db_session.expire_all()
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_storage_backend] = lambda: backends.storage
    app.dependency_overrides[get_embedding_backend] = lambda: backends.embedder.backend
    app.dependency_overrides[get_vector_store] = lambda: backends.vector_store
    app.dependency_overrides[get_rate_limiter] = lambda: backends.limiter
    app.dependency_overrides[get_task_queue] = lambda: task_queue
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def auth_headers(client):
    """Signs up a fresh tenant/user and returns ready-to-use Authorization headers."""
    resp = client.post(
        "/v1/auth/signup",
        json={"tenant_name": "Acme Corp", "email": "owner@acme.io", "password": "supersecret123"},
    )
    assert resp.status_code == 201, resp.text
    token = resp.json()["token"]["access_token"]
    return {"Authorization": f"Bearer {token}"}
