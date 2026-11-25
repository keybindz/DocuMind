"""
End-to-end: API process  ->  Redis (Celery broker)  ->  a REAL worker process  ->  Postgres (RLS)  ->  HTTP webhook.

Nothing in the asynchronous path is faked here. The worker is a separate OS
process started with `celery worker`; it shares nothing with the test process
except Redis, Postgres, and a temp directory for uploaded files.

Skipped unless all of these are set (see README):
    RLS_TEST_ADMIN_URL, RLS_TEST_APP_URL, REDIS_TEST_URL
"""
import json
import os
import subprocess
import sys
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from tests.helpers import ReceiverServer
from tests.pg_support import (  # noqa: F401 - fixtures are discovered by name in this module
    ADMIN_URL,
    APP_URL,
    PROJECT_ROOT,
    admin_engine,
    app_engine,
    clean_tables,
)

REDIS_URL = os.environ.get("REDIS_TEST_URL")

pytestmark = pytest.mark.skipif(
    not (ADMIN_URL and APP_URL and REDIS_URL),
    reason="set RLS_TEST_ADMIN_URL, RLS_TEST_APP_URL and REDIS_TEST_URL to run Celery integration tests",
)

BROKER_URL = (REDIS_URL or "redis://localhost:6379/15").rsplit("/", 1)[0] + "/14"  # own Redis DB, away from the limiter tests


def wait_for(predicate, *, timeout=30.0, interval=0.2, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"Timed out after {timeout}s waiting for {what}")


@pytest.fixture(autouse=True)
def flush_broker():
    import redis

    redis.Redis.from_url(BROKER_URL).flushdb()
    yield
    redis.Redis.from_url(BROKER_URL).flushdb()


@pytest.fixture()
def storage_dir(tmp_path):
    return str(tmp_path / "uploads")


@pytest.fixture()
def start_worker(storage_dir, tmp_path, app_engine):
    """Factory that launches a real `celery worker` subprocess and waits until it's consuming."""
    processes = []

    def _start():
        log_path = tmp_path / f"worker-{uuid.uuid4().hex[:6]}.log"
        env = {
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT),
            "DATABASE_URL": APP_URL,  # the worker connects as the RESTRICTED role: RLS applies to it
            "CELERY_BROKER_URL": BROKER_URL,
            "STORAGE_DIR": storage_dir,
            "EMBEDDING_BACKEND": "fake",
            "EMBEDDING_DIMENSION": "32",
            "VECTOR_STORE_BACKEND": "memory",
            "AUTO_CREATE_TABLES": "false",
            "WEBHOOK_ALLOW_PRIVATE_URLS": "true",  # the receiver below lives on 127.0.0.1
        }
        env.pop("MIGRATION_DATABASE_URL", None)
        log = open(log_path, "w")
        proc = subprocess.Popen(
            [sys.executable, "-m", "celery", "-A", "app.worker.celery_app", "worker",
             "--pool=solo", "--loglevel=INFO", "-n", f"it-{uuid.uuid4().hex[:6]}@%h"],
            cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        processes.append((proc, log, log_path))

        def ready():
            assert proc.poll() is None, f"worker exited early:\n{log_path.read_text()}"
            return " ready." in log_path.read_text()

        wait_for(ready, timeout=45, what="celery worker to report ready")
        return proc

    yield _start

    for proc, log, log_path in processes:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


@pytest.fixture()
def api(app_engine, storage_dir):
    """The API process: real Postgres sessions (RLS), publishing to the real Redis broker."""
    from app.database import get_db
    from app.main import app
    from app.services.rate_limit import InMemoryTokenBucket, get_rate_limiter
    from app.services.storage import LocalDiskStorage, get_storage_backend
    from app.services.task_queue import CeleryTaskQueue, get_task_queue

    factory = sessionmaker(bind=app_engine, autoflush=False)

    def override_get_db():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    queue = CeleryTaskQueue(broker_url=BROKER_URL)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_storage_backend] = lambda: LocalDiskStorage(base_dir=storage_dir)
    app.dependency_overrides[get_task_queue] = lambda: queue
    app.dependency_overrides[get_rate_limiter] = lambda: InMemoryTokenBucket()
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture()
def receiver():
    server = ReceiverServer()
    yield server
    server.close()


def _signup(client, name="Alpha"):
    r = client.post(
        "/v1/auth/signup",
        json={"tenant_name": name, "email": f"o@{name.lower()}.io", "password": "supersecret123"},
    )
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['token']['access_token']}"}


def _upload(client, headers, name="a.txt", text="hello from the integration test", mime="text/plain"):
    r = client.post("/v1/documents", headers=headers, files={"file": (name, text.encode(), mime)})
    assert r.status_code == 202, r.text
    return r.json()


def _status(client, headers, doc_id):
    return client.get(f"/v1/documents/{doc_id}", headers=headers).json()


def test_document_flows_through_a_real_worker_and_notifies_a_real_receiver(api, start_worker, receiver):
    from app.services.webhooks import verify_signature

    start_worker()
    headers = _signup(api)
    hook = api.post("/v1/webhooks", headers=headers, json={"url": receiver.url}).json()

    words = " ".join(f"word{i}" for i in range(500))  # several chunks
    doc = _upload(api, headers, text=words)
    assert doc["status"] == "pending"  # the API answered before any processing

    # Reaching 'ready' proves the worker (a separate process, running as the
    # restricted DB role) could read this tenant's document -- i.e. it applied
    # the tenant context from the job arguments. Without it RLS would hide the
    # row, the job would no-op, and this would time out.
    final = wait_for(
        lambda: (d := _status(api, headers, doc["id"]))["status"] in ("ready", "failed") and d,
        what="document to finish processing",
    )
    assert final["status"] == "ready", final
    assert final["attempts"] == 1
    assert api.get(f"/v1/documents/{doc['id']}/chunks", headers=headers).json()["total"] >= 2

    # The worker then POSTed a signed event to our receiver, over real HTTP.
    wait_for(lambda: receiver.requests, what="webhook to arrive")
    (req,) = receiver.requests
    assert req.headers["x-documind-event"] == "document.ready"
    assert verify_signature(hook["secret"], req.body, req.headers["x-documind-signature"])
    assert json.loads(req.body)["data"]["document"]["id"] == doc["id"]

    delivery = wait_for(
        lambda: (items := api.get(f"/v1/webhooks/{hook['id']}/deliveries", headers=headers).json()["items"])
        and items[0]["status"] == "succeeded" and items[0],
        what="delivery to be recorded as succeeded",
    )
    assert delivery["attempts"] == 1


def test_a_bad_file_fails_permanently_and_fires_a_failure_webhook(api, start_worker, receiver):
    start_worker()
    headers = _signup(api)
    api.post("/v1/webhooks", headers=headers, json={"url": receiver.url})
    doc = _upload(api, headers, name="broken.pdf", text="not really a pdf", mime="application/pdf")

    final = wait_for(
        lambda: (d := _status(api, headers, doc["id"]))["status"] in ("ready", "failed") and d,
        what="document to fail",
    )
    assert final["status"] == "failed" and final["attempts"] == 1  # permanent: no retries burned
    wait_for(lambda: receiver.requests, what="failure webhook")
    assert receiver.requests[0].headers["x-documind-event"] == "document.failed"


def test_jobs_wait_in_the_broker_and_survive_until_a_worker_appears(api, start_worker):
    """Durability: the API accepted the work while NO worker was running; it isn't lost."""
    headers = _signup(api)
    doc = _upload(api, headers)
    time.sleep(1.5)
    assert _status(api, headers, doc["id"])["status"] == "pending"  # nobody has touched it

    start_worker()  # a worker comes online later and drains the backlog
    final = wait_for(
        lambda: (d := _status(api, headers, doc["id"]))["status"] in ("ready", "failed") and d,
        what="backlogged document to be processed",
    )
    assert final["status"] == "ready"


def test_two_tenants_through_one_worker_stay_isolated(api, start_worker):
    start_worker()
    a, b = _signup(api, "Alpha"), _signup(api, "Beta")
    doc_a = _upload(api, a, text="alpha's private words")
    doc_b = _upload(api, b, text="beta's private words")

    for headers, doc in ((a, doc_a), (b, doc_b)):
        wait_for(lambda h=headers, d=doc: _status(api, h, d["id"])["status"] == "ready", what="both documents ready")

    assert api.get(f"/v1/documents/{doc_b['id']}", headers=a).status_code == 404
    assert api.get(f"/v1/documents/{doc_a['id']}/chunks", headers=b).status_code == 404
    chunks_a = api.get(f"/v1/documents/{doc_a['id']}/chunks", headers=a).json()["items"]
    assert chunks_a and all("alpha" in c["content"] for c in chunks_a)  # each tenant got only its own text


def test_broker_outage_gives_a_clean_503_through_the_real_publisher(app_engine, storage_dir):
    """The real CeleryTaskQueue against a dead broker: the upload is saved, marked failed, and answered with 503."""
    from app.database import get_db
    from app.main import app
    from app.services.rate_limit import InMemoryTokenBucket, get_rate_limiter
    from app.services.storage import LocalDiskStorage, get_storage_backend
    from app.services.task_queue import CeleryTaskQueue, get_task_queue

    factory = sessionmaker(bind=app_engine, autoflush=False)

    def override_get_db():
        s = factory()
        try:
            yield s
        finally:
            s.close()

    dead_queue = CeleryTaskQueue(broker_url="redis://127.0.0.1:1/0")
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_storage_backend] = lambda: LocalDiskStorage(base_dir=storage_dir)
    app.dependency_overrides[get_task_queue] = lambda: dead_queue
    app.dependency_overrides[get_rate_limiter] = lambda: InMemoryTokenBucket()
    try:
        with TestClient(app) as client:
            headers = _signup(client)
            started = time.monotonic()
            r = client.post("/v1/documents", headers=headers, files={"file": ("a.txt", b"x", "text/plain")})
            assert r.status_code == 503 and r.headers["Retry-After"] == "30"
            assert time.monotonic() - started < 15  # failed fast; the request didn't hang
            listing = client.get("/v1/documents", headers=headers).json()
            assert listing["items"][0]["status"] == "failed"
    finally:
        app.dependency_overrides.clear()
