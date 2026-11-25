"""
Job handlers: the functions that actually do background work.

Both runners call these -- the Celery tasks (production) and the in-memory
queue (tests) -- so the logic under test is the logic that runs in production.
Each handler opens its own short-lived, tenant-scoped session (RLS applies to
workers too) and returns a JobResult telling the runner whether to retry.
"""
import uuid

from app.core.tenancy import tenant_session
from app.services.embeddings import EmbeddingBackend
from app.services.jobs import JobResult
from app.services.pipeline import fail_document, run_ingestion_job
from app.services.storage import StorageBackend
from app.services.vector_store import VectorStore
from app.services.webhooks import WebhookSender, attempt_delivery


def handle_ingestion(
    *,
    session_factory,
    document_id: uuid.UUID,
    tenant_id: uuid.UUID,
    attempt: int,
    max_attempts: int,
    storage: StorageBackend,
    embedding_backend: EmbeddingBackend,
    vector_store: VectorStore,
) -> JobResult:
    with tenant_session(session_factory, tenant_id) as db:
        return run_ingestion_job(
            db,
            document_id=document_id,
            tenant_id=tenant_id,
            attempt=attempt,
            max_attempts=max_attempts,
            storage=storage,
            embedding_backend=embedding_backend,
            vector_store=vector_store,
        )


def handle_unexpected_ingestion_failure(*, session_factory, document_id, tenant_id, message: str) -> JobResult:
    with tenant_session(session_factory, tenant_id) as db:
        return fail_document(db, document_id, tenant_id, message)


def handle_webhook_delivery(
    *,
    session_factory,
    delivery_id: uuid.UUID,
    tenant_id: uuid.UUID,
    attempt: int,
    max_attempts: int,
    sender: WebhookSender,
) -> JobResult:
    with tenant_session(session_factory, tenant_id) as db:
        return attempt_delivery(
            db,
            delivery_id=delivery_id,
            tenant_id=tenant_id,
            attempt=attempt,
            max_attempts=max_attempts,
            sender=sender,
        )
