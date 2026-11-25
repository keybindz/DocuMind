"""
Celery tasks. Deliberately thin: each one runs a handler and translates its
JobResult into Celery's retry mechanism. All policy (what's retryable, how
long to back off, when to give up) lives in the handlers and services.
"""
import logging
import uuid

from app.config import get_settings
from app.database import SessionLocal
from app.services.embeddings import get_embedding_backend
from app.services.jobs import JobResult, exponential_backoff
from app.services.storage import get_storage_backend
from app.services.task_queue import DELIVER_TASK_NAME, INGEST_TASK_NAME, CeleryTaskQueue
from app.services.vector_store import get_vector_store
from app.services.webhooks import get_webhook_sender
from app.worker import handlers
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)
settings = get_settings()

_queue: CeleryTaskQueue | None = None


def _publisher() -> CeleryTaskQueue:
    global _queue
    if _queue is None:
        _queue = CeleryTaskQueue()
    return _queue


def _enqueue_deliveries(result: JobResult, tenant_id: uuid.UUID) -> None:
    for delivery_id in result.delivery_ids:
        try:
            _publisher().enqueue_webhook_delivery(delivery_id, tenant_id)
        except Exception:  # noqa: BLE001
            # The delivery row is already committed as 'pending' (outbox), so
            # nothing is lost -- it just isn't nudged yet. A sweeper that
            # re-enqueues stale pending rows is on the Phase 5 list.
            logger.exception("Could not enqueue webhook delivery %s", delivery_id)


@celery_app.task(bind=True, name=INGEST_TASK_NAME, max_retries=settings.INGESTION_MAX_RETRIES)
def ingest_document(self, document_id: str, tenant_id: str) -> None:
    doc_id, ten_id = uuid.UUID(document_id), uuid.UUID(tenant_id)
    attempt = self.request.retries + 1
    max_attempts = self.max_retries + 1
    try:
        result = handlers.handle_ingestion(
            session_factory=SessionLocal,
            document_id=doc_id,
            tenant_id=ten_id,
            attempt=attempt,
            max_attempts=max_attempts,
            storage=get_storage_backend(),
            embedding_backend=get_embedding_backend(),
            vector_store=get_vector_store(),
        )
    except Exception as exc:  # noqa: BLE001 - infrastructure error the handler didn't classify (e.g. DB blip)
        logger.exception("Unexpected error ingesting %s (attempt %d/%d)", doc_id, attempt, max_attempts)
        if attempt >= max_attempts:
            result = handlers.handle_unexpected_ingestion_failure(
                session_factory=SessionLocal,
                document_id=doc_id,
                tenant_id=ten_id,
                message=f"Gave up after {attempt} attempts: {exc}",
            )
        else:
            raise self.retry(exc=exc, countdown=exponential_backoff(attempt, base=5, cap=60))

    _enqueue_deliveries(result, ten_id)
    if result.retry_in is not None:
        raise self.retry(countdown=result.retry_in)


@celery_app.task(bind=True, name=DELIVER_TASK_NAME, max_retries=settings.WEBHOOK_MAX_ATTEMPTS - 1)
def deliver_webhook(self, delivery_id: str, tenant_id: str) -> None:
    attempt = self.request.retries + 1
    result = handlers.handle_webhook_delivery(
        session_factory=SessionLocal,
        delivery_id=uuid.UUID(delivery_id),
        tenant_id=uuid.UUID(tenant_id),
        attempt=attempt,
        max_attempts=self.max_retries + 1,
        sender=get_webhook_sender(),
    )
    if result.retry_in is not None:
        raise self.retry(countdown=result.retry_in)
