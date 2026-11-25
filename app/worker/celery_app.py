"""
Celery application for the background worker.

Run with:  celery -A app.worker.celery_app worker --loglevel=info

Reliability settings, and why:

- task_acks_late + task_reject_on_worker_lost: a task is acknowledged only
  AFTER it finishes, and is redelivered if the worker process dies mid-task.
  That gives at-least-once delivery -- which is why every handler is
  idempotent (see services/pipeline.py).
- worker_prefetch_multiplier=1: a worker takes one task at a time. With long
  ingestion jobs and acks_late, prefetching would let one slow job strand a
  queue of tasks behind it on a single worker.
- visibility_timeout (Redis transport): if a task isn't acked within this
  window Redis redelivers it to another worker. It must exceed the longest
  retry countdown (webhook backoff is capped at 1h), or delayed tasks would
  be delivered twice.
- No result backend: job state lives in Postgres (document.status,
  webhook_deliveries), which is queryable and tenant-scoped. A result backend
  would be a second, unscoped source of truth.
"""
from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery("documind", broker=settings.CELERY_BROKER_URL, include=["app.worker.tasks"])
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    task_ignore_result=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": 7200},
    timezone="UTC",
)
