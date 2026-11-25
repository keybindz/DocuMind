"""
Task queue abstraction (same swappable-backend pattern as storage / embeddings /
vector store / rate limiter).

- CeleryTaskQueue: production. Publishes by TASK NAME via `send_task`, so the
  API process never imports the worker's code (or its heavy dependencies); the
  two are coupled only by a task name and its JSON arguments.

- InMemoryTaskQueue: tests. Two modes:
    * lazy (default): jobs sit in a list until the test calls `drain()`. This
      is what lets tests observe the real asynchronous contract -- upload
      returns 202 with status "pending" *before* any work has happened.
    * eager: jobs run immediately on enqueue, like Celery's
      `task_always_eager`. Used by tests that only care about the end result.
  Both modes run the same handler functions the Celery tasks run (see
  app/worker/handlers.py), so retry/idempotency behavior is the real thing,
  not a re-implementation.
"""
import abc
import logging
import uuid
from collections import deque
from typing import Callable

from app.config import get_settings
from app.services.jobs import JobResult

logger = logging.getLogger(__name__)
settings = get_settings()

INGEST_TASK_NAME = "documind.ingest_document"
DELIVER_TASK_NAME = "documind.deliver_webhook"


class QueueUnavailable(Exception):
    """The broker could not be reached; the job was NOT enqueued."""


class TaskQueue(abc.ABC):
    @abc.abstractmethod
    def enqueue_ingestion(self, document_id: uuid.UUID, tenant_id: uuid.UUID) -> None:
        ...

    @abc.abstractmethod
    def enqueue_webhook_delivery(self, delivery_id: uuid.UUID, tenant_id: uuid.UUID) -> None:
        ...


class CeleryTaskQueue(TaskQueue):
    def __init__(self, broker_url: str | None = None):
        from celery import Celery

        self._app = Celery("documind-publisher", broker=broker_url or settings.CELERY_BROKER_URL)
        # Fail fast if the broker is down instead of hanging the upload request.
        self._app.conf.broker_connection_timeout = 2
        self._retry_policy = {"max_retries": 1, "interval_start": 0, "interval_step": 0.2}

    def _publish(self, name: str, args: list) -> None:
        try:
            self._app.send_task(name, args=args, retry=True, retry_policy=self._retry_policy)
        except Exception as exc:  # noqa: BLE001 - kombu/redis raise a variety of connection errors
            raise QueueUnavailable(f"Could not publish {name}: {exc}") from exc

    def enqueue_ingestion(self, document_id: uuid.UUID, tenant_id: uuid.UUID) -> None:
        self._publish(INGEST_TASK_NAME, [str(document_id), str(tenant_id)])

    def enqueue_webhook_delivery(self, delivery_id: uuid.UUID, tenant_id: uuid.UUID) -> None:
        self._publish(DELIVER_TASK_NAME, [str(delivery_id), str(tenant_id)])


class InMemoryTaskQueue(TaskQueue):
    def __init__(
        self,
        *,
        ingest_handler: Callable[[uuid.UUID, uuid.UUID, int], JobResult],
        webhook_handler: Callable[[uuid.UUID, uuid.UUID, int], JobResult],
        eager: bool = False,
        max_ingest_attempts: int | None = None,
        max_delivery_attempts: int | None = None,
    ):
        self._ingest_handler = ingest_handler
        self._webhook_handler = webhook_handler
        self.eager = eager
        self.max_ingest_attempts = max_ingest_attempts or settings.INGESTION_MAX_RETRIES + 1
        self.max_delivery_attempts = max_delivery_attempts or settings.WEBHOOK_MAX_ATTEMPTS
        self.jobs: deque = deque()
        self.fail_enqueue = False  # test hook: simulate the broker being down
        self.retry_delays: list[float] = []  # what the handlers asked for (not actually slept)
        self.executed: list[tuple] = []
        self._draining = False

    # -- enqueue --
    def enqueue_ingestion(self, document_id, tenant_id) -> None:
        self._enqueue(("ingest", document_id, tenant_id, 1))

    def enqueue_webhook_delivery(self, delivery_id, tenant_id) -> None:
        self._enqueue(("deliver", delivery_id, tenant_id, 1))

    def _enqueue(self, job: tuple) -> None:
        if self.fail_enqueue:
            raise QueueUnavailable("simulated broker outage")
        self.jobs.append(job)
        if self.eager:
            self.drain()

    # -- execution --
    def drain(self, max_jobs: int = 500) -> int:
        """Run queued jobs (including retries and follow-up deliveries) until empty."""
        if self._draining:  # re-entrant call from an eager enqueue inside a handler
            return 0
        self._draining = True
        count = 0
        try:
            while self.jobs and count < max_jobs:
                kind, entity_id, tenant_id, attempt = self.jobs.popleft()
                count += 1
                self.executed.append((kind, entity_id, attempt))
                if kind == "ingest":
                    result = self._ingest_handler(entity_id, tenant_id, attempt)
                    limit = self.max_ingest_attempts
                else:
                    result = self._webhook_handler(entity_id, tenant_id, attempt)
                    limit = self.max_delivery_attempts
                for delivery_id in result.delivery_ids:
                    self.jobs.append(("deliver", delivery_id, tenant_id, 1))
                if result.retry_in is not None:
                    assert attempt < limit, "handler asked to retry past its own attempt limit"
                    self.retry_delays.append(result.retry_in)
                    self.jobs.append((kind, entity_id, tenant_id, attempt + 1))
        finally:
            self._draining = False
        return count


_queue: TaskQueue | None = None


def get_task_queue() -> TaskQueue:
    global _queue
    if _queue is None:
        if settings.TASK_QUEUE_BACKEND == "celery":
            _queue = CeleryTaskQueue()
        else:
            raise ValueError(
                f"TASK_QUEUE_BACKEND={settings.TASK_QUEUE_BACKEND!r} has no process-wide default; "
                "the in-memory queue is only constructed by tests."
            )
    return _queue
